"""
CF-HPINO research training loop.

Features:
  - Train / validation / test splits (held-out parameters)
  - Checkpoint on validation relative L2 (not training loss only)
  - Per-epoch CSV logs + JSON experiment manifest
  - Optional AMP (mixed precision on CUDA)
  - Adam + cosine LR + optional L-BFGS polish
  - Curriculum across BS → fractional BS → Merton
  - Resume from checkpoint
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from ..cf_hpino_loss import CFHPINOLoss, LossConfig, LossPDEType
from ..cf_hpino_model import CF_HPINO, build_cf_hpino
from ..data import DatasetConfig, OptionPricingDataset, PricingModel
from ..data.sampling import collate_option_batch
from ..eval.metrics import relative_l2
from ..model_utils import ModelEMA

if TYPE_CHECKING:
    pass


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lbfgs_max_iter: int = 20
    use_lbfgs: bool = True
    lbfgs_start_epoch: int = 50
    patience: int = 15
    min_epochs: int = 10
    scheduler: str = "cosine"  # cosine | plateau
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    num_workers: int = 0
    curriculum: List[str] = field(
        default_factory=lambda: ["black_scholes", "fractional_bs", "merton"]
    )
    epochs_per_stage: int = 30
    dataset: Optional[DatasetConfig] = None
    # Research protocol
    val_fraction: float = 0.15
    test_fraction: float = 0.10
    save_every: int = 10
    use_amp: bool = True
    grad_clip: float = 1.0
    monitor: str = "val_rel_l2"  # val_rel_l2 | val_loss
    experiment_name: str = "experiment"
    resume: Optional[str] = None
    run_test_after_train: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    warmup_epochs: int = 5


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def wrap_ddp(model: CF_HPINO, device: torch.device, local_rank: int) -> torch.nn.Module:
    return DDP(model.to(device), device_ids=[local_rank], output_device=local_rank)


def _pde_type_from_name(name: str) -> LossPDEType:
    return LossPDEType(name)


def _pricing_model_from_name(name: str) -> PricingModel:
    return {
        "black_scholes": PricingModel.BLACK_SCHOLES,
        "fractional_bs": PricingModel.FRACTIONAL_BS,
        "merton": PricingModel.MERTON,
    }[name]


def _split_indices(
    n: int, val_fraction: float, test_fraction: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - n_test - 1)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return train_idx, val_idx, test_idx


class CFHPINOTrainer:
    def __init__(
        self,
        model: CF_HPINO,
        loss_fn: CFHPINOLoss,
        cfg: TrainConfig,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.best_metric = float("inf")
        self.wait = 0
        self.history: List[Dict[str, float]] = []
        self.test_indices: Dict[str, List[int]] = {}
        self.ema: Optional[ModelEMA] = (
            ModelEMA(self._unwrap(), decay=cfg.ema_decay) if cfg.use_ema else None
        )

        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.log_dir = self.ckpt_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / "metrics.csv"
        self._init_csv()

        if cfg.resume:
            self._load_resume(Path(cfg.resume))

    def _init_csv(self) -> None:
        if not self._csv_path.exists():
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "stage",
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_rel_l2",
                        "lr",
                        "time_s",
                    ]
                )

    def _append_csv(self, row: Dict) -> None:
        with self._csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    row["stage"],
                    row["epoch"],
                    f"{row['train_loss']:.6e}",
                    f"{row.get('val_loss', 0):.6e}",
                    f"{row.get('val_rel_l2', 0):.6e}",
                    f"{row.get('lr', 0):.6e}",
                    f"{row.get('time_s', 0):.2f}",
                ]
            )

    def _build_scheduler(self):
        if self.cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=8
            )
        total = self.cfg.epochs_per_stage * max(len(self.cfg.curriculum), 1)

        def lr_lambda(epoch: int) -> float:
            if epoch < self.cfg.warmup_epochs:
                return max((epoch + 1) / max(self.cfg.warmup_epochs, 1), 0.1)
            progress = (epoch - self.cfg.warmup_epochs) / max(
                total - self.cfg.warmup_epochs, 1
            )
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _dataset_cfg(self, stage: str) -> DatasetConfig:
        base = self.cfg.dataset
        stage_seed = (base.seed if base else self.cfg.seed) + hash(stage) % 100_000
        return DatasetConfig(
            model=_pricing_model_from_name(stage),
            n_param_samples=base.n_param_samples if base else 256,
            n_spatial=base.n_spatial if base else 64,
            n_temporal=base.n_temporal if base else 32,
            normalize_coords=base.normalize_coords if base else True,
            option_style=base.option_style if base else "european",
            merton_paths=base.merton_paths if base else 20_000,
            seed=stage_seed,
        )

    def _make_loaders(
        self,
        stage: str,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        print(f"Building dataset for stage '{stage}' (this may take a few minutes)...")
        t0 = time.time()
        full_ds = OptionPricingDataset(self._dataset_cfg(stage))
        print(f"  Dataset ready: {len(full_ds)} surfaces in {time.time() - t0:.1f}s")
        self.loss_fn.set_geometry_meta(full_ds.meta)

        train_idx, val_idx, test_idx = _split_indices(
            len(full_ds),
            self.cfg.val_fraction,
            self.cfg.test_fraction,
            self.cfg.seed + hash(stage) % 10_000,
        )
        self.test_indices[stage] = test_idx

        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
        test_ds = Subset(full_ds, test_idx)

        kwargs = dict(
            batch_size=self.cfg.batch_size,
            collate_fn=collate_option_batch,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        if distributed:
            train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world_size)
            train_loader = DataLoader(train_ds, sampler=train_sampler, **kwargs)
            val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
            test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
        else:
            train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
            val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
            test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

    def train_epoch(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total = 0.0
        n = 0
        breakdown_acc: Dict[str, float] = {}

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                loss, breakdown = self.loss_fn(
                    self._unwrap(), batch, return_breakdown=True
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update(self._unwrap())

            total += breakdown["total"]
            for k, v in breakdown.items():
                breakdown_acc[k] = breakdown_acc.get(k, 0.0) + v
            n += 1

        avg_bd = {k: v / max(n, 1) for k, v in breakdown_acc.items()}
        return total / max(n, 1), avg_bd

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Validation on held-out parameters: data MSE + relative L2 (no physics grad)."""
        self.model.eval()
        total_mse = 0.0
        total_rel = 0.0
        n = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast(enabled=self.use_amp):
                pred = self._eval_model()(batch["params"], batch["coords"])
            total_mse += torch.mean((pred - batch["prices"]) ** 2).item()
            total_rel += relative_l2(pred, batch["prices"])
            n += 1

        avg_mse = total_mse / max(n, 1)
        avg_rel = total_rel / max(n, 1)
        return avg_mse, avg_rel, {"data": avg_mse, "val_rel_l2": avg_rel}

    def _unwrap(self) -> CF_HPINO:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _eval_model(self) -> CF_HPINO:
        if self.ema is not None:
            return self.ema.shadow
        return self._unwrap()

    def _monitor_value(self, val_loss: float, val_rel: float) -> float:
        return val_rel if self.cfg.monitor == "val_rel_l2" else val_loss

    def lbfgs_polish(self, loader: DataLoader) -> None:
        batches = list(loader)
        if not batches:
            return

        def closure():
            self.optimizer.zero_grad()
            loss_sum = 0.0
            for batch in batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss_sum = loss_sum + self.loss_fn(self._unwrap(), batch)
            loss_sum.backward()
            return loss_sum

        lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.cfg.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )
        lbfgs.step(closure)

    def save_checkpoint(
        self,
        path: Path,
        stage: str,
        epoch: int,
        extra: Optional[Dict] = None,
    ) -> None:
        from ..utils.config_loader import model_config_to_dict

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._unwrap().state_dict(),
            "model_config": model_config_to_dict(self._unwrap().cfg),
            "loss_fn": self.loss_fn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "ema": self.ema.state_dict() if self.ema else None,
            "stage": stage,
            "epoch": epoch,
            "history": self.history,
            "best_metric": self.best_metric,
            "test_indices": self.test_indices,
            "train_config": {k: v for k, v in asdict(self.cfg).items() if k != "dataset"},
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def _load_resume(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._unwrap().load_state_dict(ckpt["model"])
        if "loss_fn" in ckpt:
            self.loss_fn.load_state_dict(ckpt["loss_fn"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and self.use_amp:
            self.scaler.load_state_dict(ckpt["scaler"])
        if "ema" in ckpt and ckpt["ema"] is not None and self.ema is not None:
            self.ema.load_state_dict(ckpt["ema"])
        self.history = ckpt.get("history", [])
        self.best_metric = ckpt.get("best_metric", float("inf"))
        self.test_indices = ckpt.get("test_indices", {})
        print(f"Resumed from {path} (best {self.cfg.monitor}={self.best_metric:.6e})")

    def save_manifest(self, config_path: Optional[str] = None) -> None:
        manifest = {
            "experiment_name": self.cfg.experiment_name,
            "config_path": config_path,
            "monitor": self.cfg.monitor,
            "best_metric": self.best_metric,
            "curriculum": self.cfg.curriculum,
            "seed": self.cfg.seed,
            "splits": {
                "val_fraction": self.cfg.val_fraction,
                "test_fraction": self.cfg.test_fraction,
            },
        }
        with (self.log_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @torch.no_grad()
    def evaluate_test(self, test_loader: DataLoader, stage: str) -> Dict[str, float]:
        self._eval_model().eval()
        mse_acc, rel_acc, n = 0.0, 0.0, 0
        for batch in test_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            pred = self._eval_model()(batch["params"], batch["coords"])
            mse_acc += torch.mean((pred - batch["prices"]) ** 2).item()
            rel_acc += relative_l2(pred, batch["prices"])
            n += 1
        results = {
            "test_mse": mse_acc / max(n, 1),
            "test_rel_l2": rel_acc / max(n, 1),
            "stage": stage,
        }
        out_path = self.log_dir / f"test_{stage}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Test [{stage}]: rel-L2={results['test_rel_l2']:.6e} MSE={results['test_mse']:.6e}")
        return results

    def train(
        self,
        distributed: bool = False,
        rank: int = 0,
        config_path: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        torch.manual_seed(self.cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.cfg.seed)

        if rank == 0:
            self.save_manifest(config_path)

        global_epoch = 0
        all_test_results: Dict[str, Dict] = {}

        for stage in self.cfg.curriculum:
            self.loss_fn.cfg.pde_type = _pde_type_from_name(stage)
            train_loader, val_loader, test_loader = self._make_loaders(
                stage, distributed=distributed, rank=rank
            )

            if rank == 0:
                print(f"\n=== Stage: {stage} | train={len(train_loader.dataset)} "
                      f"val={len(val_loader.dataset)} test={len(test_loader.dataset)} ===")

            stage_wait = 0
            for ep in range(self.cfg.epochs_per_stage):
                t0 = time.time()
                if distributed and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(global_epoch)

                train_loss, train_bd = self.train_epoch(train_loader)
                val_loss, val_rel, val_bd = self.validate(val_loader)
                metric = self._monitor_value(val_loss, val_rel)

                lr = self.optimizer.param_groups[0]["lr"]
                if self.cfg.scheduler == "plateau":
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

                record = {
                    "stage": stage,
                    "epoch": global_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_rel_l2": val_rel,
                    "lr": lr,
                    "time_s": time.time() - t0,
                    **{f"train_{k}": v for k, v in train_bd.items()},
                    **{f"val_{k}": v for k, v in val_bd.items()},
                }
                self.history.append(record)

                if rank == 0:
                    self._append_csv(record)
                    print(
                        f"[{stage}] ep {ep+1}/{self.cfg.epochs_per_stage} "
                        f"train={train_loss:.4e} val={val_loss:.4e} "
                        f"val_relL2={val_rel:.4e} lr={lr:.2e} ({record['time_s']:.1f}s)"
                    )

                if metric < self.best_metric:
                    self.best_metric = metric
                    stage_wait = 0
                    if rank == 0:
                        self.save_checkpoint(
                            self.ckpt_dir / "best.pt",
                            stage,
                            global_epoch,
                            extra={"val_rel_l2": val_rel, "val_loss": val_loss},
                        )
                else:
                    stage_wait += 1

                if rank == 0 and (ep + 1) % self.cfg.save_every == 0:
                    self.save_checkpoint(
                        self.ckpt_dir / f"epoch_{global_epoch}.pt", stage, global_epoch
                    )

                if (
                    ep + 1 >= self.cfg.min_epochs
                    and stage_wait >= self.cfg.patience
                ):
                    if rank == 0:
                        print(f"Early stop stage={stage} (no val improvement for {self.cfg.patience} epochs)")
                    break

                if (
                    self.cfg.use_lbfgs
                    and global_epoch >= self.cfg.lbfgs_start_epoch
                    and ep == self.cfg.epochs_per_stage - 1
                ):
                    if rank == 0:
                        print("L-BFGS polish...")
                    self.lbfgs_polish(train_loader)

                global_epoch += 1

            if rank == 0 and self.cfg.run_test_after_train:
                ckpt = torch.load(
                    self.ckpt_dir / "best.pt",
                    map_location=self.device,
                    weights_only=False,
                )
                self._unwrap().load_state_dict(ckpt["model"])
                if ckpt.get("ema") and self.ema is not None:
                    self.ema.load_state_dict(ckpt["ema"])
                all_test_results[stage] = self.evaluate_test(test_loader, stage)

        if rank == 0:
            with (self.log_dir / "all_test_results.json").open("w", encoding="utf-8") as f:
                json.dump(all_test_results, f, indent=2)
            self.save_checkpoint(self.ckpt_dir / "last.pt", stage, global_epoch - 1)

        return self.history


def run_training(
    backbone: str = "fno",
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
) -> CF_HPINO:
    cfg = TrainConfig(device=device, checkpoint_dir=checkpoint_dir)
    model = build_cf_hpino(backbone=backbone)
    loss_fn = CFHPINOLoss(LossConfig(adaptive=True))
    trainer = CFHPINOTrainer(model, loss_fn, cfg)
    trainer.train()
    return trainer._unwrap()
