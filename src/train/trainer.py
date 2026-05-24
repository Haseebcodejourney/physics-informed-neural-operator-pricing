"""
CF-HPINO training loop: Adam warm-up + optional L-BFGS polish, schedulers,
early stopping, and DDP helpers for cloud multi-GPU jobs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..data.synthetic_pde import DatasetConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..cf_hpino_loss import CFHPINOLoss, LossConfig, LossPDEType
from ..cf_hpino_model import CF_HPINO, build_cf_hpino
from ..data import DatasetConfig, OptionPricingDataset, PricingModel
from ..data.sampling import collate_option_batch


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
    scheduler: str = "cosine"  # cosine | plateau
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    num_workers: int = 0
    # Curriculum
    curriculum: List[str] = field(
        default_factory=lambda: ["black_scholes", "fractional_bs", "merton"]
    )
    epochs_per_stage: int = 30
    # Optional base dataset template (from YAML); stage overrides `model` field
    dataset: Optional["DatasetConfig"] = None


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def wrap_ddp(model: CF_HPINO, device: torch.device, local_rank: int) -> torch.nn.Module:
    return DDP(model.to(device), device_ids=[local_rank], output_device=local_rank)


def _pde_type_from_name(name: str) -> LossPDEType:
    return LossPDEType(name if name != "fractional_bs" else "fractional_bs")


def _pricing_model_from_name(name: str) -> PricingModel:
    mapping = {
        "black_scholes": PricingModel.BLACK_SCHOLES,
        "fractional_bs": PricingModel.FRACTIONAL_BS,
        "merton": PricingModel.MERTON,
    }
    return mapping[name]


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
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.best_loss = float("inf")
        self.wait = 0
        self.history: List[Dict[str, float]] = []

    def _build_scheduler(self):
        if self.cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        total_epochs = self.cfg.epochs_per_stage * max(len(self.cfg.curriculum), 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_epochs, 1)
        )

    def _make_loader(
        self,
        stage: str,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> DataLoader:
        base = self.cfg.dataset
        ds_cfg = DatasetConfig(
            model=_pricing_model_from_name(stage),
            n_param_samples=base.n_param_samples if base else 256,
            n_spatial=base.n_spatial if base else 64,
            n_temporal=base.n_temporal if base else 32,
            normalize_coords=base.normalize_coords if base else True,
            option_style=base.option_style if base else "european",
            merton_paths=base.merton_paths if base else 20_000,
            seed=(self.cfg.seed + hash(stage) % 10000) if base is None else base.seed,
        )
        dataset = OptionPricingDataset(ds_cfg)
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_option_batch,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)
            loss, breakdown = self.loss_fn(
                self._unwrap(), batch, return_breakdown=True
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += breakdown["total"]
            n += 1
        return total / max(n, 1)

    def _unwrap(self) -> CF_HPINO:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def lbfgs_polish(self, loader: DataLoader) -> None:
        """Full-batch L-BFGS on last epoch data snapshot (small synthetic sets)."""
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

    def save_checkpoint(self, path: Path, stage: str, epoch: int) -> None:
        from ..utils.config_loader import model_config_to_dict

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self._unwrap().state_dict(),
                "model_config": model_config_to_dict(self._unwrap().cfg),
                "loss_fn": self.loss_fn.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "stage": stage,
                "epoch": epoch,
                "history": self.history,
            },
            path,
        )

    def train(self, distributed: bool = False, rank: int = 0) -> List[Dict[str, float]]:
        torch.manual_seed(self.cfg.seed)
        global_epoch = 0
        ckpt_dir = Path(self.cfg.checkpoint_dir)

        for stage in self.cfg.curriculum:
            self.loss_fn.cfg.pde_type = _pde_type_from_name(stage)
            loader = self._make_loader(stage, distributed=distributed, rank=rank)
            print(f"=== Curriculum stage: {stage} ===")

            for ep in range(self.cfg.epochs_per_stage):
                t0 = time.time()
                if distributed and loader.sampler is not None:
                    loader.sampler.set_epoch(global_epoch)
                avg_loss = self.train_epoch(loader)

                if self.cfg.scheduler == "plateau":
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()

                if (
                    self.cfg.use_lbfgs
                    and global_epoch >= self.cfg.lbfgs_start_epoch
                    and ep == self.cfg.epochs_per_stage - 1
                ):
                    self.lbfgs_polish(loader)

                record = {
                    "stage": stage,
                    "epoch": global_epoch,
                    "loss": avg_loss,
                    "time_s": time.time() - t0,
                }
                self.history.append(record)

                if rank == 0:
                    print(
                        f"[{stage}] epoch {ep+1}/{self.cfg.epochs_per_stage} "
                        f"loss={avg_loss:.6e} ({record['time_s']:.1f}s)"
                    )

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.wait = 0
                    if rank == 0:
                        self.save_checkpoint(
                            ckpt_dir / "best.pt", stage, global_epoch
                        )
                else:
                    self.wait += 1
                    if self.wait >= self.cfg.patience:
                        print(f"Early stop at stage {stage}, epoch {ep}")
                        break

                global_epoch += 1

        return self.history


def run_training(
    backbone: str = "fno",
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
) -> CF_HPINO:
    """Single-process entry point for scripts/train.py."""
    cfg = TrainConfig(device=device, checkpoint_dir=checkpoint_dir)
    model = build_cf_hpino(backbone=backbone)
    loss_fn = CFHPINOLoss(LossConfig(adaptive=True))
    trainer = CFHPINOTrainer(model, loss_fn, cfg)
    trainer.train()
    return trainer._unwrap()
