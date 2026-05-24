#!/usr/bin/env python3
"""Multi-GPU training entry (torchrun / SageMaker)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.cf_hpino_loss import CFHPINOLoss, LossConfig
from src.cf_hpino_model import build_cf_hpino
from src.data import DatasetConfig, OptionPricingDataset, PricingModel
from src.data.sampling import collate_option_batch
from src.train.ddp_utils import cleanup_distributed, init_distributed, wrap_model
from src.train.trainer import CFHPINOTrainer, TrainConfig


def main():
    rank, world_size, device = init_distributed()
    cfg = TrainConfig(
        device=str(device),
        epochs_per_stage=20,
        batch_size=4,
        curriculum=["black_scholes"],
    )
    model = build_cf_hpino()
    model = wrap_model(model, device, rank)
    loss_fn = CFHPINOLoss(LossConfig(adaptive=True)).to(device)

    ds = OptionPricingDataset(
        DatasetConfig(model=PricingModel.BLACK_SCHOLES, n_param_samples=128)
    )
    sampler = DistributedSampler(ds, rank=rank, num_replicas=world_size) if world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_option_batch,
    )

    trainer = CFHPINOTrainer(
        model.module if hasattr(model, "module") else model,
        loss_fn,
        cfg,
        device=device,
    )
    trainer.model = model
    trainer.train(distributed=world_size > 1, rank=rank)
    cleanup_distributed()
    if rank == 0:
        print("DDP training finished.")


if __name__ == "__main__":
    main()
