#!/usr/bin/env python3
"""
Cloud deployment helpers (AWS SageMaker / GCP Vertex).

SageMaker training job entry:
    python scripts/export_cloud.py --mode train

Environment variables (SageMaker):
    SM_MODEL_DIR, SM_CHANNEL_TRAINING, SM_NUM_GPUS
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package_source(dest: Path) -> None:
    """Copy src + scripts + requirements for container upload."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("src", "scripts", "requirements.txt"):
        src = ROOT / name
        tgt = dest / name
        if src.is_dir():
            shutil.copytree(src, tgt, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, tgt)


def sagemaker_train():
    model_dir = Path(os.environ.get("SM_MODEL_DIR", ROOT / "model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    from src.train.trainer import CFHPINOTrainer, TrainConfig
    from src.cf_hpino_model import build_cf_hpino
    from src.cf_hpino_loss import CFHPINOLoss, LossConfig
    import torch

    n_gpus = int(os.environ.get("SM_NUM_GPUS", "1"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig(device=device, checkpoint_dir=str(model_dir / "checkpoints"))
    model = build_cf_hpino()
    loss_fn = CFHPINOLoss(LossConfig(adaptive=True))
    trainer = CFHPINOTrainer(model, loss_fn, cfg)
    trainer.train()
    shutil.copy(model_dir / "checkpoints" / "best.pt", model_dir / "model.pt")
    with (model_dir / "manifest.json").open("w") as f:
        json.dump({"framework": "pytorch", "n_gpus": n_gpus}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["package", "train"], default="package")
    parser.add_argument("--out", default="cloud_bundle")
    args = parser.parse_args()

    if args.mode == "package":
        package_source(Path(args.out))
        print(f"Bundle written to {args.out}/")
        print("Upload to S3 and point SageMaker TrainingJob at scripts/export_cloud.py --mode train")
    else:
        if os.environ.get("SM_MODEL_DIR"):
            sagemaker_train()
        else:
            print("Set SM_MODEL_DIR for SageMaker, or use scripts/train.py locally.")


if __name__ == "__main__":
    main()
