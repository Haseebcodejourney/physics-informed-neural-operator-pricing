"""
YAML experiment configuration loader.

Example:
    exp = load_experiment("configs/black_scholes.yaml")
    model, loss_fn, trainer = build_trainer_from_experiment(exp)
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from ..cf_hpino_loss import CFHPINOLoss, LossConfig, LossPDEType
from ..cf_hpino_model import CF_HPINO, CFHPINOConfig, OptionStyle, build_cf_hpino
from ..data.synthetic_pde import DatasetConfig, PricingModel
from ..train.trainer import CFHPINOTrainer, TrainConfig


def _filter_dataclass(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def _parse_pricing_model(name: str) -> PricingModel:
    return PricingModel(name)


def _parse_pde_type(name: str) -> LossPDEType:
    return LossPDEType(name)


def _parse_option_style(name: str) -> OptionStyle:
    return OptionStyle(name.lower())


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def experiment_from_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse YAML dict into component configs."""
    model_raw = dict(raw.get("model", {}))
    data_raw = dict(raw.get("data", {}))
    loss_raw = dict(raw.get("loss", {}))
    train_raw = dict(raw.get("train", {}))

    if "model" in data_raw and isinstance(data_raw["model"], str):
        data_raw["model"] = _parse_pricing_model(data_raw["model"])

    if "pde_type" in loss_raw and isinstance(loss_raw["pde_type"], str):
        loss_raw["pde_type"] = _parse_pde_type(loss_raw["pde_type"])

    if "option_style" in model_raw and isinstance(model_raw["option_style"], str):
        model_raw["option_style"] = _parse_option_style(model_raw["option_style"])
    if "backbone" in model_raw and isinstance(model_raw["backbone"], str):
        from ..cf_hpino_model import OperatorBackbone

        model_raw["backbone"] = OperatorBackbone(model_raw["backbone"].lower())

    model_cfg = CFHPINOConfig(**_filter_dataclass(CFHPINOConfig, model_raw))
    if "option_style" in data_raw and isinstance(data_raw["option_style"], str):
        data_raw["option_style"] = data_raw["option_style"].lower()
    dataset_cfg = DatasetConfig(**_filter_dataclass(DatasetConfig, data_raw))
    loss_cfg = LossConfig(**_filter_dataclass(LossConfig, loss_raw))
    train_cfg = TrainConfig(**_filter_dataclass(TrainConfig, train_raw))

    return {
        "model": model_cfg,
        "data": dataset_cfg,
        "loss": loss_cfg,
        "train": train_cfg,
        "raw": raw,
    }


def load_experiment(path: str | Path) -> Dict[str, Any]:
    return experiment_from_dict(load_yaml(path))


def model_config_to_dict(cfg: CFHPINOConfig) -> Dict[str, Any]:
    from dataclasses import asdict

    d = asdict(cfg)
    d["backbone"] = cfg.backbone.value
    d["option_style"] = cfg.option_style.value
    return d


def model_config_from_dict(data: Dict[str, Any]) -> CFHPINOConfig:
    return experiment_from_dict({"model": data})["model"]


def build_model_from_experiment(exp: Dict[str, Any]) -> CF_HPINO:
    cfg: CFHPINOConfig = exp["model"]
    model = CF_HPINO(cfg)
    model.apply_drdd_init(seed=exp["train"].seed)
    return model


def build_trainer_from_experiment(
    exp: Dict[str, Any],
    device: Optional[str] = None,
) -> Tuple[CF_HPINO, CFHPINOLoss, CFHPINOTrainer]:
    train_cfg: TrainConfig = exp["train"]
    if device is not None:
        train_cfg.device = device
    train_cfg.dataset = exp["data"]

    model = build_model_from_experiment(exp)
    loss_fn = CFHPINOLoss(exp["loss"])
    trainer = CFHPINOTrainer(model, loss_fn, train_cfg)
    return model, loss_fn, trainer


def merge_cli_overrides(
    exp: Dict[str, Any],
    *,
    backbone: Optional[str] = None,
    device: Optional[str] = None,
    epochs_per_stage: Optional[int] = None,
    batch_size: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    lr: Optional[float] = None,
) -> Dict[str, Any]:
    """Apply command-line overrides without mutating the original YAML file."""
    if backbone:
        from ..cf_hpino_model import OperatorBackbone

        exp["model"].backbone = OperatorBackbone(backbone.lower())
    train: TrainConfig = exp["train"]
    if device is not None:
        train.device = device
    if epochs_per_stage is not None:
        train.epochs_per_stage = epochs_per_stage
    if batch_size is not None:
        train.batch_size = batch_size
    if checkpoint_dir is not None:
        train.checkpoint_dir = checkpoint_dir
    if lr is not None:
        train.lr = lr
    return exp
