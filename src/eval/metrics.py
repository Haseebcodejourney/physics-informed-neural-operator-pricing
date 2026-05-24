"""
Evaluation metrics for option pricing models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class MetricResult:
    mse: float
    rmse: float
    relative_l2: float
    max_abs_error: float
    inference_ms: float
    greeks: Dict[str, float] = field(default_factory=dict)


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    num = torch.norm(pred - target)
    den = torch.norm(target) + eps
    return (num / den).item()


def max_abs_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.max(torch.abs(pred - target)).item()


@torch.no_grad()
def measure_inference_ms(
    model: nn.Module,
    params: torch.Tensor,
    coords: torch.Tensor,
    n_warmup: int = 3,
    n_runs: int = 20,
    device: Optional[torch.device] = None,
) -> float:
    device = device or params.device
    model = model.to(device).eval()
    params = params.to(device)
    coords = coords.to(device)
    for _ in range(n_warmup):
        _ = model(params, coords)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(params, coords)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0


def greek_errors(
    model: nn.Module,
    params: torch.Tensor,
    coords: torch.Tensor,
    reference: Dict[str, torch.Tensor],
    keys: List[str] = ("delta", "gamma", "vega"),
) -> Dict[str, float]:
    if not hasattr(model, "greeks"):
        return {}
    pred_g = model.greeks(params, coords, compute=tuple(keys))
    errs = {}
    for k in keys:
        if k in reference and k in pred_g:
            errs[k] = relative_l2(pred_g[k], reference[k])
    return errs


def evaluate_model(
    model: nn.Module,
    params: torch.Tensor,
    coords: torch.Tensor,
    target: torch.Tensor,
    reference_greeks: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> MetricResult:
    device = device or target.device
    model.eval()
    with torch.no_grad():
        pred = model(params.to(device), coords.to(device))
    inf_ms = measure_inference_ms(model, params[:1], coords[:1], device=device)
    greek_errs = {}
    if reference_greeks is not None and hasattr(model, "greeks"):
        try:
            greek_errs = greek_errors(model, params[:1], coords[:1], reference_greeks)
        except Exception:
            pass
    return MetricResult(
        mse=mse(pred, target.to(device)),
        rmse=float(np.sqrt(mse(pred, target.to(device)))),
        relative_l2=relative_l2(pred, target.to(device)),
        max_abs_error=max_abs_error(pred, target.to(device)),
        inference_ms=inf_ms,
        greeks=greek_errs,
    )


def generalization_gap(
    in_sample: MetricResult,
    out_sample: MetricResult,
) -> Dict[str, float]:
    """Relative degradation on held-out parameter region."""
    return {
        "mse_gap": out_sample.mse - in_sample.mse,
        "rel_l2_ratio": out_sample.relative_l2 / (in_sample.relative_l2 + 1e-8),
    }


def compare_methods(
    models: Dict[str, nn.Module],
    params: torch.Tensor,
    coords: torch.Tensor,
    target: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, MetricResult]:
    dev = torch.device(device)
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, params, coords, target, device=dev)
    return results


def results_to_table(results: Dict[str, MetricResult]) -> str:
    header = f"{'Model':<16} {'MSE':>12} {'Rel-L2':>10} {'MaxErr':>10} {'Inf(ms)':>10}"
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        lines.append(
            f"{name:<16} {r.mse:12.4e} {r.relative_l2:10.4e} "
            f"{r.max_abs_error:10.4e} {r.inference_ms:10.2f}"
        )
    return "\n".join(lines)
