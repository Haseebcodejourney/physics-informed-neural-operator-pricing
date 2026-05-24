"""
Publication-style plots for CF-HPINO experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_price_surface(
    S: np.ndarray,
    t: np.ndarray,
    V: np.ndarray,
    title: str = "Option price surface",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": "3d"})
    Sg, Tg = np.meshgrid(S, t, indexing="ij")
    ax.plot_surface(Sg, Tg, V, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("S")
    ax.set_ylabel("t")
    ax.set_zlabel("V")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_surface_from_grid(
    grid: torch.Tensor,
    prices: torch.Tensor,
    title: str = "Predicted surface",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """grid (H,W,2), prices (H,W) or flattened."""
    g = grid.detach().cpu().numpy()
    if prices.dim() == 1:
        h = int(np.sqrt(prices.numel()))
        w = prices.numel() // h
        V = prices.view(h, w).detach().cpu().numpy()
        S = g[:, 0, 0]
        t = g[0, :, 1]
    else:
        V = prices.detach().cpu().numpy()
        S = g[:, 0, 0]
        t = g[0, :, 1]
    return plot_price_surface(S, t, V, title=title, save_path=save_path)


def plot_convergence(
    history: List[Dict],
    key: str = "loss",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    stages = [h.get("stage", "") for h in history]
    losses = [h[key] for h in history]
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(stages))))
    stage_color = {s: colors[i] for i, s in enumerate(sorted(set(stages)))}
    for i, (s, l) in enumerate(zip(stages, losses)):
        ax.scatter(i, l, c=[stage_color[s]], s=20)
    ax.plot(losses, "k-", alpha=0.3, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Global epoch")
    ax.set_ylabel(key)
    ax.set_title("Training convergence")
    for s in set(stages):
        ax.scatter([], [], c=[stage_color[s]], label=s)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_error_heatmap(
    grid: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    err = (pred - target).detach().cpu().numpy()
    if err.ndim == 1:
        g = grid.detach().cpu().numpy()
        if g.ndim == 4:
            h, w = g.shape[1], g.shape[2]
        elif g.ndim == 3:
            h, w = g.shape[0], g.shape[1]
        else:
            h = int(np.sqrt(err.size))
            w = err.size // h
        err = err.reshape(h, w)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.abs(err), aspect="auto", cmap="hot")
    ax.set_title("|Prediction − Target|")
    ax.set_xlabel("time index")
    ax.set_ylabel("spot index")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_comparison_bar(
    results: Dict[str, float],
    metric: str = "relative_l2",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    vals = [results[n] for n in names]
    ax.bar(names, vals, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison ({metric})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
