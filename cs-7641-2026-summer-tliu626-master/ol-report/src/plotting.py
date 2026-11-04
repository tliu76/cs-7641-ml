"""
Shared plotting + results-IO helpers.

Figures are compact and multi-panel where possible (8-page limit). Every figure
in the report needs a takeaway tied to optimizer/regularization/RO behavior;
these helpers just standardize style and saving.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")   # switch to "MacOSX" locally for interactive viewing.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

plt.rcParams.update({
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 11, "legend.fontsize": 8,
})


def save_fig(fig, name: str) -> str:
    path = os.path.join(C.FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(C.RESULT_DIR, name)
    df.to_csv(path, index=False)
    return path


def save_json(obj, name: str) -> str:
    path = os.path.join(C.RESULT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: getattr(o, "tolist", lambda: str(o))())
    return path


# ── Reusable panels ───────────────────────────────────────────────────────────
def plot_best_so_far(histories: dict[str, list[dict]], name: str,
                     x_key: str = "func_evals", title: str = "RO best-so-far"):
    """Best-so-far objective vs function evaluations (Part 1 core figure)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, hist in histories.items():
        xs = [h[x_key] for h in hist]
        ys = [h["best_loss"] for h in hist]
        ax.plot(xs, ys, label=label, lw=1.5)
    ax.set(xlabel=x_key.replace("_", " "), ylabel="best validation loss", title=title)
    ax.legend()
    return save_fig(fig, name)


def plot_trajectories(logs: dict[str, dict], name: str, x_key: str = "grad_evals",
                      title: str = "Optimizer trajectories"):
    """Validation loss vs compute for gradient-based optimizers (Part 2)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, log in logs.items():
        ax.plot(log[x_key], log["val_loss"], label=label, lw=1.5)
    ax.set(xlabel=x_key.replace("_", " "), ylabel="validation loss", title=title)
    ax.legend()
    return save_fig(fig, name)


def plot_heatmap(values: np.ndarray, xticks, yticks, name: str,
                 xlabel: str, ylabel: str, title: str, cbar: str = "val loss"):
    """Coarse sensitivity heatmap, e.g. (alpha, beta1) (Part 2)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xticks)), [f"{t:g}" for t in xticks])
    ax.set_yticks(range(len(yticks)), [f"{t:g}" for t in yticks])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center",
                    color="w", fontsize=7)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(im, ax=ax, label=cbar)
    return save_fig(fig, name)


def plot_confusion(cm: np.ndarray, name: str, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ticks = [f"C{i+1}" for i in range(C.N_CLASSES)]
    ax.set_xticks(range(C.N_CLASSES), ticks)
    ax.set_yticks(range(C.N_CLASSES), ticks)
    for i in range(C.N_CLASSES):
        for j in range(C.N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.colorbar(im, ax=ax)
    return save_fig(fig, name)


def stability_band(per_seed: list[list[float]]):
    """Median + IQR band across seeds; returns (median, q25, q75) aligned by index."""
    arr = np.array(per_seed, dtype=float)
    return np.median(arr, 0), np.percentile(arr, 25, 0), np.percentile(arr, 75, 0)
