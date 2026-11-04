"""
Shared plotting + results-IO helpers for the UL Report.

Figures are compact and multi-panel where possible (8-page limit). Every figure
in the report needs a takeaway tied to clustering behaviour, DR diagnostics,
label alignment, or downstream NN behaviour -- these helpers only standardize
style and saving. Legible at normal grading zoom is a scored requirement.
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
    "axes.titlesize": 11, "legend.fontsize": 8, "axes.labelsize": 10,
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
        json.dump(obj, f, indent=2,
                  default=lambda o: getattr(o, "tolist", lambda: str(o))())
    return path


# ── Reusable panels ───────────────────────────────────────────────────────────
def plot_metric_vs_k(df: pd.DataFrame, k_col: str, metric_cols: list[str],
                     name: str, title: str) -> str:
    """One small-multiple axis per internal/external metric vs k / n_components."""
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), squeeze=False)
    for ax, m in zip(axes[0], metric_cols):
        ax.plot(df[k_col], df[m], marker="o", lw=1.4)
        ax.set(xlabel=k_col, ylabel=m, title=m)
    fig.suptitle(title, y=1.03)
    return save_fig(fig, name)


def plot_scree(evr: np.ndarray, name: str, target: float | None = None) -> str:
    """PCA explained-variance-ratio bars + cumulative curve."""
    cum = np.cumsum(evr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(1, len(evr) + 1), evr, alpha=0.5, label="per-component")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(evr) + 1), cum, color="crimson", marker=".",
             label="cumulative")
    ax2.set_ylabel("cumulative explained variance")
    ax2.grid(False)
    if target is not None:
        ax2.axhline(target, color="gray", ls="--", lw=1)
        k = int(np.searchsorted(cum, target) + 1)
        ax2.annotate(f"k={k} @ {target:.0%}", (k, target), fontsize=8,
                     xytext=(k + 1, target - 0.08),
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set(xlabel="principal component", ylabel="explained variance ratio",
           title="PCA scree / cumulative variance")
    return save_fig(fig, name)


def plot_kurtosis(kurt: np.ndarray, name: str) -> str:
    """ICA per-component absolute excess kurtosis (non-Gaussianity)."""
    order = np.argsort(np.abs(kurt))[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(kurt)), np.abs(kurt)[order])
    ax.set(xlabel="ICA component (sorted)", ylabel="|excess kurtosis|",
           title="ICA component non-Gaussianity")
    return save_fig(fig, name)


def plot_embedding(emb: np.ndarray, color, name: str, title: str,
                   cbar_label: str = "") -> str:
    """2D scatter of an embedding, colored by cluster id or Cover_Type."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=color, s=4, cmap="tab10", alpha=0.6)
    ax.set(xlabel="dim 1", ylabel="dim 2", title=title)
    if cbar_label:
        fig.colorbar(sc, ax=ax, label=cbar_label)
    return save_fig(fig, name)


def plot_contingency(ct: np.ndarray, name: str,
                     title: str = "Cluster vs Cover_Type") -> str:
    """Row-normalized cluster-label contingency heatmap."""
    row_sum = ct.sum(1, keepdims=True)
    frac = ct / np.maximum(row_sum, 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(frac, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(ct.shape[1]), [f"C{j+1}" for j in range(ct.shape[1])])
    ax.set_yticks(range(ct.shape[0]), [f"k{i}" for i in range(ct.shape[0])])
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, ct[i, j], ha="center", va="center", color="w", fontsize=6)
    ax.set(xlabel="Cover_Type", ylabel="cluster", title=title)
    fig.colorbar(im, ax=ax, label="row fraction")
    return save_fig(fig, name)


def stability_band(per_seed: list[list[float]]):
    """Median + IQR band across seeds; returns (median, q25, q75) aligned by index."""
    arr = np.array(per_seed, dtype=float)
    return np.median(arr, 0), np.percentile(arr, 25, 0), np.percentile(arr, 75, 0)
