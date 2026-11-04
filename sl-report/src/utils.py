"""Shared utilities: constants, figure helpers, metric serialization."""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR    = os.path.join(ROOT, "outputs", "figures")
RES_DIR    = os.path.join(ROOT, "outputs", "results")
SAMPLE_CSV = os.path.join(ROOT, "outputs", "sample.csv")

RANDOM_SEED = 42
N_CV        = 5          # stratified CV folds
SCORING     = "f1_macro" # primary CV criterion

CLASS_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}
CLASS_LABELS = [CLASS_NAMES[i] for i in range(1, 8)]


def ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_json(data: dict, name: str) -> str:
    path = os.path.join(RES_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


# ── Metric helpers ────────────────────────────────────────────────────────────
def eval_metrics(y_true, y_pred, fit_time: float = 0.0, pred_time: float = 0.0) -> dict:
    return {
        "macro_f1":         round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy":round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "fit_s":            round(fit_time,  3),
        "predict_s":        round(pred_time, 3),
    }


def print_per_class_metrics(y_true, y_pred, title: str = "") -> None:
    """Print and return per-class precision, recall, F1 table."""
    if title:
        print(f"\n  Per-class metrics — {title}")
    report = classification_report(
        y_true, y_pred,
        labels=list(range(1, 8)),
        target_names=[f"C{i} {CLASS_NAMES[i][:12]}" for i in range(1, 8)],
        zero_division=0,
    )
    for line in report.split("\n"):
        print(f"    {line}")


def plot_per_class_bar(y_true, y_pred, title: str, fname: str) -> None:
    """Save a per-class F1 bar chart."""
    from sklearn.metrics import f1_score as f1
    per_class_f1 = f1(y_true, y_pred, average=None,
                      labels=list(range(1, 8)), zero_division=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3"]
    bars = ax.bar(range(7), per_class_f1, color=colors, edgecolor="white")
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"C{i}\n{CLASS_NAMES[i][:8]}" for i in range(1, 8)], fontsize=7)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.axhline(float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
               color="red", ls="--", lw=1.2, label="Macro F1")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    save_fig(fig, fname)


# ── Learning curve ────────────────────────────────────────────────────────────
def compute_learning_curve(estimator, X, y, n_points: int = 8):
    n = len(y)
    # learning_curve with N_CV folds uses at most n*(N_CV-1)/N_CV samples per fold
    max_train = int(n * (N_CV - 1) / N_CV)
    sizes = np.unique(np.linspace(int(0.1 * n), max_train, n_points, dtype=int))
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_SEED)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=sizes,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
    )
    return train_sizes, train_scores.mean(1), val_scores.mean(1)


def plot_learning_curve(
    train_sizes, train_scores, val_scores,
    title: str, fname: str,
    ylabel: str = "Macro F1",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_scores, "o-", label="Train",      color="#4C72B0")
    ax.plot(train_sizes, val_scores,   "s--", label="Validation", color="#DD8452")
    ax.fill_between(train_sizes, train_scores - 0.02, train_scores + 0.02, alpha=0.15, color="#4C72B0")
    ax.fill_between(train_sizes, val_scores   - 0.02, val_scores   + 0.02, alpha=0.15, color="#DD8452")
    ax.set_xlabel("Training samples")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, fname)


# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title: str, fname: str) -> None:
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(1, 8)))
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"C{i}" for i in range(1, 8)], fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels([f"C{i}\n{CLASS_NAMES[i][:10]}" for i in range(1, 8)], fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if norm[i,j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Recall (row-normalised)")
    save_fig(fig, fname)


# ── CV complexity curve ───────────────────────────────────────────────────────
def cv_scores(estimator, X, y):
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_SEED)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return scores.mean(), scores.std()


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
