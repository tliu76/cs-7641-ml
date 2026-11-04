"""
Part 2: Adam-family optimizer ablations on the FULL PyTorch network.

This is an ablation study, not a tour of optimizers. Required variants:
  - SGD (no momentum)          - Adam
  - SGD (momentum)             - Adam without bias correction  (custom)
  - Nesterov momentum          - Adam with beta1 = 0  (RMSProp-like)
                               - AdamW (decoupled weight decay)

Required analyses:
  - validation loss vs wall-clock  and  vs gradient evaluations/updates
  - time & steps to fixed validation-loss threshold ell (train/val chosen only)
  - coarse sensitivity heatmaps over (alpha, beta1) and (alpha, beta2)
  - stability across >= 3 seeds
  - generalization gap (train vs val) and final test metrics + class-level behavior

Fairness: coarse comparable LR searches per family under matched budgets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import config as C
from .adam_no_bias import AdamNoBiasCorrection
from .data import get_split, make_loaders
from .metrics import compute_metrics
from .model import build_model
from .plotting import plot_heatmap, plot_trajectories, save_json, save_table
from .seeds import set_all_seeds
from .train import final_test_metrics, make_criterion_for, train_model


# ── Optimizer factory (one place, so protocols stay comparable) ───────────────
def make_optimizer(name: str, params, lr: float, **kw):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    if name == "adam_no_bias":
        return AdamNoBiasCorrection(params, lr=lr, betas=(0.9, 0.999))
    if name == "adam_beta1_0":                       # RMSProp-like condition
        return torch.optim.Adam(params, lr=lr, betas=(0.0, 0.999))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                 weight_decay=kw.get("weight_decay", 1e-2))
    raise ValueError(f"unknown optimizer {name}")


OPTIMIZERS = ["sgd", "sgd_momentum", "nesterov", "adam",
              "adam_no_bias", "adam_beta1_0", "adamw"]


def run_ablations(seeds=None, lr_by_opt: dict | None = None) -> pd.DataFrame:
    """Train each required optimizer variant across seeds; log trajectories + metrics."""
    seeds = seeds or C.SEEDS
    # Per-family learning rates from the coarse pilot LR search (LR_GRID_SGD /
    # LR_GRID_ADAM, 1 seed, reported in REPRO). SGD-family best at 0.1 (matches
    # the SL baseline); Adam-family best at 1e-3. AdamW uses the same LR as Adam.
    lr_by_opt = lr_by_opt or {
        "sgd": 1e-1, "sgd_momentum": 1e-1, "nesterov": 1e-1, "adam": 1e-3,
        "adam_no_bias": 1e-3, "adam_beta1_0": 1e-3, "adamw": 1e-3,
    }
    split = get_split()
    criterion = make_criterion_for(split)
    rows, traj_for_plot = [], {}

    for name in OPTIMIZERS:
        for seed in seeds:
            set_all_seeds(seed)
            train_loader, val_loader, test_loader = make_loaders(split, seed=seed)
            model = build_model(dropout_p=0.0)
            opt = make_optimizer(name, [p for p in model.parameters()], lr_by_opt[name])
            log = train_model(model, train_loader, val_loader, opt, criterion,
                              seed=seed, ell=C.ELL_VAL_LOSS)

            test_m = final_test_metrics(model, test_loader, criterion)
            gen_gap = (log.val_loss[-1] - log.train_loss[-1]) if log.val_loss else None
            row = {"optimizer": name, "seed": seed, "lr": lr_by_opt[name],
                   "grad_evals": log.total_grad_evals,
                   "wall_clock_s": log.wall_clock[-1] if log.wall_clock else None,
                   "reached_ell_at": log.reached_ell_at, "gen_gap": gen_gap,
                   "status": log.status}
            row.update(test_m.summary_row())
            rows.append(row)
            save_json(log.to_dict(), f"part2_{name}_seed{seed}.json")
            if seed == seeds[0]:
                traj_for_plot[name] = log.to_dict()

    df = pd.DataFrame(rows)
    save_table(df, "part2_ablation_results.csv")
    plot_trajectories(traj_for_plot, "part2_trajectories.png", x_key="grad_evals",
                      title="Adam-family: val loss vs gradient evals")
    plot_trajectories(traj_for_plot, "part2_trajectories_time.png", x_key="wall_clock",
                      title="Adam-family: val loss vs wall-clock")
    return df


def run_sensitivity(seed: int = 0, beta_axis: str = "beta1") -> np.ndarray:
    """Coarse (alpha, beta) heatmap of final validation loss for Adam.

    beta_axis in {"beta1","beta2"}. Keep grids to 3x3/4x4 so seed analysis stays
    feasible. Returns the value matrix and saves the heatmap.
    """
    split = get_split()
    criterion = make_criterion_for(split)
    betas = C.BETA1_GRID if beta_axis == "beta1" else C.BETA2_GRID
    grid = np.full((len(betas), len(C.ALPHA_GRID)), np.nan)

    for i, b in enumerate(betas):
        for j, a in enumerate(C.ALPHA_GRID):
            set_all_seeds(seed)
            train_loader, val_loader, _ = make_loaders(split, seed=seed)
            model = build_model(dropout_p=0.0)
            betas_tuple = (b, 0.999) if beta_axis == "beta1" else (0.9, b)
            opt = torch.optim.Adam([p for p in model.parameters()], lr=a, betas=betas_tuple)
            log = train_model(model, train_loader, val_loader, opt, criterion,
                              epochs=max(10, C.EPOCHS // 3), seed=seed)
            grid[i, j] = log.val_loss[-1] if log.val_loss else np.nan

    plot_heatmap(grid, C.ALPHA_GRID, betas, f"part2_heatmap_{beta_axis}.png",
                 xlabel="alpha (lr)", ylabel=beta_axis,
                 title=f"Adam sensitivity: (alpha, {beta_axis})")
    return grid


if __name__ == "__main__":
    print(run_ablations())
    run_sensitivity(beta_axis="beta1")
    run_sensitivity(beta_axis="beta2")
