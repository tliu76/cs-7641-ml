"""
Part 4 (EXTRA CREDIT): Integrated best combination.

Composition, NOT a new search. Combine:
  - best standard Adam hyperparameters from Part 2 (same batch size)
  - best regularization combination from Part 3 (exact values + placements)
  - best RO fine-tuning method from Part 1 on the same final 1-3 layers

Constraints:
  - no new optimizers / no fresh wide sweeps
  - RO fine-tuning <= RO_FINETUNE_FRAC of the Part-1 function-eval budget
  - <= INTERACTION_TRIALS small interaction trials
  - seeds, hardware, splits, batch size consistent with earlier parts

Report: whether the integrated recipe beats the best of Parts 1-3, the compute
trade-off (test metric vs total evaluations/time), stability across seeds, and a
concise hypothesis-resolution paragraph.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import get_split, make_loaders
from .model import build_model, freeze_to_final_layers, get_flat_params, set_flat_params
from .part1_ro import _make_val_objective
from .part3_reg import ADAM_BETAS, ADAM_LR
from .plotting import save_json, save_table
from .ro_algorithms import RO_ALGORITHMS
from .seeds import set_all_seeds
from .train import evaluate, final_test_metrics, make_criterion_for, train_model

# Fallback winners if Part 1/3 artifacts are missing; normally read from disk.
_FALLBACK_RO_ALGO = "sa"
_FALLBACK_REG = {"weight_decay": 1e-4, "dropout": 0.1, "label_smoothing": 0.05}


def _load_winners() -> tuple[str, dict]:
    """Best RO algo from Part 1 results and best reg combo from Part 3 selection."""
    ro_algo = _FALLBACK_RO_ALGO
    p1 = os.path.join(C.RESULT_DIR, "part1_ro_results.csv")
    if os.path.exists(p1):
        df = pd.read_csv(p1)
        ro_algo = str(df.groupby("algo")["best_val_loss"].mean().idxmin())

    reg = dict(_FALLBACK_REG)
    p3 = os.path.join(C.RESULT_DIR, "part3_selection.json")
    if os.path.exists(p3):
        with open(p3) as f:
            reg = json.load(f)["best_combo"]["reg"]
    return ro_algo, reg


def run_part4(seeds=None) -> pd.DataFrame:
    seeds = seeds or C.SEEDS
    split = get_split()
    ro_budget = max(1, int(C.RO_FUNC_EVAL_BUDGET * C.RO_FINETUNE_FRAC))
    best_ro_algo, best_reg = _load_winners()
    print(f"[part4] integrating best RO={best_ro_algo}, best reg combo={best_reg}")
    rows = []

    for seed in seeds:
        set_all_seeds(seed)
        train_loader, val_loader, test_loader = make_loaders(split, seed=seed)

        # 1) best Adam + best regularization combo
        model = build_model(dropout_p=best_reg.get("dropout", 0.0))
        criterion = make_criterion_for(split, label_smoothing=best_reg.get("label_smoothing", 0.0))
        opt = torch.optim.Adam([p for p in model.parameters()], lr=ADAM_LR,
                               betas=ADAM_BETAS, weight_decay=best_reg.get("weight_decay", 0.0))
        train_model(model, train_loader, val_loader, opt, criterion, seed=seed)
        pre = final_test_metrics(model, test_loader, criterion)

        # 2) best RO fine-tuning on final layers under a small function-eval budget
        params, n_params = freeze_to_final_layers(model, C.RO_TRAINABLE_LAYERS)
        x0 = get_flat_params(params).numpy()
        objective = _make_val_objective(model, params, val_loader, criterion)
        best_x, best_loss, hist = RO_ALGORITHMS[best_ro_algo](
            objective, x0, ro_budget, seed=seed)
        set_flat_params(params, torch.from_numpy(best_x).float())
        post = final_test_metrics(model, test_loader, criterion)

        rows.append({
            "seed": seed, "ro_algo": best_ro_algo, "ro_func_evals": hist[-1]["func_evals"],
            "macro_f1_pre_ro": pre.macro_f1, "macro_f1_post_ro": post.macro_f1,
            "bal_acc_pre_ro": pre.balanced_accuracy, "bal_acc_post_ro": post.balanced_accuracy,
            "val_loss_post_ro": best_loss,
        })
        save_json(hist, f"part4_ro_finetune_seed{seed}.json")

    df = pd.DataFrame(rows)
    save_table(df, "part4_integrated_results.csv")
    return df


if __name__ == "__main__":
    print(run_part4())
