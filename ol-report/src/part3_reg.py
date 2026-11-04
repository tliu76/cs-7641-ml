"""
Part 3: Regularization study with STANDARD ADAM (best Part-2 HPs, kept fixed).

Required techniques: L2 weight decay (coupled), early stopping, dropout,
label smoothing, and modest train-only input noise/augmentation.

Required comparisons: Adam baseline (no added reg), best SINGLE regularizer,
best COMBINATION. Selection uses VALIDATION evidence only; test is final-only.

Required analysis: val loss + test metrics under a fixed budget; whether reg
improved Macro-F1 / balanced acc / minority-class recall / val loss / stability;
whether it reduced overfitting (train-val gap); whether any class was harmed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import get_split, make_loaders
from .model import build_model
from .plotting import save_json, save_table
from .seeds import set_all_seeds
from .train import final_test_metrics, make_criterion_for, train_model

# Best Part-2 Adam hyperparameters — standard Adam kept FIXED for the whole part
# (assignment: do not retune Adam or switch to AdamW inside Part 3).
ADAM_LR = 1e-3
ADAM_BETAS = (0.9, 0.999)


def _train_one(reg: dict, seed: int, split, epochs: int = C.EPOCHS):
    """Train standard Adam with one regularization config. `reg` keys:
    weight_decay, dropout, label_smoothing, input_noise, early_stop (bool)."""
    set_all_seeds(seed)
    train_loader, val_loader, test_loader = make_loaders(split, seed=seed)
    model = build_model(dropout_p=reg.get("dropout", 0.0))
    criterion = make_criterion_for(split, label_smoothing=reg.get("label_smoothing", 0.0))
    opt = torch.optim.Adam(
        [p for p in model.parameters()], lr=ADAM_LR, betas=ADAM_BETAS,
        weight_decay=reg.get("weight_decay", 0.0),   # coupled L2 (standard Adam)
    )
    log = train_model(
        model, train_loader, val_loader, opt, criterion, epochs=epochs, seed=seed,
        input_noise_std=reg.get("input_noise", 0.0),
        early_stop_patience=C.EARLY_STOP_PATIENCE if reg.get("early_stop") else None,
    )
    test_m = final_test_metrics(model, test_loader, criterion)
    gap = (log.val_loss[-1] - log.train_loss[-1]) if log.val_loss else None
    return log, test_m, gap


# One representative config per regularizer family for the single-regularizer
# sweep (family -> {label: reg-dict}). Detailed per-value grids go to REPRO.
def _single_configs() -> dict:
    configs = {"adam_baseline": {}}
    for wd in C.WEIGHT_DECAY_GRID:
        configs[f"l2_{wd:g}"] = {"weight_decay": wd, "_family": "l2"}
    for dp in C.DROPOUT_GRID:
        configs[f"dropout_{dp:g}"] = {"dropout": dp, "_family": "dropout"}
    for ls in C.LABEL_SMOOTHING_GRID:
        configs[f"labelsmooth_{ls:g}"] = {"label_smoothing": ls, "_family": "labelsmooth"}
    for ns in C.INPUT_NOISE_GRID:
        configs[f"inputnoise_{ns:g}"] = {"input_noise": ns, "_family": "inputnoise"}
    configs["early_stop"] = {"early_stop": True, "_family": "early_stop"}
    return configs


def _clean(reg: dict) -> dict:
    """Strip bookkeeping keys (_family) before passing a reg-dict to training."""
    return {k: v for k, v in reg.items() if not k.startswith("_")}


def run_part3(seeds=None) -> pd.DataFrame:
    """Adam baseline + single-regularizer sweep, then a DATA-DRIVEN best combo.

    Selection uses validation loss (mean across seeds) only; the held-out test
    set is reported but never used for selection. The chosen best single
    regularizer and best combination are persisted to part3_selection.json so
    Part 4 can consume them without hand-editing.
    """
    seeds = seeds or C.SEEDS
    split = get_split()
    configs = _single_configs()

    rows = []
    for name, reg in configs.items():
        for seed in seeds:
            log, test_m, gap = _train_one(_clean(reg), seed, split)
            row = {"config": name, "seed": seed, "gen_gap": gap,
                   "grad_evals": log.total_grad_evals, "status": log.status,
                   "val_loss_final": log.val_loss[-1] if log.val_loss else None}
            row.update(test_m.summary_row())
            rows.append(row)
            save_json(log.to_dict(), f"part3_{name}_seed{seed}.json")

    df = pd.DataFrame(rows)

    # ── Data-driven selection (validation loss only) ──────────────────────────
    mean_val = df.groupby("config")["val_loss_final"].mean()
    baseline_val = mean_val.get("adam_baseline", np.inf)
    singles = mean_val.drop(labels=["adam_baseline"], errors="ignore")
    best_single_name = singles.idxmin()
    best_single_reg = _clean(configs[best_single_name])

    # Best combination: take the best value within each family, but keep only the
    # families whose best member actually beat the Adam baseline on val loss.
    best_combo, combo_members = {}, []
    for fam in ("l2", "dropout", "labelsmooth", "inputnoise"):
        fam_cfgs = {n: c for n, c in configs.items() if c.get("_family") == fam}
        if not fam_cfgs:
            continue
        fam_best = min(fam_cfgs, key=lambda n: mean_val.get(n, np.inf))
        if mean_val.get(fam_best, np.inf) < baseline_val:
            best_combo.update(_clean(configs[fam_best]))
            combo_members.append(fam_best)

    # Fall back to the single best regularizer if no family beat the baseline.
    if not best_combo:
        best_combo = dict(best_single_reg)
        combo_members = [best_single_name]

    for seed in seeds:
        log, test_m, gap = _train_one(best_combo, seed, split)
        row = {"config": "best_combo", "seed": seed, "gen_gap": gap,
               "grad_evals": log.total_grad_evals, "status": log.status,
               "val_loss_final": log.val_loss[-1] if log.val_loss else None}
        row.update(test_m.summary_row())
        rows.append(row)
        save_json(log.to_dict(), f"part3_best_combo_seed{seed}.json")

    df = pd.DataFrame(rows)
    save_table(df, "part3_regularization_results.csv")
    save_json(
        {"best_single": {"name": best_single_name, "reg": best_single_reg},
         "best_combo": {"members": combo_members, "reg": best_combo},
         "adam_baseline_mean_val_loss": float(baseline_val),
         "adam_lr": ADAM_LR, "adam_betas": list(ADAM_BETAS)},
        "part3_selection.json",
    )
    print(f"[part3] best single = {best_single_name} {best_single_reg}")
    print(f"[part3] best combo  = {combo_members} {best_combo}")
    return df


if __name__ == "__main__":
    print(run_part3())
