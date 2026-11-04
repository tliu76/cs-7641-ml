"""
Part 1: Randomized Optimization (RHC, SA, GA) on the final 1-3 layers.

Setup:
  1. Train (or load) the SGD baseline to get a trained backbone.
  2. Freeze all but the final RO_TRAINABLE_LAYERS layers (<= RO_PARAM_CAP params).
  3. Optimize VALIDATION LOSS over the flat trainable vector with each RO method,
     under a shared FUNCTION-EVALUATION budget, deterministic eval (model.eval()).
  4. Write the best vector back, evaluate on the held-out test set.

Required analysis (build from the returned artifacts):
  - best-so-far validation loss vs function evaluations  (plot_best_so_far)
  - validation loss vs wall-clock time
  - final test Accuracy / Macro-F1 / balanced accuracy / class-level behavior
  - compute trade-off vs the SGD baseline and Adam
  - whether RO improved performance, improved stability, or mainly burned compute
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import get_split, make_loaders
from .model import build_model, freeze_to_final_layers, get_flat_params, set_flat_params
from .plotting import plot_best_so_far, save_json, save_table
from .ro_algorithms import RO_ALGORITHMS
from .seeds import set_all_seeds
from .train import evaluate, final_test_metrics, make_criterion_for, train_model


def _make_val_objective(model, params, val_loader, criterion):
    """Return objective(vector)->validation loss. Each call == 1 function eval."""
    def objective(vec: np.ndarray) -> float:
        set_flat_params(params, torch.from_numpy(vec).float())
        loss, _, _ = evaluate(model, val_loader, criterion)  # model.eval() inside
        return loss
    return objective


def run_part1(seeds=None, algorithms=("rhc", "sa", "ga"),
              budget: int = C.RO_FUNC_EVAL_BUDGET) -> pd.DataFrame:
    seeds = seeds or C.SEEDS
    split = get_split()
    criterion = make_criterion_for(split)
    rows, all_histories = [], {}

    for seed in seeds:
        set_all_seeds(seed)
        train_loader, val_loader, test_loader = make_loaders(split, seed=seed)

        # 1) trained backbone == the SL SGD-only baseline (same optimizer/LR), so
        #    RO fine-tunes the final layer of the exact OL reference network.
        model = build_model(dropout_p=0.0)
        opt = torch.optim.SGD([p for p in model.parameters()], lr=C.BASELINE_LR, momentum=0.0)
        train_model(model, train_loader, val_loader, opt, criterion, seed=seed)

        # 2) freeze to final layers under the parameter cap
        params, n_params = freeze_to_final_layers(model, C.RO_TRAINABLE_LAYERS)
        x0 = get_flat_params(params).numpy()

        # 3) optimize validation loss with each RO method
        objective = _make_val_objective(model, params, val_loader, criterion)
        for algo in algorithms:
            set_all_seeds(seed)
            best_x, best_loss, hist = RO_ALGORITHMS[algo](objective, x0, budget, seed=seed)
            all_histories[f"{algo}-s{seed}"] = hist

            # 4) write best vector back, evaluate on TEST
            set_flat_params(params, torch.from_numpy(best_x).float())
            test_m = final_test_metrics(model, test_loader, criterion)
            row = {"algo": algo, "seed": seed, "n_ro_params": n_params,
                   "func_evals": hist[-1]["func_evals"],
                   "wall_clock_s": hist[-1]["wall_clock"], "best_val_loss": best_loss}
            row.update(test_m.summary_row())
            rows.append(row)
            save_json(hist, f"part1_{algo}_seed{seed}_history.json")

    df = pd.DataFrame(rows)
    save_table(df, "part1_ro_results.csv")
    # One compact best-so-far figure per algorithm (seed 0 shown; others in REPRO).
    plot_best_so_far({k: v for k, v in all_histories.items() if k.endswith("s0")},
                     "part1_best_so_far.png", title="RO best-so-far (validation loss)")
    return df


if __name__ == "__main__":
    print(run_part1())
