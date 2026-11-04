#!/usr/bin/env python3
"""Part 4 standalone re-run: NN after DR under the SL/OL-faithful recipe.

Reuses the component counts Part 2 already justified (part2_component_choice.json)
so nothing is re-selected here. Honours config.NN_EARLY_STOP_PATIENCE, which now
defaults to None (fixed 100-epoch budget, the OL Part 2/3 protocol) because OL
measured val-loss early stopping as harmful to Macro-F1. Set UL_EARLY_STOP=1 to
reproduce the early-stopping variant instead.

Outputs are suffixed by regime so both variants can coexist:
    part4_nn_per_seed[_es].csv, part4_nn_summary[_es].csv,
    part4_nn_dim_sweep[_es].csv, part4_majority_baseline.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as C
from src import nn_dr as NN
from src import plotting as P
from src.seeds import set_all_seeds


def main() -> None:
    t0 = time.time()
    set_all_seeds(C.SEEDS[0])
    es = C.NN_EARLY_STOP_PATIENCE is not None
    tag = "_es" if es else ""
    regime = (f"early stopping (patience {C.NN_EARLY_STOP_PATIENCE})" if es
              else f"fixed {C.NN_EPOCHS}-epoch budget")
    jc = json.load(open(os.path.join(C.RESULT_DIR, "part2_component_choice.json")))
    print(f"[part4] regime = {regime}\n[part4] dims = "
          f"PCA {jc['pca']}, ICA {jc['ica']}, RP {jc['rp']}", flush=True)

    df = NN.run_nn_after_dr(jc, seeds=C.SEEDS)
    P.save_table(df, f"part4_nn_per_seed{tag}.csv")
    summ = NN.summarize_nn(df)
    P.save_table(summ, f"part4_nn_summary{tag}.csv")
    print("\n" + summ.to_string(index=False), flush=True)

    base = NN.majority_class_baseline()
    P.save_json(base, "part4_majority_baseline.json")
    print(f"\n[part4] majority floor: acc={base['accuracy']:.3f} "
          f"macroF1={base['macro_f1']:.3f}", flush=True)

    sweep = NN.run_nn_dim_sweep(dims=C.NN_DR_SWEEP_DIMS, seeds=C.SEEDS)
    P.save_table(sweep, f"part4_nn_dim_sweep{tag}.csv")
    agg = sweep.groupby(["representation", "n_components"]).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        acc_mean=("accuracy", "mean"), fit_s=("fit_time_s", "mean")).reset_index()
    print("\n" + agg.round(4).to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for rep, g in agg.groupby("representation"):
        ax.errorbar(g["n_components"], g["macro_f1_mean"], yerr=g["macro_f1_std"],
                    marker="o", capsize=3, label=rep)
    orig = summ[summ["representation"] == "original"]
    if len(orig):
        ax.axhline(float(orig["macro_f1_mean"].iloc[0]), color="gray", ls="--",
                   lw=1, label="original (54-d)")
    ax.set(xlabel="n_components", ylabel="test Macro-F1",
           title=f"NN Macro-F1 vs DR dimension ({regime})")
    ax.legend(fontsize=8)
    P.save_fig(fig, f"part4_nn_dim_sweep{tag}.png")

    print(f"\n[done] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
