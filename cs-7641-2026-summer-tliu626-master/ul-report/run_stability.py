#!/usr/bin/env python3
"""Part 3 add-on: clustering stability per representation.

Reads the settings Part 3 already selected (outputs/results/part3_dr_clustering.csv)
and re-runs those exact configurations across seeds to measure how reproducible
each clustering is. Writes:

    outputs/results/part3_stability_algorithm.csv
    outputs/results/part3_stability_representation.csv
    outputs/figures/part3_stability.png

Run:  python run_stability.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src import plotting as P
from src import stability as S
from src.seeds import set_all_seeds

SEEDS = [0, 1, 2, 3, 4]        # clustering-initialization seeds
REP_SEEDS = [0, 1, 2, 3, 4]    # DR-transform seeds (ICA / RP)


def main() -> None:
    t0 = time.time()
    set_all_seeds(0)
    work = D.get_working_sample()
    part3 = pd.read_csv(os.path.join(C.RESULT_DIR, "part3_dr_clustering.csv"))
    print(f"[stab] X={work.X.shape}  spaces={list(part3['space'])}  "
          f"seeds={SEEDS}  rep_seeds={REP_SEEDS}", flush=True)

    algo, rep = S.run_stability(work.X, work.y, part3,
                                seeds=SEEDS, rep_seeds=REP_SEEDS)
    P.save_table(algo, "part3_stability_algorithm.csv")
    P.save_table(rep, "part3_stability_representation.csv")
    print("\n=== algorithm stability (same representation, 5 clustering seeds) ===")
    print(algo.to_string(index=False))
    print("\n=== representation stability (5 redrawn transforms) ===")
    print(rep.to_string(index=False))

    # Figure: self-agreement (label-free) next to label alignment, per space.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(algo))
    axes[0].bar(x - 0.2, algo["km_self_ari"], 0.4, label="K-Means")
    axes[0].bar(x + 0.2, algo["gmm_self_ari"], 0.4, label="GMM")
    axes[0].axhline(1.0, color="gray", ls="--", lw=1)
    axes[0].set_xticks(x, algo["space"])
    axes[0].set(ylabel="mean pairwise ARI between runs", ylim=(0, 1.05),
                title="Self-agreement across clustering seeds")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - 0.2, algo["km_label_ari_mean"], 0.4,
                yerr=algo["km_label_ari_std"], capsize=3, label="K-Means")
    axes[1].bar(x + 0.2, algo["gmm_label_ari_mean"], 0.4,
                yerr=algo["gmm_label_ari_std"], capsize=3, label="GMM")
    axes[1].set_xticks(x, algo["space"])
    axes[1].set(ylabel="ARI vs Cover_Type", title="Label alignment (mean $\\pm$ s.d.)")
    axes[1].legend(fontsize=8)
    P.save_fig(fig, "part3_stability.png")

    print(f"\n[done] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
