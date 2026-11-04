#!/usr/bin/env python3
"""Part 4b only: NN Macro-F1 vs component count, under the fixed-epoch recipe.

Split out from run_part4.py because the main comparison is already saved and only
this sweep needs re-running. Writes incrementally so a walltime kill still leaves
usable partial results instead of nothing.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src import config as C
from src import data as D
from src import plotting as P
from src.model import train_eval_nn
from src.nn_dr import _fit_transform_dr
from src.seeds import set_all_seeds

OUT = os.path.join(C.RESULT_DIR, "part4_nn_dim_sweep.csv")


def main() -> None:
    t0 = time.time()
    set_all_seeds(C.SEEDS[0])
    split = D.get_split()
    Xtr, Xva, Xte = split.X_train, split.X_val, split.X_test
    ytr, yva, yte = split.y_train, split.y_val, split.y_test
    print(f"[sweep] fixed {C.NN_EPOCHS}-epoch budget, "
          f"early_stop={C.NN_EARLY_STOP_PATIENCE}", flush=True)

    rows = []
    for kind in ("pca", "ica", "rp"):
        for d in C.NN_DR_SWEEP_DIMS:
            for seed in C.SEEDS:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Ztr, Zva, Zte = _fit_transform_dr(kind, d, Xtr, Xva, Xte, seed)
                r = train_eval_nn(Ztr, ytr, Zva, yva, Zte, yte, seed=seed)
                rows.append({"representation": kind, "n_components": d,
                             "seed": seed, "accuracy": r.metrics.accuracy,
                             "macro_f1": r.metrics.macro_f1,
                             "balanced_accuracy": r.metrics.balanced_accuracy,
                             "fit_time_s": r.fit_time_s})
            # checkpoint after every (method, d) cell
            pd.DataFrame(rows).to_csv(OUT, index=False)
            m = pd.DataFrame(rows).query(
                "representation==@kind and n_components==@d")["macro_f1"].mean()
            print(f"[sweep] {kind:>4s} d={d:>2d} macroF1={m:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    P.save_table(df, "part4_nn_dim_sweep.csv")
    print("\n" + df.groupby(["representation", "n_components"])["macro_f1"]
          .agg(["mean", "std"]).round(4).to_string())
    print(f"\n[done] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
