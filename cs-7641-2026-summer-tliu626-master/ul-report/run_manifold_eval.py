#!/usr/bin/env python3
"""Extra-credit add-on: quantify the nonlinear embedding instead of eyeballing it.

Embeds the same subsample with the nonlinear method and with PCA-2D, then scores
both with trustworthiness (label-free local-structure preservation), k-NN label
agreement, and K-Means ARI/NMI. Adds the original 54-d space as the reference
ceiling. Writes outputs/results/extra_manifold_metrics.csv.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src import manifold as M
from src import plotting as P
from src.seeds import set_all_seeds


def main() -> None:
    t0 = time.time()
    set_all_seeds(C.SEEDS[0])
    work = D.get_working_sample()
    res = M.run_manifold(work.X, work.y, seed=C.SEEDS[0])

    n = res["n"]
    rng = np.random.default_rng(C.SEEDS[0])
    idx = rng.choice(len(work.X), size=n, replace=False)   # same draw as run_manifold
    Xs, ys = work.X[idx], work.y[idx]

    rows = []
    for name, emb in ((res["method"], res["embedding"]), ("PCA-2D", res["pca2"])):
        d = M.embedding_diagnostics(Xs, emb, ys, seed=C.SEEDS[0])
        d["space"] = name
        rows.append(d)
        print(f"[eval] {name:8s} trust={d['trustworthiness']:.3f} "
              f"kNN-agree={d['knn_label_agreement']:.3f} "
              f"ARI={d['kmeans_ari']:.3f} NMI={d['kmeans_nmi']:.3f}", flush=True)

    # reference ceiling: same diagnostics computed in the original 54-d space
    d = M.embedding_diagnostics(Xs, Xs, ys, seed=C.SEEDS[0])
    d["space"] = "original-54d"
    rows.append(d)
    print(f"[eval] {'orig-54d':8s} trust={d['trustworthiness']:.3f} "
          f"kNN-agree={d['knn_label_agreement']:.3f} "
          f"ARI={d['kmeans_ari']:.3f} NMI={d['kmeans_nmi']:.3f}", flush=True)

    df = pd.DataFrame(rows)[["space", "trustworthiness", "knn_label_agreement",
                             "kmeans_ari", "kmeans_nmi"]]
    P.save_table(df, "extra_manifold_metrics.csv")
    print("\n" + df.round(4).to_string(index=False))
    print(f"\n[done] n={n} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
