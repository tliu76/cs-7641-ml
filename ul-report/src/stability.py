"""
Part 3 (stability) -- how reproducible is a clustering in each representation?

Table `part3_dr_clustering.csv` reports cluster QUALITY and LABEL ALIGNMENT for
one seed per space. That leaves the third question the assignment asks about
DR + clustering unanswered: did PCA/ICA/RP make the clusters more or less
*stable*? Instability has two distinct sources here and they are worth separating:

  algorithm stability   -- refit the SAME clustering on the SAME representation
                           with different clustering seeds. Measures sensitivity
                           to initialization, given the n_init restarts that the
                           reported pipeline actually uses.

  representation stability -- refit the DR TRANSFORM with a different seed, then
                           recluster. PCA is deterministic (identically 1.0 by
                           construction); ICA has a sign/permutation freedom; RP
                           draws a whole new random matrix. This is the source
                           that only exists after DR.

The headline metric is label-free: mean pairwise Adjusted Rand Index BETWEEN the
resulting partitions. It asks "do two runs put the same points together?" without
consulting Cover_Type, so it is a legitimate unsupervised diagnostic. Cover_Type
enters only afterwards, to report the spread of external alignment across runs.

Cluster settings are held at the values Table `part3_dr_clustering.csv` selected
for each space -- we are measuring the stability of the reported configuration,
not searching for a new one.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from . import clustering as CL
from . import config as C
from . import dr as DR
from .seeds import set_all_seeds


# ── label-free agreement between repeated runs ────────────────────────────────
def mean_pairwise_ari(partitions: list[np.ndarray]) -> float:
    """Mean ARI over all unordered pairs of partitions. 1.0 = identical grouping.

    Label-free: compares clusterings to each other, never to Cover_Type.
    """
    pairs = list(itertools.combinations(range(len(partitions)), 2))
    if not pairs:
        return float("nan")
    return float(np.mean([adjusted_rand_score(partitions[i], partitions[j])
                          for i, j in pairs]))


def _kmeans_partition(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return CL.kmeans_labels(X, k, seed=seed)


def _gmm_partition(X: np.ndarray, n: int, cov: str, seed: int) -> np.ndarray:
    set_all_seeds(seed)
    gmm = GaussianMixture(n_components=n, covariance_type=cov,
                          n_init=C.GMM_N_INIT, max_iter=C.GMM_MAX_ITER,
                          reg_covar=C.GMM_REG_COVAR, random_state=seed)
    return gmm.fit_predict(X)


def _transform(space: str, X: np.ndarray, d: int, seed: int) -> np.ndarray:
    if space == "Original":
        return X
    if space == "PCA":
        return DR.fit_pca(X, d, seed).transform(X)
    if space == "ICA":
        return DR.fit_ica(X, d, seed).transform(X)
    if space == "RP":
        return DR.fit_rp(X, d, seed).transform(X)
    raise ValueError(space)


# ── algorithm stability: same representation, different clustering seeds ──────
def algorithm_stability(X: np.ndarray, y: np.ndarray, space: str, d: int,
                        km_k: int, gmm_n: int, gmm_cov: str,
                        seeds: list[int], rep_seed: int = 0) -> dict:
    Z = _transform(space, X, d, rep_seed)

    km_parts, km_ari = [], []
    for s in seeds:
        p = _kmeans_partition(Z, km_k, s)
        km_parts.append(p)
        km_ari.append(adjusted_rand_score(y, p))       # post-hoc only

    gm_parts, gm_ari = [], []
    for s in seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = _gmm_partition(Z, gmm_n, gmm_cov, s)
        gm_parts.append(p)
        gm_ari.append(adjusted_rand_score(y, p))

    return {
        "space": space, "n_dim": Z.shape[1],
        "km_self_ari": mean_pairwise_ari(km_parts),
        "km_label_ari_mean": float(np.mean(km_ari)),
        "km_label_ari_std": float(np.std(km_ari)),
        "gmm_self_ari": mean_pairwise_ari(gm_parts),
        "gmm_label_ari_mean": float(np.mean(gm_ari)),
        "gmm_label_ari_std": float(np.std(gm_ari)),
    }


# ── representation stability: refit the DR transform, then recluster ──────────
def representation_stability(X: np.ndarray, y: np.ndarray, space: str, d: int,
                             km_k: int, rep_seeds: list[int]) -> dict:
    """Redraw the transform per seed and ask whether the clustering survives.

    Also records the k that maximum silhouette would select in each redrawn
    representation: a method that cannot agree with itself about how many
    clusters exist is unstable in a way a fixed-k comparison would hide.
    """
    parts, ks, sils = [], [], []
    for s in rep_seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = _transform(space, X, d, s)
        parts.append(_kmeans_partition(Z, km_k, 0))     # clustering seed fixed
        sweep = CL.run_kmeans_sweep(Z, y, seed=0)       # y used only post-hoc
        best = sweep.loc[sweep["silhouette"].idxmax()]
        ks.append(int(best["k"]))
        sils.append(float(best["silhouette"]))

    return {
        "space": space,
        "km_self_ari_across_reps": mean_pairwise_ari(parts),
        "selected_k_values": "/".join(str(k) for k in ks),
        "selected_k_unique": len(set(ks)),
        "silhouette_mean": float(np.mean(sils)),
        "silhouette_std": float(np.std(sils)),
    }


def run_stability(X: np.ndarray, y: np.ndarray, part3: pd.DataFrame,
                  seeds: list[int], rep_seeds: list[int]) -> tuple:
    """Both tables, driven by the settings Part 3 already selected per space."""
    algo_rows, rep_rows = [], []
    for _, r in part3.iterrows():
        space = str(r["space"])
        d = int(r["n_dim"])
        algo_rows.append(algorithm_stability(
            X, y, space, d, int(r["km_k"]), int(r["gmm_n"]), str(r["gmm_cov"]),
            seeds=seeds))
        print(f"[stab] {space:>8s} algorithm-stability done", flush=True)

        if space in ("ICA", "RP"):        # PCA deterministic; Original has no transform
            rep_rows.append(representation_stability(
                X, y, space, d, int(r["km_k"]), rep_seeds=rep_seeds))
            print(f"[stab] {space:>8s} representation-stability done", flush=True)

    return pd.DataFrame(algo_rows), pd.DataFrame(rep_rows)
