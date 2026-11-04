"""
Part 3 -- Clustering AFTER dimensionality reduction.

Reruns both clustering algorithms on each DR representation (PCA, ICA, RP),
giving the six required combinations: {K-Means, GMM} x {PCA, ICA, RP}. For each
representation we RESELECT the clustering setting by the same internal criterion
used in Part 1 (K-Means: max silhouette over the k sweep; GMM: min BIC over the
component sweep at a fixed covariance type), rather than blindly reusing the
original-space setting -- the transformed geometry can support a different k
(Part 3 requirement). We then report internal quality and post-hoc label
alignment so original-space vs DR-space clustering can be compared in one table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import clustering as CL
from . import config as C
from . import dr as DR


def _best_kmeans(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    df = CL.run_kmeans_sweep(X, y, seed=seed)
    best = df.loc[df["silhouette"].idxmax()]
    return {
        "km_k": int(best["k"]),
        "km_silhouette": float(best["silhouette"]),
        "km_davies_bouldin": float(best["davies_bouldin"]),
        "km_ari": float(best["ari"]),
        "km_nmi": float(best["nmi"]),
    }


def _best_gmm(X: np.ndarray, y: np.ndarray, seed: int, cov_types=None) -> dict:
    """Reselect BOTH the component count AND the covariance type by BIC.

    DR changes the feature geometry, so the transformed space may favour a
    different covariance structure than the original space (Part 3 requirement:
    reselect, do not blindly reuse). We sweep (n_components x covariance_type) and
    pick the min-BIC model, then report its post-hoc label alignment.
    """
    from sklearn.mixture import GaussianMixture
    cov_types = cov_types or C.GMM_COV_TYPES
    rows = []
    for cov in cov_types:
        for n in C.GMM_COMPONENTS:
            CL.set_all_seeds(seed)
            gmm = GaussianMixture(n_components=n, covariance_type=cov,
                                  n_init=C.GMM_N_INIT, max_iter=C.GMM_MAX_ITER,
                                  reg_covar=C.GMM_REG_COVAR, random_state=seed)
            labels = gmm.fit_predict(X)
            ext = CL.external_scores(y, labels)
            rows.append({"n": n, "cov": cov, "bic": float(gmm.bic(X)),
                         "ari": ext["ari"], "nmi": ext["nmi"]})
    df = pd.DataFrame(rows)
    best = df.loc[df["bic"].idxmin()]
    return {
        "gmm_n": int(best["n"]),
        "gmm_cov": str(best["cov"]),
        "gmm_bic": float(best["bic"]),
        "gmm_ari": float(best["ari"]),
        "gmm_nmi": float(best["nmi"]),
    }


def _transform(space: str, X: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    if space == "PCA":
        return DR.fit_pca(X, n_components, seed).transform(X)
    if space == "ICA":
        return DR.fit_ica(X, n_components, seed).transform(X)
    if space == "RP":
        return DR.fit_rp(X, n_components, seed).transform(X)
    raise ValueError(space)


def run_dr_clustering(X: np.ndarray, y: np.ndarray, components,
                      seed: int = 0) -> pd.DataFrame:
    """One row per representation (Original, PCA, ICA, RP) with KM + GMM summary.

    ``components`` may be a single int (same dim for all methods) or a dict
    {"pca":.., "ica":.., "rp":..} of per-method justified dims. Both K-Means (k)
    and GMM (n_components + covariance type) are RESELECTED in each space.
    """
    if isinstance(components, dict):
        dims = {s: components[s.lower()] for s in ("PCA", "ICA", "RP")}
    else:
        dims = {s: int(components) for s in ("PCA", "ICA", "RP")}

    rows = []
    row = {"space": "Original", "n_dim": X.shape[1]}
    row.update(_best_kmeans(X, y, seed))
    row.update(_best_gmm(X, y, seed))
    rows.append(row)

    for space in ("PCA", "ICA", "RP"):
        Z = _transform(space, X, dims[space], seed)
        row = {"space": space, "n_dim": Z.shape[1]}
        row.update(_best_kmeans(Z, y, seed))
        row.update(_best_gmm(Z, y, seed))
        rows.append(row)

    return pd.DataFrame(rows)
