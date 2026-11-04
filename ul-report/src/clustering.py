"""
Part 1 -- Clustering on the ORIGINAL Covertype feature space.

Fits K-Means and EM/GMM WITHOUT labels, then uses Cover_Type only afterwards as
external evidence (ARI/NMI/homogeneity/completeness/V-measure + contingency
table). Model selection (k, n_components, covariance type) is driven by INTERNAL
/ model-selection metrics only -- never by label agreement -- to keep the study
genuinely unsupervised (FAQ §5).

Internal metrics
  K-Means: inertia (elbow), silhouette, Davies-Bouldin, Calinski-Harabasz
  GMM    : log-likelihood, AIC, BIC (+ silhouette on hard assignments)
External metrics (post-hoc, both): ARI, NMI, homogeneity, completeness, V-measure
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from . import config as C
from .seeds import set_all_seeds


# ── external label-alignment (computed AFTER fitting only) ────────────────────
def external_scores(y_true: np.ndarray, labels: np.ndarray) -> dict:
    hom, com, vme = homogeneity_completeness_v_measure(y_true, labels)
    return {
        "ari": adjusted_rand_score(y_true, labels),
        "nmi": normalized_mutual_info_score(y_true, labels),
        "homogeneity": hom,
        "completeness": com,
        "v_measure": vme,
    }


def _silhouette(X: np.ndarray, labels: np.ndarray, seed: int) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    n = len(labels)
    if n > C.SILHOUETTE_SAMPLE:
        return float(silhouette_score(X, labels, sample_size=C.SILHOUETTE_SAMPLE,
                                      random_state=seed))
    return float(silhouette_score(X, labels))


def contingency_table(y_true: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """rows = clusters, cols = Cover_Type (0..6). Counts."""
    k = labels.max() + 1
    ct = np.zeros((k, C.N_CLASSES), dtype=int)
    for c, t in zip(labels, y_true):
        ct[c, t] += 1
    return ct


# ── K-Means sweep ─────────────────────────────────────────────────────────────
def run_kmeans_sweep(X: np.ndarray, y: np.ndarray, seed: int = 0) -> pd.DataFrame:
    rows = []
    for k in C.KMEANS_KS:
        set_all_seeds(seed)
        km = KMeans(n_clusters=k, init="k-means++", n_init=C.KMEANS_N_INIT,
                    max_iter=C.KMEANS_MAX_ITER, random_state=seed)
        labels = km.fit_predict(X)
        row = {
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": _silhouette(X, labels, seed),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        }
        row.update(external_scores(y, labels))
        rows.append(row)
    return pd.DataFrame(rows)


# ── GMM sweep ─────────────────────────────────────────────────────────────────
def run_gmm_sweep(X: np.ndarray, y: np.ndarray, seed: int = 0) -> pd.DataFrame:
    rows = []
    for cov in C.GMM_COV_TYPES:
        for n in C.GMM_COMPONENTS:
            set_all_seeds(seed)
            gmm = GaussianMixture(
                n_components=n, covariance_type=cov, n_init=C.GMM_N_INIT,
                max_iter=C.GMM_MAX_ITER, reg_covar=C.GMM_REG_COVAR,
                random_state=seed,
            )
            labels = gmm.fit_predict(X)
            row = {
                "cov_type": cov,
                "n_components": n,
                "log_likelihood": float(gmm.score(X) * len(X)),
                "aic": float(gmm.aic(X)),
                "bic": float(gmm.bic(X)),
                "converged": bool(gmm.converged_),
                "n_iter": int(gmm.n_iter_),
                "silhouette": _silhouette(X, labels, seed),
            }
            row.update(external_scores(y, labels))
            rows.append(row)
    return pd.DataFrame(rows)


# ── labels at a chosen setting (for contingency / embeddings) ─────────────────
def kmeans_labels(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    set_all_seeds(seed)
    return KMeans(n_clusters=k, init="k-means++", n_init=C.KMEANS_N_INIT,
                  max_iter=C.KMEANS_MAX_ITER, random_state=seed).fit_predict(X)


def gmm_labels(X: np.ndarray, n: int, cov: str, seed: int = 0):
    set_all_seeds(seed)
    gmm = GaussianMixture(n_components=n, covariance_type=cov, n_init=C.GMM_N_INIT,
                          max_iter=C.GMM_MAX_ITER, reg_covar=C.GMM_REG_COVAR,
                          random_state=seed)
    labels = gmm.fit_predict(X)
    resp = gmm.predict_proba(X)
    return labels, resp


@dataclass
class Selection:
    kmeans_k: int
    gmm_n: int
    gmm_cov: str


def select_settings(km_df: pd.DataFrame, gmm_df: pd.DataFrame) -> Selection:
    """Pick clustering settings by INTERNAL metrics only (label-free).

    K-Means: highest silhouette (compact/separated geometry).
    GMM     : lowest BIC (model selection with complexity penalty).
    """
    km_best = km_df.loc[km_df["silhouette"].idxmax()]
    gmm_best = gmm_df.loc[gmm_df["bic"].idxmin()]
    return Selection(
        kmeans_k=int(km_best["k"]),
        gmm_n=int(gmm_best["n_components"]),
        gmm_cov=str(gmm_best["cov_type"]),
    )
