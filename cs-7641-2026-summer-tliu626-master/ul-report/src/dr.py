"""
Part 2 -- Dimensionality reduction: PCA, ICA, Randomized Projections.

Each method is characterized with method-appropriate diagnostics (FAQ §8):

  PCA : explained-variance ratio + cumulative curve, reconstruction error vs
        n_components, effective rank, top loadings per component.
  ICA : per-component excess kurtosis (non-Gaussianity), convergence behaviour,
        seed stability via absolute-correlation component matching.
  RP  : pairwise-distance preservation (mean/std distortion) across multiple
        seeds; Gaussian vs sparse projection.

For the exploratory analysis these transforms are fit on the whole working
sample (labels unused). The leakage-safe, train-only fitting for the downstream
NN lives in nn_dr.py.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)

from sklearn.random_projection import johnson_lindenstrauss_min_dim

from . import config as C
from .seeds import set_all_seeds


# ── Per-method component-count justification (FAQ §8) ──────────────────────────
def justified_components(X: np.ndarray, seed: int = 0) -> dict:
    """Choose each method's n_components by METHOD-APPROPRIATE evidence.

    PCA : smallest k reaching PCA_VAR_TARGET cumulative explained variance.
    ICA : the PCA-retained rank at the same variance target -- FastICA whitens to
          a k-dim subspace, so independent components beyond the informative rank
          only add noise/instability; we justify ICA's k by that retained rank.
    RP  : the Johnson-Lindenstrauss min-dim for n points at eps=RP_JL_EPS is the
          theoretical reference; it is huge for a strict eps, so we instead pick
          the smallest sweep dim whose across-seed mean distortion std falls below
          eps, and report the JL number as context.
    """
    pca = PCA(svd_solver="full", random_state=seed).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k_pca = int(np.searchsorted(cum, C.PCA_VAR_TARGET) + 1)

    jl_dim = int(johnson_lindenstrauss_min_dim(n_samples=len(X), eps=C.RP_JL_EPS))
    k_rp = None
    for d in C.DR_COMPONENTS:
        rp = rp_distance_preservation(X, d, seeds=C.RP_SEEDS, kind=C.RP_KIND)
        # distortion = spread of the per-pair ratio around 1, averaged over seeds
        mean_std = float(rp["per_seed"]["std_ratio"].mean())
        if mean_std <= C.RP_JL_EPS:
            k_rp = d
            break
    if k_rp is None:
        k_rp = max(C.DR_COMPONENTS)

    return {
        "pca": k_pca,
        "ica": k_pca,               # justified by the same retained rank
        "rp": k_rp,
        "jl_min_dim": jl_dim,
        "jl_eps": C.RP_JL_EPS,
        "pca_var_target": C.PCA_VAR_TARGET,
    }


# ── PCA ───────────────────────────────────────────────────────────────────────
def pca_diagnostics(X: np.ndarray, seed: int = 0) -> dict:
    """Full-rank PCA for the scree/cumulative curve + reconstruction sweep."""
    set_all_seeds(seed)
    pca = PCA(svd_solver="full", random_state=seed).fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    k_target = int(np.searchsorted(cum, C.PCA_VAR_TARGET) + 1)

    # Reconstruction error vs component count (squared error, standardized space).
    recon = []
    for k in C.DR_COMPONENTS:
        Xr = PCA(n_components=k, svd_solver="full", random_state=seed).fit(X)
        Z = Xr.transform(X)
        Xhat = Xr.inverse_transform(Z)
        recon.append({"k": k, "recon_mse": float(np.mean((X - Xhat) ** 2)),
                      "cum_evr": float(np.cumsum(Xr.explained_variance_ratio_)[-1])})

    # Effective rank (participation ratio of the eigenvalue spectrum).
    ev = pca.explained_variance_
    eff_rank = float((ev.sum() ** 2) / np.sum(ev ** 2))

    return {
        "evr": evr,
        "cum_evr": cum,
        "k_target": k_target,
        "var_target": C.PCA_VAR_TARGET,
        "effective_rank": eff_rank,
        "recon": pd.DataFrame(recon),
        "components": pca.components_,      # (n_features, n_features)
    }


def pca_top_loadings(components: np.ndarray, feature_names: list[str],
                     n_comp: int = 3, top: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n_comp):
        load = components[i]
        order = np.argsort(np.abs(load))[::-1][:top]
        for j in order:
            rows.append({"component": f"PC{i+1}", "feature": feature_names[j],
                         "loading": float(load[j])})
    return pd.DataFrame(rows)


def fit_pca(X: np.ndarray, n_components: int, seed: int = 0) -> PCA:
    return PCA(n_components=n_components, svd_solver="full", random_state=seed).fit(X)


# ── ICA ───────────────────────────────────────────────────────────────────────
def _abs_excess_kurtosis(Z: np.ndarray) -> np.ndarray:
    return np.abs(kurtosis(Z, axis=0, fisher=True, bias=False))


def ica_diagnostics(X: np.ndarray, n_components: int,
                    seeds=None) -> dict:
    """FastICA across seeds: kurtosis distribution, convergence, stability."""
    seeds = seeds or C.SEEDS
    per_seed_kurt, converged_flags, iters, transforms = [], [], [], []
    for s in seeds:
        set_all_seeds(s)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            ica = FastICA(n_components=n_components, fun=C.ICA_FUN,
                          max_iter=C.ICA_MAX_ITER, tol=C.ICA_TOL,
                          whiten="unit-variance", random_state=s)
            Z = ica.fit_transform(X)
        did_converge = not any("did not converge" in str(w.message).lower()
                               for w in wlist)
        per_seed_kurt.append(_abs_excess_kurtosis(Z))
        converged_flags.append(did_converge)
        iters.append(int(ica.n_iter_))
        transforms.append(Z)

    # Stability: match components across the first two seeds by |correlation|.
    stability = float("nan")
    if len(transforms) >= 2:
        A, B = transforms[0], transforms[1]
        corr = np.abs(np.corrcoef(A.T, B.T)[:n_components, n_components:])
        stability = float(np.mean(corr.max(axis=1)))  # best match per component

    kurt_arr = np.array(per_seed_kurt)     # (n_seeds, n_components)
    return {
        "n_components": n_components,
        "kurtosis_mean": kurt_arr.mean(0),
        "kurtosis_per_seed": kurt_arr,
        "converged": converged_flags,
        "n_iter": iters,
        "stability_abscorr": stability,
        "seeds": list(seeds),
    }


def fit_ica(X: np.ndarray, n_components: int, seed: int = 0):
    set_all_seeds(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=n_components, fun=C.ICA_FUN,
                      max_iter=C.ICA_MAX_ITER, tol=C.ICA_TOL,
                      whiten="unit-variance", random_state=seed)
        ica.fit(X)
    return ica


# ── Randomized Projections ────────────────────────────────────────────────────
def _make_rp(n_components: int, seed: int, kind: str):
    if kind == "sparse":
        return SparseRandomProjection(n_components=n_components, random_state=seed)
    return GaussianRandomProjection(n_components=n_components, random_state=seed)


def _pairwise_dist_sample(X: np.ndarray, n_pairs: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(X)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    return i[mask], j[mask]


def rp_distance_preservation(X: np.ndarray, n_components: int,
                             seeds=None, kind: str | None = None) -> dict:
    """Distance-distortion ratio ||Rp_i - Rp_j|| / ||x_i - x_j|| across seeds."""
    seeds = seeds or C.SEEDS
    kind = kind or C.RP_KIND
    i, j = _pairwise_dist_sample(X, C.DIST_PAIRS, seed=12345)
    d_orig = np.linalg.norm(X[i] - X[j], axis=1)
    d_orig = np.maximum(d_orig, 1e-12)

    per_seed = []
    for s in seeds:
        rp = _make_rp(n_components, s, kind)
        Xp = rp.fit_transform(X)
        d_proj = np.linalg.norm(Xp[i] - Xp[j], axis=1)
        ratio = d_proj / d_orig
        per_seed.append({
            "seed": s,
            "mean_ratio": float(ratio.mean()),
            "std_ratio": float(ratio.std()),
            "median_ratio": float(np.median(ratio)),
        })
    df = pd.DataFrame(per_seed)
    return {
        "n_components": n_components,
        "kind": kind,
        "per_seed": df,
        "mean_ratio_across_seeds": float(df["mean_ratio"].mean()),
        "spread_of_mean_ratio": float(df["mean_ratio"].std()),
    }


def rp_distortion_vs_dim(X: np.ndarray, dims=None, seeds=None,
                         kind: str | None = None) -> pd.DataFrame:
    """Across-seed distance distortion as a function of RP target dimension.

    Shows the core RP trade-off: more components -> distortion shrinks toward the
    PCA/original geometry, at the cost of less compression. One row per dim with
    the mean and spread of the per-pair distance-ratio std across seeds.
    """
    dims = dims or C.DR_COMPONENTS
    rows = []
    for d in dims:
        rp = rp_distance_preservation(X, d, seeds=seeds or C.RP_SEEDS, kind=kind)
        ps = rp["per_seed"]
        rows.append({
            "n_components": d,
            "mean_ratio": float(ps["mean_ratio"].mean()),
            "distortion_std_mean": float(ps["std_ratio"].mean()),
            "mean_ratio_spread": float(ps["mean_ratio"].std()),
        })
    return pd.DataFrame(rows)


def fit_rp(X: np.ndarray, n_components: int, seed: int = 0, kind: str | None = None):
    kind = kind or C.RP_KIND
    return _make_rp(n_components, seed, kind).fit(X)


def rp_pinv_reconstruction(X: np.ndarray, n_components: int, seeds=None,
                           kind: str | None = None) -> pd.DataFrame:
    """Moore-Penrose reconstruction error for RP, per seed.

    RP has no inverse, but the projection is linear, so the pseudo-inverse gives
    the least-squares reconstruction achievable from the projected coordinates:
    with ``Z = X R^T`` we take ``Xhat = Z (R^T)^+``. Reporting this makes the
    contrast with PCA explicit -- PCA's rank-d projection is MSE-optimal by
    construction, so any gap is the price of choosing the subspace at random
    rather than from the data. Errors are in the same standardized units as
    ``pca_diagnostics()['recon']``, so the two are directly comparable.
    """
    seeds = seeds or C.RP_SEEDS
    kind = kind or C.RP_KIND
    rows = []
    for s in seeds:
        rp = _make_rp(n_components, s, kind).fit(X)
        R = np.asarray(rp.components_, dtype=np.float64)   # (d, p)
        Z = X @ R.T
        Xhat = Z @ np.linalg.pinv(R.T)
        rows.append({"seed": s, "kind": kind, "n_components": n_components,
                     "recon_mse": float(np.mean((X - Xhat) ** 2))})
    return pd.DataFrame(rows)


# ── timing helper (runtime column for the cross-method comparison table) ──────
def time_fit_transform(factory, X: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    obj = factory()
    Z = obj.fit_transform(X)
    return Z, time.perf_counter() - t0
