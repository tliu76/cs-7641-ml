#!/usr/bin/env python3
"""
Top-level orchestrator for the UL Report experiments.

Run from the ul-report/ directory:
    python run_ul.py all         # part1 -> part2 -> part3 -> part4 -> manifold
    python run_ul.py part1       # individual stages also work
    python run_ul.py part2 part4

Stages
  part1     Clustering on original features (K-Means, GMM) + label alignment
  part2     Dimensionality reduction diagnostics (PCA, ICA, RP)
  part3     Clustering after DR (six combinations) comparison table
  part4     NN after DR (leakage-safe): original vs PCA/ICA/RP, seeds + runtime
  manifold  Extra credit: nonlinear manifold embedding vs PCA (visual)
  all       Everything above, then writes repro.json

Outputs land under outputs/{figures,results,logs}/. A fast smoke test that
shrinks every sweep/seed/epoch:  UL_FAST=1 python run_ul.py all
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from src import config as C  # noqa: E402

# Tiny MLP + sklearn: cap threads to avoid dispatch overhead on shared nodes.
try:
    import torch
    torch.set_num_threads(int(os.environ.get("UL_TORCH_THREADS", "4")))
except Exception:
    pass

from src import clustering as CL          # noqa: E402
from src import data as D                 # noqa: E402
from src import dr as DR                  # noqa: E402
from src import dr_clustering as DRC       # noqa: E402
from src import nn_dr as NN               # noqa: E402
from src import plotting as P             # noqa: E402
from src import repro as REPRO            # noqa: E402
from src.seeds import set_all_seeds        # noqa: E402

import matplotlib.pyplot as plt           # noqa: E402


def _banner(msg: str) -> None:
    print("\n" + "=" * 64 + f"\n  {msg}\n" + "=" * 64, flush=True)


def _resolve_dr_components(work) -> dict:
    """Per-method justified component counts (PCA var / ICA rank / RP distortion)."""
    jc = DR.justified_components(work.X, seed=C.SEEDS[0])
    if C.ICA_N_COMPONENTS:
        jc["ica"] = int(C.ICA_N_COMPONENTS)
    if C.RP_N_COMPONENTS:
        jc["rp"] = int(C.RP_N_COMPONENTS)
    return jc


# ── Part 1 ────────────────────────────────────────────────────────────────────
def part1(work) -> CL.Selection:
    _banner("PART 1  Clustering on original features")
    seed = C.SEEDS[0]
    km = CL.run_kmeans_sweep(work.X, work.y, seed=seed)
    gm = CL.run_gmm_sweep(work.X, work.y, seed=seed)
    P.save_table(km, "part1_kmeans_sweep.csv")
    P.save_table(gm, "part1_gmm_sweep.csv")

    sel = CL.select_settings(km, gm)
    print(f"[part1] selected  K-Means k={sel.kmeans_k} | "
          f"GMM n={sel.gmm_n} cov={sel.gmm_cov}")

    # Figure: internal + external metrics vs k for K-Means.
    P.plot_metric_vs_k(
        km, "k",
        ["silhouette", "davies_bouldin", "calinski_harabasz", "ari", "nmi"],
        "part1_kmeans_metrics.png", "K-Means: internal vs external metrics vs k")

    # Figure: GMM BIC/AIC vs components per covariance type.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for cov, g in gm.groupby("cov_type"):
        axes[0].plot(g["n_components"], g["bic"], marker="o", label=cov)
        axes[1].plot(g["n_components"], g["ari"], marker="o", label=cov)
    axes[0].set(xlabel="n_components", ylabel="BIC", title="GMM BIC (lower=better)")
    axes[1].set(xlabel="n_components", ylabel="ARI", title="GMM label alignment")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    P.save_fig(fig, "part1_gmm_metrics.png")

    # Contingency table for the selected K-Means (rows=clusters, cols=Cover_Type).
    labels = CL.kmeans_labels(work.X, sel.kmeans_k, seed=seed)
    ct = CL.contingency_table(work.y, labels)
    P.save_json(ct.tolist(), "part1_kmeans_contingency.json")
    P.plot_contingency(ct, "part1_kmeans_contingency.png",
                       title=f"K-Means (k={sel.kmeans_k}) vs Cover_Type")
    return sel


# ── Part 2 ────────────────────────────────────────────────────────────────────
def part2(work) -> dict:
    _banner("PART 2  Dimensionality reduction (PCA / ICA / RP)")
    seed = C.SEEDS[0]
    jc = _resolve_dr_components(work)
    print(f"[part2] per-method justified dims: PCA={jc['pca']} (>={C.PCA_VAR_TARGET:.0%} var), "
          f"ICA={jc['ica']} (retained rank), RP={jc['rp']} "
          f"(distortion<={C.RP_JL_EPS}; JL min-dim@eps={C.RP_JL_EPS} would be {jc['jl_min_dim']})")
    P.save_json(jc, "part2_component_choice.json")

    # PCA.
    pca = DR.pca_diagnostics(work.X, seed=seed)
    P.plot_scree(pca["evr"], "part2_pca_scree.png", target=C.PCA_VAR_TARGET)
    P.save_table(pca["recon"], "part2_pca_reconstruction.csv")
    loadings = DR.pca_top_loadings(pca["components"], work.feature_names)
    P.save_table(loadings, "part2_pca_loadings.csv")
    print(f"[part2] PCA: k@{C.PCA_VAR_TARGET:.0%}={pca['k_target']}  "
          f"effective_rank={pca['effective_rank']:.1f}")

    # ICA (at its own justified dim).
    ica = DR.ica_diagnostics(work.X, jc["ica"], seeds=C.SEEDS)
    P.plot_kurtosis(ica["kurtosis_mean"], "part2_ica_kurtosis.png")
    P.save_json({k: v for k, v in ica.items()
                 if k not in ("kurtosis_per_seed",)}, "part2_ica_summary.json")
    print(f"[part2] ICA: converged={ica['converged']}  "
          f"stability(|corr|)={ica['stability_abscorr']:.3f}")

    # RP: distance preservation across MANY seeds (RP_SEEDS), Gaussian + sparse.
    rp_rows = []
    for kind in ("gaussian", "sparse"):
        rp = DR.rp_distance_preservation(work.X, jc["rp"], seeds=C.RP_SEEDS, kind=kind)
        d = rp["per_seed"].copy(); d["kind"] = kind
        rp_rows.append(d)
        print(f"[part2] RP({kind}, d={jc['rp']}, {len(C.RP_SEEDS)} seeds): mean ratio "
              f"{rp['mean_ratio_across_seeds']:.3f} "
              f"+/- {rp['spread_of_mean_ratio']:.3f}")
    rp_df = pd.concat(rp_rows, ignore_index=True)
    P.save_table(rp_df, "part2_rp_distance.csv")

    # RP distortion vs target dim (the "more components -> less distortion" curve).
    rp_dim = DR.rp_distortion_vs_dim(work.X, dims=C.DR_COMPONENTS,
                                     seeds=C.RP_SEEDS, kind=C.RP_KIND)
    P.save_table(rp_dim, "part2_rp_distortion_vs_dim.csv")

    # Figure: RP seed spread (boxplot) + distortion-vs-dim, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    box_data = [rp_df[rp_df["kind"] == k]["mean_ratio"].values
                for k in ("gaussian", "sparse")]
    axes[0].boxplot(box_data, tick_labels=["gaussian", "sparse"])
    axes[0].axhline(1.0, color="gray", ls="--", lw=1)
    axes[0].set(ylabel="mean ||proj||/||orig||",
                title=f"RP seed variability (d={jc['rp']}, {len(C.RP_SEEDS)} seeds)")
    axes[1].errorbar(rp_dim["n_components"], rp_dim["distortion_std_mean"],
                     yerr=rp_dim["mean_ratio_spread"], marker="o", capsize=3)
    axes[1].axhline(C.RP_JL_EPS, color="crimson", ls="--", lw=1,
                    label=f"eps={C.RP_JL_EPS}")
    axes[1].set(xlabel="n_components", ylabel="across-seed distortion std",
                title="RP distortion shrinks with dim")
    axes[1].legend(fontsize=8)
    P.save_fig(fig, "part2_rp_distance.png")

    # Cross-method comparison table WITH runtime for all three methods.
    from sklearn.decomposition import PCA as _PCA, FastICA as _ICA
    from sklearn.random_projection import GaussianRandomProjection as _GRP
    rows = []
    _, t_pca = DR.time_fit_transform(
        lambda: _PCA(n_components=jc["pca"], random_state=seed), work.X)
    _, t_ica = DR.time_fit_transform(
        lambda: _ICA(n_components=jc["ica"], fun=C.ICA_FUN, max_iter=C.ICA_MAX_ITER,
                     tol=C.ICA_TOL, whiten="unit-variance", random_state=seed), work.X)
    _, t_rp = DR.time_fit_transform(
        lambda: _GRP(n_components=jc["rp"], random_state=seed), work.X)
    rows.append({"method": "PCA", "n_components": jc["pca"],
                 "main_diagnostic": f"cum_evr={pca['cum_evr'][jc['pca']-1]:.2f}",
                 "stability": "deterministic", "fit_time_s": round(t_pca, 4)})
    rows.append({"method": "ICA", "n_components": jc["ica"],
                 "main_diagnostic": f"median|kurt|={np.median(ica['kurtosis_mean']):.2f}",
                 "stability": f"|corr|={ica['stability_abscorr']:.2f}",
                 "fit_time_s": round(t_ica, 4)})
    rows.append({"method": "RP(gauss)", "n_components": jc["rp"],
                 "main_diagnostic": f"distortion_std={rp_dim.iloc[C.DR_COMPONENTS.index(jc['rp'])]['distortion_std_mean']:.2f}"
                 if jc['rp'] in C.DR_COMPONENTS else "dist-preserving",
                 "stability": "seed-varying", "fit_time_s": round(t_rp, 4)})
    P.save_table(pd.DataFrame(rows), "part2_cross_method.csv")
    print(f"[part2] DR fit time: PCA={t_pca:.3f}s  ICA={t_ica:.3f}s  RP={t_rp:.4f}s")
    return jc


# ── Part 3 ────────────────────────────────────────────────────────────────────
def part3(work, jc: dict) -> None:
    _banner("PART 3  Clustering after dimensionality reduction")
    seed = C.SEEDS[0]
    df = DRC.run_dr_clustering(work.X, work.y, jc, seed=seed)
    P.save_table(df, "part3_dr_clustering.csv")
    print(df.to_string(index=False))

    # Figure: KM label alignment (ARI/NMI) across representations.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(df))
    axes[0].bar(x - 0.2, df["km_ari"], 0.4, label="K-Means")
    axes[0].bar(x + 0.2, df["gmm_ari"], 0.4, label="GMM")
    axes[0].set_xticks(x, df["space"]); axes[0].set(ylabel="ARI",
                                                    title="Label alignment (ARI)")
    axes[0].legend(fontsize=8)
    axes[1].bar(x - 0.2, df["km_silhouette"], 0.4, label="K-Means silhouette")
    axes[1].bar(x + 0.2, df["gmm_bic"] / df["gmm_bic"].abs().max(), 0.4,
                label="GMM BIC (norm.)")
    axes[1].set_xticks(x, df["space"]); axes[1].set(title="Internal quality")
    axes[1].legend(fontsize=8)
    P.save_fig(fig, "part3_dr_clustering.png")


# ── Part 4 ────────────────────────────────────────────────────────────────────
def part4(jc: dict) -> None:
    _banner("PART 4  Neural network after DR (leakage-safe)")
    df = NN.run_nn_after_dr(jc, seeds=C.SEEDS)
    P.save_table(df, "part4_nn_per_seed.csv")
    summ = NN.summarize_nn(df)

    # Majority-class floor: makes the class-weighting effect (balanced acc > acc)
    # explicit so accuracy is not read against ~0 chance.
    base = NN.majority_class_baseline()
    print(f"[part4] majority-class floor: acc={base['accuracy']:.3f} "
          f"macroF1={base['macro_f1']:.3f} balAcc={base['balanced_accuracy']:.3f}")
    P.save_json(base, "part4_majority_baseline.json")
    P.save_table(summ, "part4_nn_summary.csv")
    print(summ.to_string(index=False))

    reps = summ["representation"].tolist()
    x = np.arange(len(reps))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].bar(x, summ["accuracy_mean"], yerr=summ["accuracy_std"], capsize=3)
    axes[0].axhline(base["accuracy"], color="crimson", ls="--", lw=1,
                    label=f"majority={base['accuracy']:.2f}")
    axes[0].set_xticks(x, reps); axes[0].set(ylabel="accuracy", title="Test accuracy")
    axes[0].legend(fontsize=7)
    axes[1].bar(x, summ["macro_f1_mean"], yerr=summ["macro_f1_std"], capsize=3,
                color="tab:orange")
    axes[1].set_xticks(x, reps); axes[1].set(title="Macro-F1 (class-weighted CE)")
    axes[2].bar(x, summ["fit_time_mean"], color="tab:green")
    axes[2].set_xticks(x, reps); axes[2].set(ylabel="seconds", title="NN train time")
    fig.suptitle("NN after DR at per-method dims "
                 f"(PCA={jc['pca']}, ICA={jc['ica']}, RP={jc['rp']}; "
                 f"mean +/- std, RP over {len(C.RP_SEEDS)} seeds)", y=1.04)
    P.save_fig(fig, "part4_nn_after_dr.png")

    # Performance-vs-n_components sweep: "how many components are enough?".
    _banner("PART 4b  NN performance vs n_components (PCA/ICA/RP)")
    sweep = NN.run_nn_dim_sweep(dims=C.NN_DR_SWEEP_DIMS, seeds=C.SEEDS)
    P.save_table(sweep, "part4_nn_dim_sweep.csv")
    agg = sweep.groupby(["representation", "n_components"]).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        acc_mean=("accuracy", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for rep, g in agg.groupby("representation"):
        ax.errorbar(g["n_components"], g["macro_f1_mean"], yerr=g["macro_f1_std"],
                    marker="o", capsize=3, label=rep)
    # original baseline as a reference band.
    orig = summ[summ["representation"] == "original"]
    if len(orig):
        ax.axhline(float(orig["macro_f1_mean"].iloc[0]), color="gray", ls="--",
                   lw=1, label="original (54-d)")
    ax.set(xlabel="n_components", ylabel="test Macro-F1",
           title="NN Macro-F1 vs DR dimension")
    ax.legend(fontsize=8)
    P.save_fig(fig, "part4_nn_dim_sweep.png")


# ── Extra credit ──────────────────────────────────────────────────────────────
def manifold(work) -> None:
    _banner(f"EXTRA CREDIT  Nonlinear manifold ({C.MANIFOLD_METHOD}) vs PCA")
    from src import manifold as M
    res = M.run_manifold(work.X, work.y, seed=C.SEEDS[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, emb, title in ((axes[0], res["pca2"], "PCA (top-2)"),
                           (axes[1], res["embedding"],
                            f"{res['method']} (nonlinear)")):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=res["labels"], s=5,
                        cmap="tab10", alpha=0.6)
        ax.set(title=title, xlabel="dim 1", ylabel="dim 2")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="Cover_Type (0..6)",
                 shrink=0.8)
    fig.suptitle(f"Linear vs nonlinear 2D embedding "
                 f"(n={res['n']}, {res['method']} {res['runtime_s']:.1f}s)", y=1.02)
    P.save_fig(fig, "extra_manifold.png")
    print(f"[manifold] {res['method']} embedded {res['n']} pts in "
          f"{res['runtime_s']:.1f}s")


# ── Driver ────────────────────────────────────────────────────────────────────
STAGES = ("part1", "part2", "part3", "part4", "manifold")


def main(argv: list[str]) -> None:
    stages = argv or ["all"]
    if "all" in stages:
        stages = list(STAGES)

    t0 = time.time()
    set_all_seeds(C.SEEDS[0])
    work = D.get_working_sample()
    print(f"[data] working sample: X={work.X.shape}  "
          f"classes={np.bincount(work.y)}  scale_all={C.SCALE_ALL}")

    jc = None
    sel = None
    if "part1" in stages:
        sel = part1(work)
    if "part2" in stages or "part3" in stages or "part4" in stages:
        jc = part2(work) if "part2" in stages else _resolve_dr_components(work)
    if "part3" in stages:
        part3(work, jc)
    if "part4" in stages:
        part4(jc)
    if "manifold" in stages:
        manifold(work)

    REPRO.write(sample_n=len(work.y))
    print(f"\n[done] stages={stages}  elapsed={time.time()-t0:.1f}s")
    print(f"  figures -> {C.FIG_DIR}")
    print(f"  results -> {C.RESULT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
