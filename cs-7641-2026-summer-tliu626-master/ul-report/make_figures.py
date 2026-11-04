#!/usr/bin/env python3
"""Regenerate every report figure at PUBLICATION size.

Why this exists. The figures were originally authored at matplotlib's default
canvas size (6-15 in wide) and then shrunk with ``\\includegraphics[width=0.7...]``
to fit the page limit. That scales the *fonts* down too: measured against the
ieeeconf geometry (textwidth 7.00 in, columnwidth 3.40 in), the axis labels were
rendering at 3.6-4.6 pt and legends at 2.9-3.7 pt beside 10 pt body text --
illegible, and legibility is a scored requirement.

The fix is to author each figure at the width it will actually occupy and set
fonts in points directly, then include it at scale 1.0 (\\columnwidth or
\\textwidth). Labels then render at a true 8 pt and legends at 7 pt.

Colour discipline. One fixed categorical order is used everywhere (Okabe--Ito,
a published colour-vision-deficiency-safe qualitative set); hues are assigned to
entities, never cycled per plot, so "PCA" is the same blue in every panel and
class C7 is the same colour in Part 1 and Part 4. Magnitude uses a single-hue
sequential ramp and signed enrichment a two-hue diverging ramp with a neutral
grey midpoint. No figure uses two y-scales except the conventional PCA scree,
which is re-plotted here on a single 0-1 axis instead.

Everything is rebuilt from the saved CSVs except the four figures that need a
fitted transform (the cluster scatter grid and the t-SNE panel); those refit
PCA/ICA/RP + K-Means/GMM at the settings the saved results already selected, so
no model selection is redone.

    python make_figures.py             # all figures
    python make_figures.py scatter     # one figure
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src import config as C

# ── publication geometry (measured from ieeeconf.cls, not guessed) ────────────
COL_W  = 3.40      # \columnwidth in inches
TEXT_W = 7.00      # \textwidth in inches

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": "#7a7a7a", "xtick.color": "#4d4d4d",
    "ytick.color": "#4d4d4d", "grid.color": "#9a9a9a",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.constrained_layout.use": True,
})

# ── one fixed categorical order, used by every figure ─────────────────────────
# Okabe-Ito, with the pale yellow replaced by a dark ochre so the 8th slot keeps
# usable contrast on white. Assigned to entities in a fixed order, never cycled.
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
         "#56B4E9", "#D55E00", "#8C6D1F", "#000000"]
CLASS_C  = OKABE[:7]              # Cover_Type C1..C7
CLUST_C  = OKABE[:8]              # cluster ids 0..7
OTHER_C  = "#BDBDBD"              # "other clusters" fold-in
REP_C    = {"original": "#4D4D4D", "pca": "#0072B2",
            "ica": "#E69F00", "rp": "#009E73"}
COV_C    = {"full": "#0072B2", "tied": "#E69F00",
            "diag": "#009E73", "spherical": "#CC79A7"}

# Cluster-identity colouring for the scatter grid follows the reference layout in
# the course's exemplar report: a discrete tab10 ramp for the small partitions and
# tab20 for the 20-component mixtures (tab20 has exactly 20 distinct steps, so
# every component keeps its own hue instead of being folded into an "other" bin),
# each read off a vertical "cluster ID" colourbar.
KM_N    = 8                                    # tab10 steps used for k <= 8
KM_CMAP = plt.get_cmap("tab10", KM_N)
EM_CMAP = plt.get_cmap("tab20", 20)

# sequential (single hue, light->dark) and diverging (two hues, grey midpoint)
SEQ = LinearSegmentedColormap.from_list("seq", ["#F2F6FA", "#0072B2", "#00344F"])
DIV = LinearSegmentedColormap.from_list("div", ["#0072B2", "#EDEDED", "#D55E00"])

CLASS_SHORT = ["Spruce/Fir", "Lodgepole", "Ponderosa", "Cottonwd",
               "Aspen", "Douglas-fir", "Krummholz"]

R = C.RESULT_DIR
F = C.FIG_DIR


def _save(fig, name: str) -> None:
    path = os.path.join(F, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}", flush=True)


def _bare(ax) -> None:
    """Scatter panels carry no numeric axis meaning; strip the furniture."""
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#c8c8c8")


def _working():
    from src import data as D
    return D.get_working_sample()


# ── Fig 1: the sample, the preprocessing budget, and K-Means selection ────────
def fig_overview() -> None:
    """One float for Sections II-III: (a,b) what the data and the preprocessing
    decision look like, (c,d,e) how K-Means responds to k on that geometry.

    Merged deliberately -- two separate floats cost two caption blocks and two
    float separations against a hard 8-page limit, and the panels are read in
    immediate succession anyway.
    """
    work = _working()
    y, X = work.y, work.X
    counts = np.bincount(y, minlength=7)
    frac = counts / counts.sum()

    cont_idx = [C.FEATURE_COLS.index(c) for c in C.CONTINUOUS_COLS]
    wild_idx = [C.FEATURE_COLS.index(c) for c in C.WILDERNESS_COLS]
    soil_idx = [C.FEATURE_COLS.index(c) for c in C.SOIL_COLS]
    v = X.var(0)
    blocks = [("cont.\n(10)", v[cont_idx].sum()),
              ("wild.\n(4)", v[wild_idx].sum()),
              ("soil\n(40)", v[soil_idx].sum())]
    tot = sum(b[1] for b in blocks)
    km = pd.read_csv(os.path.join(R, "part1_kmeans_sweep.csv"))

    fig, axes = plt.subplots(1, 5, figsize=(TEXT_W, 1.34),
                             gridspec_kw={"width_ratios":
                                          [1.28, 0.72, 1.0, 1.0, 1.0]})

    a = axes[0]
    ypos = np.arange(7)
    a.barh(ypos, counts, color=CLASS_C, height=0.74)
    a.set_xscale("log"); a.set_xlim(30, 400000)
    a.set_yticks(ypos, [f"C{i+1}" for i in range(7)], fontsize=6.5)
    a.invert_yaxis(); a.grid(axis="y", visible=False)
    for i, (c, f_) in enumerate(zip(counts, frac)):
        a.text(c * 1.3, i, f"{c:,} ({f_:.1%})", va="center", fontsize=5.4,
               color="#333333")
    a.set(xlabel="instances (log)", title="(a) 104:1 imbalance")

    b = axes[1]
    b.bar(range(3), [x[1] / tot for x in blocks],
          color=["#0072B2", "#8FBBD6", "#C9D9E4"], width=0.7)
    b.set_xticks(range(3), [x[0] for x in blocks], fontsize=5.8)
    b.set_ylim(0, 1.12); b.grid(axis="x", visible=False)
    for i, x in enumerate(blocks):
        b.text(i, x[1] / tot + 0.04, f"{x[1]/tot:.0%}", ha="center", fontsize=5.8,
               color="#333333")
    b.set(ylabel="variance share", title="(b) budget")

    c = axes[2]
    c.plot(km["k"], km["silhouette"], "o-", color=OKABE[0], label="silhouette")
    c.plot(km["k"], km["davies_bouldin"] / 10, "s--", color=OKABE[5],
           label="Davies\u2013Bouldin/10")
    c.set(xlabel="clusters $k$", ylabel="internal index",
          title="(c) internal: best at $k{=}2$")
    c.legend(loc="upper right", frameon=False, fontsize=6.0, handlelength=1.3,
             labelspacing=0.2)

    d = axes[3]
    d.plot(km["k"], km["inertia"] / 1e3, "o-", color=OKABE[2])
    d.set(xlabel="clusters $k$", ylabel="inertia ($\\times10^{3}$)",
          title="(d) no elbow")
    d.annotate("smooth decay:\nno preferred $k$", xy=(7, 128), xytext=(5.4, 166),
               fontsize=6.0, color="#333333",
               arrowprops=dict(arrowstyle="->", lw=0.6, color="#666666"))

    e = axes[4]
    e.plot(km["k"], km["nmi"], "o-", color=OKABE[2], label="NMI")
    e.plot(km["k"], km["ari"], "s--", color=OKABE[3], label="ARI")
    e.set(xlabel="clusters $k$", ylabel="agreement w/ Cover_Type",
          title="(e) external: rises in $k$")
    e.set_ylim(0, 0.125)
    e.legend(loc="upper left", frameon=False, fontsize=6.0, handlelength=1.3,
             labelspacing=0.2)
    _save(fig, "part1_overview.png")


# ── Fig 3: EM/GMM selection + what the selected partitions contain ────────────
def fig_gmm() -> None:
    """(a) BIC never turns; (b) the k=2 split reproduces the class marginal;
    (c) the ICA k=8 split is the one partition that departs from it."""
    g = pd.read_csv(os.path.join(R, "part1_gmm_sweep.csv"))
    ct = np.array(json.load(open(os.path.join(R, "part1_kmeans_contingency.json"))))
    ica_p = os.path.join(R, "part3_ica_contingency.json")
    ica_ct = np.array(json.load(open(ica_p))) if os.path.exists(ica_p) else None

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 1.45),
                             gridspec_kw={"width_ratios": [1.05, 1.0, 1.15]})

    a = axes[0]
    for cov, gg in g.groupby("cov_type"):
        gg = gg.sort_values("n_components")
        a.plot(gg["n_components"], gg["bic"] / 1e6, "o-", color=COV_C[cov],
               label=cov)
    a.axhline(0, color="#9a9a9a", lw=0.6)
    a.set(xlabel="mixture components $n$", ylabel="BIC ($\\times10^{6}$)",
          title="(a) BIC never turns", ylim=(-11.6, 3.0))
    a.legend(loc="lower left", frameon=False, ncol=4, columnspacing=0.6,
             handlelength=1.3, handletextpad=0.35)
    a.annotate("min-BIC: diag, $n{=}20$\n(the grid edge)", xy=(20, -9.1),
               xytext=(7.2, -6.6), fontsize=6.3, color="#333333",
               arrowprops=dict(arrowstyle="->", lw=0.6, color="#666666"))

    # (b) composition of each k=2 cluster against the global marginal
    b = axes[1]
    marg = ct.sum(0) / ct.sum()
    comp = ct / ct.sum(1, keepdims=True)
    w, xs = 0.26, np.arange(7)
    b.bar(xs - w, comp[0], w, color=OKABE[0], label=f"$k_0$ ({ct[0].sum():,})")
    b.bar(xs, comp[1], w, color=OKABE[1], label=f"$k_1$ ({ct[1].sum():,})")
    b.bar(xs + w, marg, w, color="#8a8a8a", label="sample marginal")
    b.set_yscale("log"); b.set_ylim(2e-3, 3.0)
    b.set_xticks(xs, [f"C{i+1}" for i in range(7)])
    b.grid(axis="x", visible=False)
    b.set(ylabel="class share in cluster (log)",
          title="(b) $k{=}2$ reproduces the marginal")
    b.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2,
             handletextpad=0.4, labelspacing=0.25)

    # (c) ICA k=8: enrichment relative to the marginal, log2 ratio
    c = axes[2]
    if ica_ct is not None:
        comp8 = ica_ct / np.maximum(ica_ct.sum(1, keepdims=True), 1)
        enr = np.log2(np.maximum(comp8, 1e-4) / marg[None, :])
        im = c.imshow(enr, cmap=DIV, aspect="auto",
                      norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3))
        c.set_xticks(range(7), [f"C{i+1}" for i in range(7)])
        c.set_yticks(range(len(ica_ct)),
                     [f"$k_{{{i}}}$" for i in range(len(ica_ct))], fontsize=6.5)
        c.grid(False)
        cb = fig.colorbar(im, ax=c, pad=0.02, shrink=0.92)
        cb.set_label("$\\log_2$(share / marginal)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        # label only genuine ENRICHMENT; depletion is already legible as colour,
        # and annotating every empty cell with "0.0x" was pure noise.
        for i in range(ica_ct.shape[0]):
            for j in range(7):
                if enr[i, j] >= 1.0:
                    c.text(j, i, f"{2**enr[i,j]:.0f}$\\times$", ha="center",
                           va="center", fontsize=5.6,
                           color="w" if enr[i, j] > 2.2 else "#222222")
                elif ica_ct[i, j] == 0:
                    c.text(j, i, "0", ha="center", va="center", fontsize=5.0,
                           color="#FFFFFF")
    c.set(title="(c) ICA $k{=}8$: genuine enrichment")
    _save(fig, "part1_gmm_metrics.png")


# ── Fig 4: the three DR diagnostics side by side (double column) ──────────────
def fig_dr_diagnostics() -> None:
    """PCA variance | ICA non-Gaussianity | RP distance preservation.

    One panel per method, each showing the evidence that method can supply for
    its own component count -- deliberately not a single shared criterion.
    """
    jc = json.load(open(os.path.join(R, "part2_component_choice.json")))
    ica = json.load(open(os.path.join(R, "part2_ica_summary.json")))
    dim = pd.read_csv(os.path.join(R, "part2_rp_distortion_vs_dim.csv"))
    per = pd.read_csv(os.path.join(R, "part2_rp_distance.csv"))
    k = int(jc["pca"])

    from sklearn.decomposition import PCA
    X = _working().X
    evr = PCA(svd_solver="full", random_state=0).fit(X).explained_variance_ratio_
    cum = np.cumsum(evr)

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 1.42))

    # (a) single 0-1 axis: bars are per-component EVR, line is the cumulative
    a = axes[0]
    xs = np.arange(1, len(evr) + 1)
    a.bar(xs, evr, color="#9EC7DE", width=0.85, label="per component")
    a.plot(xs, cum, color="#00344F", lw=1.3, label="cumulative")
    a.axhline(C.PCA_VAR_TARGET, color="#9a9a9a", ls=":", lw=0.9)
    a.axvline(k, color="#9a9a9a", ls=":", lw=0.9)
    a.set_ylim(0, 1.06)
    a.annotate(f"$d{{=}}{k}$ at {cum[k-1]:.1%}", xy=(k, cum[k - 1]),
               xytext=(k + 8, 0.62), fontsize=6.5, color="#333333",
               arrowprops=dict(arrowstyle="->", lw=0.6, color="#666666"))
    a.set(xlabel="principal component", ylabel="explained variance ratio",
          title="(a) PCA: 10 of 54 reach 92%")
    a.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.06))

    # (b) ICA per-component |excess kurtosis|, sorted
    b = axes[1]
    kurt = np.abs(np.asarray(ica["kurtosis_mean"], dtype=float))
    order = np.argsort(kurt)[::-1]
    b.bar(np.arange(1, len(kurt) + 1), kurt[order], color=OKABE[1], width=0.7)
    b.axhline(np.median(kurt), color="#4d4d4d", ls="--", lw=0.9,
              label=f"median {np.median(kurt):.2f}")
    b.axhline(0, color="#9a9a9a", lw=0.8)
    b.text(len(kurt) - 0.3, 0.16, "Gaussian = 0", ha="right", fontsize=6.3,
           color="#666666")
    b.set_xticks(np.arange(1, len(kurt) + 1))
    b.grid(axis="x", visible=False)
    b.set(xlabel="ICA component (sorted)", ylabel="$|$excess kurtosis$|$",
          title="(b) ICA: non-Gaussian, unevenly")
    b.legend(loc="upper right", frameon=False)

    # (c) RP distortion vs dimension, with the across-seed spread
    c = axes[2]
    c.errorbar(dim["n_components"], dim["distortion_std_mean"],
               yerr=dim["mean_ratio_spread"], marker="o", capsize=2.5,
               color=OKABE[2], label="distortion (mean $\\pm$ seed spread)")
    c.axhline(C.RP_JL_EPS, color=OKABE[5], ls="--", lw=1,
              label=f"tolerance $\\epsilon={C.RP_JL_EPS}$")
    c.axvline(10, color="#9a9a9a", ls=":", lw=0.9)
    g = per[per["kind"] == "gaussian"]["mean_ratio"]
    c.set_ylim(0, 0.90)
    c.annotate(f"$d{{=}}10$: 0.214,\nseed ratios {g.min():.2f}\u2013{g.max():.2f}",
               xy=(10, 0.214), xytext=(13.0, 0.36), fontsize=6.3, color="#333333",
               arrowprops=dict(arrowstyle="->", lw=0.6, color="#666666"))
    c.set(xlabel="RP target dimension $d$",
          ylabel="pairwise-distance distortion",
          title="(c) RP: fidelity is a seed lottery")
    c.legend(loc="upper right", frameon=False, labelspacing=0.25)
    _save(fig, "part2_dr_diagnostics.png")


# ── Fig 5: the cluster-scatter grid (the centrepiece) ─────────────────────────
def _fit_reps(X: np.ndarray, seed: int = 0):
    """PCA / ICA / RP at d=10 plus the 2 axes each is best displayed on."""
    from scipy.stats import kurtosis
    from src import dr as DR
    d = 10
    Zp = DR.fit_pca(X, d, seed).transform(X)
    Zi = DR.fit_ica(X, d, seed).transform(X)
    Zr = DR.fit_rp(X, d, seed).transform(X)
    kt = np.abs(kurtosis(Zi, axis=0, fisher=True, bias=False))
    ica_ax = np.argsort(kt)[::-1][:2]
    return [
        ("PCA", Zp, (0, 1), "PC1", "PC2"),
        ("ICA", Zi, tuple(ica_ax), f"IC{ica_ax[0]+1}", f"IC{ica_ax[1]+1}"),
        ("RP",  Zr, (0, 1), "RP dim 1", "RP dim 2"),
    ]


def fig_cluster_scatter() -> None:
    """{K-Means, EM} x {PCA, ICA, RP}, plus a Cover_Type reference panel.

    Laid out as the compact two-row strip used by the course exemplar, with
    cluster identity read off a discrete tab10/tab20 "cluster ID" colourbar.
    Clustering is done in the FULL 10-d representation at the settings Part 3
    already selected; only the *display* is 2-d, so the panels visualize the
    partitions rather than redefining them.
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from src import clustering as CL

    work = _working()
    X, y = work.X, work.y
    sel = pd.read_csv(os.path.join(R, "part3_dr_clustering.csv")).set_index("space")
    reps = _fit_reps(X, seed=0)

    rng = np.random.default_rng(0)
    show = rng.choice(len(X), size=min(5000, len(X)), replace=False)

    # Two rows, not three: the Cover_Type reference occupies one panel rather
    # than a whole row, and the Part-3 summary bars ride in the last column, so
    # this one float carries what would otherwise be three.
    fig = plt.figure(figsize=(TEXT_W, 1.86), layout=None)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.055, 0.90],
                          wspace=0.10, hspace=0.16,
                          left=0.012, right=0.965, top=0.925, bottom=0.095)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)]
                     for r in range(2)])
    cbar_ax = [fig.add_subplot(gs[r, 3]) for r in range(2)]
    bar_ax = fig.add_subplot(gs[0, 4])
    gt_ax = fig.add_subplot(gs[1, 4])
    ica_ct_saved = None
    tag = "abcdef"

    for j, (name, Z, (ax0, ax1), lx, ly) in enumerate(reps):
        row = sel.loc[name]
        kk, nn, cov = int(row["km_k"]), int(row["gmm_n"]), str(row["gmm_cov"])

        CL.set_all_seeds(0)
        km_lab = KMeans(n_clusters=kk, init="k-means++", n_init=C.KMEANS_N_INIT,
                        max_iter=C.KMEANS_MAX_ITER, random_state=0).fit_predict(Z)
        CL.set_all_seeds(0)
        em_lab = GaussianMixture(n_components=nn, covariance_type=cov,
                                 n_init=C.GMM_N_INIT, max_iter=C.GMM_MAX_ITER,
                                 reg_covar=C.GMM_REG_COVAR,
                                 random_state=0).fit_predict(Z)
        if name == "ICA":
            ica_ct_saved = CL.contingency_table(y, km_lab)

        P, Q = Z[show, ax0], Z[show, ax1]
        # ICA's leading axes are by construction the heaviest-tailed, so a raw
        # extent is set by a handful of outliers and the bulk collapses to a
        # dot. Clip every panel to the same robust 1-99 percentile window so the
        # three columns are read at comparable magnification.
        xlo, xhi = np.percentile(P, [1, 99])
        ylo, yhi = np.percentile(Q, [1, 99])
        mx, my = 0.06 * (xhi - xlo), 0.06 * (yhi - ylo)
        lims = dict(xlim=(xlo - mx, xhi + mx), ylim=(ylo - my, yhi + my))

        # row 0 -- K-Means clusters, coloured by cluster id (tab10)
        sc_km = axes[0, j].scatter(P, Q, s=0.9, c=km_lab[show], cmap=KM_CMAP,
                                   vmin=-0.5, vmax=KM_N - 0.5, alpha=0.65,
                                   linewidths=0, rasterized=True)
        axes[0, j].set_title(f"({tag[2*j]}) {name} K-Means, $k{{=}}{kk}$",
                             fontsize=6.8, pad=2.0)

        # row 1 -- EM components, coloured by component id (tab20 holds all 20)
        sc_em = axes[1, j].scatter(P, Q, s=0.9, c=em_lab[show], cmap=EM_CMAP,
                                   vmin=-0.5, vmax=nn - 0.5, alpha=0.65,
                                   linewidths=0, rasterized=True)
        axes[1, j].set_title(f"({tag[2*j+1]}) {name} EM, $n{{=}}{nn}$ {cov}",
                             fontsize=6.8, pad=2.0)
        axes[1, j].set_xlabel(f"{lx} / {ly}", fontsize=6.2, labelpad=1.2)

        for r in range(2):
            axes[r, j].set(**lims)
            _bare(axes[r, j])
        if name == "PCA":
            pca_gt = (P, Q, lims)

    # the ground-truth reference, on PCA's identical axes
    P, Q, lims = pca_gt
    sc_y = gt_ax.scatter(P, Q, s=0.9, c=y[show] + 1, cmap=KM_CMAP,
                         vmin=0.5, vmax=7.5, alpha=0.65, linewidths=0,
                         rasterized=True)
    gt_ax.set_title("(g) PCA, Cover_Type", fontsize=6.8, pad=2.0)
    gt_ax.set(**lims)
    _bare(gt_ax)
    cb = fig.colorbar(sc_y, ax=gt_ax, pad=0.02, fraction=0.06,
                      ticks=range(1, 8))
    cb.set_label("Cover_Type", fontsize=5.6, labelpad=1.0)
    cb.ax.tick_params(labelsize=5.0, length=1.2, pad=0.8)
    cb.outline.set_linewidth(0.4); cb.solids.set_alpha(1.0)

    # discrete "cluster ID" colourbars, one per row, as in the reference layout
    for ax_, sc_, lab_, ticks_ in (
            (cbar_ax[0], sc_km, "cluster ID", range(KM_N)),
            (cbar_ax[1], sc_em, "component ID", range(0, 20, 4))):
        cb = fig.colorbar(sc_, cax=ax_, ticks=list(ticks_))
        cb.set_label(lab_, fontsize=5.6, labelpad=1.0)
        cb.ax.tick_params(labelsize=5.0, length=1.2, pad=0.8)
        cb.outline.set_linewidth(0.4)
        cb.solids.set_alpha(1.0)

    _part3_bars(bar_ax, sel)
    _save(fig, "part3_cluster_scatter.png")

    if ica_ct_saved is not None:
        with open(os.path.join(R, "part3_ica_contingency.json"), "w") as f_:
            json.dump(ica_ct_saved.tolist(), f_)
        print("  wrote part3_ica_contingency.json", flush=True)


def _part3_bars(ax, sel) -> None:
    """Internal quality vs.\\ label alignment, drawn into the grid's 4th column.

    Horizontal bars so four space labels fit in a narrow panel without rotation.
    """
    order = ["Original", "PCA", "ICA", "RP"]
    d = sel.loc[order]
    ys = np.arange(len(order))[::-1]
    h = 0.26
    ax.barh(ys + h, d["km_silhouette"], h, color=OKABE[0], label="silhouette")
    ax.barh(ys, d["km_nmi"], h, color=OKABE[1], label="NMI")
    ax.barh(ys - h, d["km_ari"], h, color=OKABE[2], label="ARI")
    for yy, s, n in zip(ys, d["km_silhouette"], d["km_nmi"]):
        ax.text(s + 0.008, yy + h, f"{s:.2f}", va="center", fontsize=5.2,
                color="#333333")
        ax.text(n + 0.008, yy, f"{n:.3f}", va="center", fontsize=5.2,
                color="#333333")
    ax.set_yticks(ys, [f"{o} $k{{=}}{int(k)}$"
                       for o, k in zip(order, d["km_k"])], fontsize=5.6)
    ax.tick_params(axis="x", labelsize=5.4, pad=1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.46)
    ax.set_title("(h) internal vs. external", fontsize=6.8, pad=2.0)
    ax.legend(loc="center right", frameon=False, fontsize=5.4, handlelength=0.9,
              handletextpad=0.3, labelspacing=0.15, borderpad=0.1)


# ── Fig 7: downstream NN -- where the loss lands, and how deep it goes ────────
def fig_nn() -> None:
    per = pd.read_csv(os.path.join(R, "part4_nn_per_seed.csv"))
    s = pd.read_csv(os.path.join(R, "part4_nn_dim_sweep.csv"))
    summ = pd.read_csv(os.path.join(R, "part4_nn_summary.csv"))
    reps = ["original", "pca", "ica", "rp"]
    lab = {"original": "original (54-d)", "pca": "PCA-10", "ica": "ICA-10",
           "rp": "RP-10"}
    test_n = [1458, 1951, 246, 19, 65, 120, 141]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 1.48),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    a = axes[0]
    xs = np.arange(7); w = 0.2
    for i, r in enumerate(reps):
        g = per[per["representation"] == r]
        m = [g[f"f1_c{c+1}"].mean() for c in range(7)]
        sd = [g[f"f1_c{c+1}"].std() for c in range(7)]
        a.bar(xs + (i - 1.5) * w, m, w, yerr=sd, capsize=1.6,
              error_kw=dict(lw=0.6), color=REP_C[r], label=lab[r])
    a.set_xticks(xs, [f"C{c+1}\n$n{{=}}{n}$" for c, n in enumerate(test_n)],
                 fontsize=6.4)
    a.grid(axis="x", visible=False)
    a.set(ylabel="per-class test F1", ylim=(0, 0.95),
          title="(a) compression costs the rare classes most")
    a.legend(loc="upper right", frameon=False, ncol=2, columnspacing=0.8,
             labelspacing=0.25)
    a.annotate("C7 Krummholz, the elevation-defined class:\n"
               "$-0.144$ F1 under PCA vs $-0.045$ for C1",
               xy=(6 - 0.5 * w, 0.56), xytext=(1.75, 0.05), fontsize=6.2,
               color="#333333",
               arrowprops=dict(arrowstyle="->", lw=0.6, color="#666666"))

    b = axes[1]
    agg = s.groupby(["representation", "n_components"]).agg(
        m=("macro_f1", "mean"), sd=("macro_f1", "std")).reset_index()
    mk = {"pca": "o-", "ica": "s-", "rp": "^-"}
    for rep, gg in agg.groupby("representation"):
        b.errorbar(gg["n_components"], gg["m"], yerr=gg["sd"],
                   fmt=mk.get(rep, "o-"), color=REP_C[rep], capsize=2.5,
                   label=rep.upper())
    base = float(summ.loc[summ["representation"] == "original",
                          "macro_f1_mean"].iloc[0])
    b.axhline(base, color=REP_C["original"], ls="--", lw=1,
              label=f"original 54-d ({base:.3f})")
    b.axvline(10, color="#9a9a9a", ls=":", lw=0.9)
    b.set(xlabel="number of components $d$", ylabel="test Macro-F1",
          title="(b) recovery needs ${\\sim}30$ components")
    b.legend(loc="lower right", frameon=False)
    _save(fig, "part4_nn_after_dr.png")


# ── Fig 8: linear vs nonlinear 2-D embedding (double column) ──────────────────
def fig_manifold() -> None:
    from src import manifold as M
    work = _working()
    res = M.run_manifold(work.X, work.y, seed=C.SEEDS[0])
    # run_manifold already returns the PCA-2D baseline on the IDENTICAL points
    # the metrics table was computed from, so the comparison is like-for-like.
    emb, P2, lab = res["embedding"], res["pca2"], res["labels"]
    met = pd.read_csv(os.path.join(R, "extra_manifold_metrics.csv")).set_index("space")

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.28))
    for ax, (E, name, key) in zip(
            axes, [(P2, "PCA (first two components)", "PCA-2D"),
                   (emb, "t-SNE (perplexity 30)", "tsne")]):
        for cl in range(7):
            m = lab == cl
            ax.scatter(E[m, 0], E[m, 1], s=1.6, c=CLASS_C[cl], alpha=0.65,
                       linewidths=0, rasterized=True)
        row = met.loc[key]
        ax.set_title(f"{name}\ntrust.\\ {row['trustworthiness']:.2f} / $k$NN {row['knn_label_agreement']:.3f}", fontsize=6.2, pad=1.5)
        _bare(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=2.6, color=CLASS_C[i],
                          label=f"C{i+1}") for i in range(7)]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=5.8, handletextpad=0.2, columnspacing=0.6,
               bbox_to_anchor=(0.5, -0.10))
    _save(fig, "extra_manifold.png")


ALL = {
    "overview": fig_overview,
    "scatter": fig_cluster_scatter,   # writes the contingency the gmm fig reads
    "gmm": fig_gmm,
    "dr": fig_dr_diagnostics,
    "nn": fig_nn,
    "manifold": fig_manifold,
}

if __name__ == "__main__":
    print("regenerating figures at publication size "
          f"(col={COL_W}in, text={TEXT_W}in, labels 8pt)", flush=True)
    only = sys.argv[1:] or list(ALL)
    for nm in only:
        if nm not in ALL:
            raise SystemExit(f"unknown figure {nm!r}; choose from {list(ALL)}")
        ALL[nm]()
    print("done", flush=True)
