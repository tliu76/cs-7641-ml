"""
Hypothesis workflow for the Forest Cover Type (Covertype) dataset.

Derives algorithm-performance hypotheses from EDA statistics and produces
a multi-panel supporting figure saved to outputs/eda/hypothesis_summary.png.

Run after eda.py 
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# ── Shared constants (keep in sync with eda.py) ──────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "covtype.data")
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "sample.csv")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda")
RANDOM_SEED = 42

CONTINUOUS_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLS       = [f"Soil_Type{i}"       for i in range(1, 41)]
FEATURE_COLS    = CONTINUOUS_COLS + WILDERNESS_COLS + SOIL_COLS
TARGET_COL      = "Cover_Type"

CLASS_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}
CLASS_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3",
]


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load sample.csv if available; otherwise load the full dataset."""
    if os.path.exists(SAMPLE_PATH):
        df = pd.read_csv(SAMPLE_PATH)
        print(f"Loaded sample from {SAMPLE_PATH}  shape={df.shape}")
    else:
        print(f"Sample not found — loading full dataset from {DATA_PATH}")
        df = pd.read_csv(
            DATA_PATH, header=None,
            names=FEATURE_COLS + [TARGET_COL],
        )
        print(f"Loaded full dataset  shape={df.shape}")
    return df


# ── Analysis functions ────────────────────────────────────────────────────────
def class_imbalance_stats(df: pd.DataFrame) -> dict:
    vc  = df[TARGET_COL].value_counts().sort_index()
    pct = vc / len(df) * 100
    ir  = vc.max() / vc.min()
    return {"counts": vc, "pct": pct, "imbalance_ratio": ir}


def fischer_ratios(df: pd.DataFrame) -> pd.Series:
    """Fisher discriminant ratio (between-class / within-class variance) per feature."""
    overall = df[CONTINUOUS_COLS].mean()
    N       = len(df)
    ratios  = {}
    for col in CONTINUOUS_COLS:
        grps  = df.groupby(TARGET_COL)[col]
        cmean = grps.mean()
        cstd  = grps.std()
        cnts  = df[TARGET_COL].value_counts().sort_index()
        bv    = sum(cnts[c] * (cmean[c] - overall[col]) ** 2 for c in range(1, 8)) / N
        wv    = sum(cnts[c] * cstd[c] ** 2                  for c in range(1, 8)) / N
        ratios[col] = bv / wv if wv > 0 else 0.0
    return pd.Series(ratios).sort_values(ascending=False)


def elevation_ranges_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """5th / 50th / 95th percentile of Elevation per class."""
    rows = {}
    for c in range(1, 8):
        sub = df[df[TARGET_COL] == c]["Elevation"]
        rows[c] = {
            "p05":    sub.quantile(0.05),
            "median": sub.median(),
            "p95":    sub.quantile(0.95),
            "mean":   sub.mean(),
        }
    return pd.DataFrame(rows).T


def pca_overlap_score(df: pd.DataFrame) -> float:
    """
    Fraction of the 2-D PCA variance explained by the first two components —
    a proxy for how much of the class structure is linearly capturable.
    Higher => more linearly separable.
    """
    X = df[FEATURE_COLS].values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca.fit(X)
    return float(pca.explained_variance_ratio_.sum())


def binary_sparsity(df: pd.DataFrame) -> dict:
    soil_active   = df[SOIL_COLS].mean().mean()
    wilder_active = df[WILDERNESS_COLS].mean().mean()
    return {"soil": soil_active, "wilderness": wilder_active}


# ── Hypothesis derivation ─────────────────────────────────────────────────────
def derive_hypotheses(
    imb: dict,
    fisch: pd.Series,
    elev: pd.DataFrame,
    pca_var: float,
    sparse: dict,
) -> list[dict]:
    """
    Return a list of algorithm hypothesis dicts, each with:
      name, rank, macro_f1_expectation, key_reason, risk
    """
    top_feature  = fisch.index[0]
    top_f        = fisch.iloc[0]
    ir           = imb["imbalance_ratio"]
    soil_sp      = sparse["soil"]

    # ── Qualitative reasoning ──────────────────────────────────────────────
    # 1. Elevation dominates (F=1.60) — strong signal for axis-aligned splits → DT benefit
    # 2. Class ranges overlap in Elevation space (classes 2,5 and 3,4,6 share ranges)
    #    → nonlinear boundaries required → NN and RBF benefit over Linear SVM
    # 3. 44 sparse binary features (soil 2.5% active) → kNN Euclidean distance degrades
    #    → curse of dimensionality for kNN
    # 4. Mixed feature types (continuous + binary) → NN learns separate representations
    # 5. 103x imbalance → all models need class weighting, Macro F1 essential
    # 6. SGD-only NN constraint: no momentum/Adam → harder convergence, needs early stopping

    hyps = [
        {
            "name":              "Neural Network (sklearn + PyTorch)",
            "abbrev":            "NN",
            "rank":              1,
            "f1_range":          "0.68 – 0.78",
            "key_reason": (
                "Can learn complex non-linear interactions between Elevation, "
                "soil type, and wilderness area. Handles mixed continuous + binary "
                "features via learned embeddings. Large working sample "
                "supports effective gradient learning even under SGD-only constraint."
            ),
            "risk": (
                f"SGD-only constraint (no momentum/Adam) slows convergence. "
                f"{ir:.0f}x class imbalance may bias minority-class gradients."
            ),
        },
        {
            "name":    "Decision Tree",
            "abbrev":  "DT",
            "rank":    2,
            "f1_range": "0.60 – 0.72",
            "key_reason": (
                f"Elevation ({top_feature}, F={top_f:.2f}) creates clean axis-aligned "
                "splits that partition classes by altitude zone. Binary soil/wilderness "
                "features are natively handled without scaling. Cost-complexity pruning "
                "controls the bias-variance tradeoff."
            ),
            "risk": (
                f"Greedy splitting under {ir:.0f}x imbalance biases nodes toward "
                "majority classes (Spruce/Fir 36%, Lodgepole Pine 49%). "
                "Overlapping Elevation ranges for classes 2/5 and 3/4/6 require "
                "deep trees that may overfit."
            ),
        },
        {
            "name":    "SVM (RBF kernel)",
            "abbrev":  "SVM-RBF",
            "rank":    3,
            "f1_range": "0.55 – 0.68",
            "key_reason": (
                "RBF kernel captures non-linear Elevation boundaries and "
                "within-class chemical clustering. Class-weighted margin "
                "handles imbalance via balanced C parameter."
            ),
            "risk": (
                f"RBF Euclidean metric treats sparse binary soil features (2.5% active) "
                "equally with continuous features, diluting the kernel. "
                "Quadratic training cost is expensive on 20K samples. "
                "7-class OvR decomposition amplifies minority-class error."
            ),
        },
        {
            "name":    "SVM (Linear kernel)",
            "abbrev":  "SVM-Lin",
            "rank":    4,
            "f1_range": "0.45 – 0.58",
            "key_reason": (
                f"PCA explains only {pca_var*100:.1f}% of variance in 2 components, "
                "suggesting limited linear separability. Still competitive due to "
                "L2 margin regularization and the sparse binary feature structure "
                "(similar to one-hot encoding)."
            ),
            "risk": (
                "Overlapping Elevation ranges across classes 1/2 and 3/4/6 require "
                "nonlinear boundaries that a linear kernel cannot provide. "
                "Performance ceiling well below RBF."
            ),
        },
        {
            "name":    "k-Nearest Neighbors",
            "abbrev":  "kNN",
            "rank":    5,
            "f1_range": "0.40 – 0.55",
            "key_reason": (
                "Low-dimensional continuous features (Elevation in particular) "
                "provide strong local structure for nearest-neighbor queries."
            ),
            "risk": (
                f"54-dimensional space with {len(SOIL_COLS)} sparse binary soil features "
                f"(avg {soil_sp*100:.1f}% active) degrades Euclidean distance — "
                "curse of dimensionality. "
                f"{ir:.0f}x imbalance means minority-class neighborhoods are "
                "overwhelmed by majority-class examples at any k."
            ),
        },
    ]
    return hyps


# ── Printing ──────────────────────────────────────────────────────────────────
def print_hypothesis_report(
    df: pd.DataFrame,
    imb: dict,
    fisch: pd.Series,
    elev: pd.DataFrame,
    pca_var: float,
    sparse: dict,
    hyps: list[dict],
) -> None:
    n = len(df)
    ir = imb["imbalance_ratio"]

    print("=" * 70)
    print("HYPOTHESIS WORKFLOW — Forest Cover Type (Covertype)")
    print("=" * 70)

    # ── Dataset properties ────────────────────────────────────────────────
    print("\n── 1. Dataset Properties ──────────────────────────────────────────")
    print(f"  Samples (working)   : {n:,}")
    print(f"  Classes             : 7 (multiclass)")
    print(f"  Features            : {len(CONTINUOUS_COLS)} continuous, "
          f"{len(WILDERNESS_COLS)} wilderness binary, "
          f"{len(SOIL_COLS)} soil-type binary  →  {len(FEATURE_COLS)} total")
    print(f"  Missing values      : 0")
    print(f"  Class imbalance     : {ir:.1f}x  "
          f"(Class 2 {imb['pct'][2]:.1f}% vs Class 4 {imb['pct'][4]:.2f}%)")
    print(f"  Primary metric      : Macro F1  (imbalance prevents accuracy from being informative)")

    # ── Feature discriminability ──────────────────────────────────────────
    print("\n── 2. Feature Discriminability (Fisher F-ratio, continuous only) ────")
    for feat, fval in fisch.items():
        bar = "█" * int(fval * 10)
        flag = " ← most discriminating" if feat == fisch.index[0] else ""
        print(f"  {feat:<42} F={fval:.3f}  {bar}{flag}")

    # ── Elevation ranges ──────────────────────────────────────────────────
    print("\n── 3. Elevation Ranges by Class (key discriminating feature) ────────")
    print(f"  {'Class':<5} {'Name':<22} {'P5':>6} {'Median':>8} {'P95':>6} {'Mean':>8}")
    print("  " + "-" * 55)
    for c in range(1, 8):
        r = elev.loc[c]
        print(f"  {c:<5} {CLASS_NAMES[c]:<22} "
              f"{r['p05']:>6.0f} {r['median']:>8.0f} {r['p95']:>6.0f} {r['mean']:>8.1f}")
    print("  Note: Classes 2 & 5 share [2594–2929 m]; classes 3, 4, 6 share [2048–2715 m]")
    print("        → non-linear boundaries required within altitude zones")

    # ── Linear separability ───────────────────────────────────────────────
    print("\n── 4. Linear Separability Proxy ─────────────────────────────────────")
    print(f"  2-component PCA explains {pca_var*100:.1f}% of total variance")
    print("  Low fraction → significant non-linear structure → Linear SVM at a disadvantage")

    # ── Sparse binary features ────────────────────────────────────────────
    print("\n── 5. Sparse Binary Feature Analysis ────────────────────────────────")
    print(f"  Mean soil-type activity : {sparse['soil']*100:.1f}% per sample")
    print(f"  Mean wilderness activity: {sparse['wilderness']*100:.1f}% per sample")
    print("  Sparse binary features degrade Euclidean distance → kNN at a disadvantage")
    print("  Decision Trees and NNs handle binary features natively")

    # ── Hypotheses ────────────────────────────────────────────────────────
    print("\n── 6. Algorithm Hypotheses ──────────────────────────────────────────")
    print()
    print(f"  Predicted Macro F1 ranking:")
    print(f"    NN (sklearn/PyTorch) > DT > SVM-RBF > SVM-Linear > kNN")
    print()
    for h in hyps:
        print(f"  Rank {h['rank']}: {h['name']}  (expected Macro F1 ≈ {h['f1_range']})")
        # wrap key_reason to 65 chars
        reason_words = h["key_reason"].split()
        lines, cur = [], []
        for w in reason_words:
            cur.append(w)
            if len(" ".join(cur)) > 65:
                lines.append("    " + " ".join(cur[:-1]))
                cur = [w]
        if cur:
            lines.append("    " + " ".join(cur))
        print("  Why:")
        print("\n".join(lines))
        risk_words = h["risk"].split()
        lines, cur = [], []
        for w in risk_words:
            cur.append(w)
            if len(" ".join(cur)) > 65:
                lines.append("    " + " ".join(cur[:-1]))
                cur = [w]
        if cur:
            lines.append("    " + " ".join(cur))
        print("  Risk:")
        print("\n".join(lines))
        print()

    print("── 7. Preprocessing Requirements ────────────────────────────────────")
    print("  StandardScaler         : kNN, SVM, NN  (wide continuous ranges: Elevation 1859–3858)")
    print("  Decision Trees         : raw features (scale-invariant)")
    print("  Stratified 80/20 split : preserve 103x class imbalance across folds")
    print("  5-fold stratified CV   : Macro F1 as selection criterion")
    print("  class_weight='balanced': SVM and NN to compensate for imbalance")
    print()


# ── Figure ────────────────────────────────────────────────────────────────────
def plot_hypothesis_summary(
    df: pd.DataFrame,
    imb: dict,
    fisch: pd.Series,
    elev: pd.DataFrame,
    pca_var: float,
    hyps: list[dict],
) -> None:
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Hypothesis Summary — Forest Cover Type Dataset",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.93, bottom=0.07, left=0.06, right=0.97,
    )

    # ── Panel 1: Class distribution (log scale) ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = imb["counts"]
    bars = ax1.bar(
        range(1, 8), counts.values,
        color=CLASS_COLORS, edgecolor="white", linewidth=0.5,
    )
    ax1.set_yscale("log")
    ax1.set_xticks(range(1, 8))
    ax1.set_xticklabels(
        [f"C{i}\n{CLASS_NAMES[i][:7]}" for i in range(1, 8)],
        fontsize=7,
    )
    ax1.set_ylabel("Count (log scale)")
    ax1.set_title("Class Distribution\n(103× imbalance → Macro F1)", fontsize=9, fontweight="bold")
    for bar, cnt in zip(bars, counts.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
            f"{cnt:,}", ha="center", va="bottom", fontsize=6, rotation=30,
        )
    ax1.set_ylim(top=counts.max() * 5)

    # ── Panel 2: Fisher F-ratios ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    short_names = [c.replace("Horizontal_Distance_To_", "H_").replace("Vertical_Distance_To_", "V_")
                   for c in fisch.index]
    colors2 = ["#e07b39" if i == 0 else "#4C72B0" for i in range(len(fisch))]
    ax2.barh(range(len(fisch)), fisch.values, color=colors2, edgecolor="white")
    ax2.set_yticks(range(len(fisch)))
    ax2.set_yticklabels(short_names, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Fisher F-ratio")
    ax2.set_title("Feature Discriminability\n(Elevation dominates → DT benefit)", fontsize=9, fontweight="bold")
    ax2.axvline(fisch.mean(), color="gray", lw=1, ls="--", alpha=0.7, label=f"mean={fisch.mean():.3f}")
    ax2.legend(fontsize=7)
    # annotate top feature
    ax2.text(fisch.iloc[0] * 0.02, 0, f"F={fisch.iloc[0]:.3f}", va="center", fontsize=7, color="white", fontweight="bold")

    # ── Panel 3: Elevation ranges by class ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    y_pos = list(range(7))
    for i, c in enumerate(range(1, 8)):
        p05    = elev.loc[c, "p05"]
        p95    = elev.loc[c, "p95"]
        median = elev.loc[c, "median"]
        ax3.barh(
            i, p95 - p05, left=p05,
            color=CLASS_COLORS[i], alpha=0.75, edgecolor="white", height=0.6,
        )
        ax3.plot(median, i, "w|", markersize=10, markeredgewidth=2)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f"C{c} {CLASS_NAMES[c][:10]}" for c in range(1, 8)], fontsize=7)
    ax3.set_xlabel("Elevation (m)")
    ax3.set_title("Elevation Ranges by Class\n(5th–95th pct, white = median)", fontsize=9, fontweight="bold")
    ax3.invert_yaxis()
    # shade overlap zones
    ax3.axvspan(2594, 2929, alpha=0.08, color="red", label="C2/C5 overlap")
    ax3.axvspan(2048, 2715, alpha=0.05, color="blue", label="C3/C4/C6 overlap")
    ax3.legend(fontsize=6, loc="lower right")

    # ── Panel 4: Feature type breakdown ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    categories   = ["Continuous\n(10)", "Wilderness\nBinary (4)", "Soil-type\nBinary (40)"]
    cat_counts   = [len(CONTINUOUS_COLS), len(WILDERNESS_COLS), len(SOIL_COLS)]
    cat_colors   = ["#4C72B0", "#55A868", "#C44E52"]
    wedges, texts, autotexts = ax4.pie(
        cat_counts, labels=categories, colors=cat_colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 8},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")
    ax4.set_title("Feature Type Breakdown\n(44/54 sparse binary → kNN penalty)", fontsize=9, fontweight="bold")

    # ── Panel 5: PCA 2D scatter (class overlap visualization) ─────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sample_idx = df.sample(n=min(3000, len(df)), random_state=RANDOM_SEED).index
    sub = df.loc[sample_idx]
    X   = StandardScaler().fit_transform(sub[FEATURE_COLS].values)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z   = pca.fit_transform(X)
    for i, c in enumerate(range(1, 8)):
        mask = sub[TARGET_COL].values == c
        ax5.scatter(
            Z[mask, 0], Z[mask, 1],
            c=CLASS_COLORS[i], s=4, alpha=0.4, label=CLASS_NAMES[c],
        )
    ax5.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax5.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax5.set_title(
        f"PCA 2D Projection (total var={pca_var*100:.1f}%)\n"
        "Low → significant non-linear structure",
        fontsize=9, fontweight="bold",
    )
    ax5.legend(fontsize=5, markerscale=2, ncol=2, loc="upper right")

    # ── Panel 6: Hypothesis ranking table ─────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.set_title("Predicted Algorithm Ranking\n(Macro F1)", fontsize=9, fontweight="bold", pad=10)

    row_colors_h = [
        "#d4edda",   # green  – NN
        "#cce5ff",   # blue   – DT
        "#fff3cd",   # yellow – SVM-RBF
        "#fde8d8",   # orange – SVM-Linear
        "#f8d7da",   # red    – kNN
    ]
    medal = [" #1 ", " #2 ", " #3 ", " #4 ", " #5 "]
    rows = []
    for h, m in zip(hyps, medal):
        rows.append([m, h["abbrev"], h["f1_range"]])

    tbl = ax6.table(
        cellText=rows,
        colLabels=["Rank", "Model", "Exp. Macro F1"],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.15, 1.0, 0.75],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(3):
        tbl[0, j].set_facecolor("#343a40")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, rc in enumerate(row_colors_h):
        for j in range(3):
            tbl[i + 1, j].set_facecolor(rc)

    # sub-note
    ax6.text(
        0.5, 0.08,
        "Hypothesis: NN can model non-linear Elevation ×\n"
        "soil/wilderness interactions; kNN limited by\n"
        "54-dim sparse binary space (curse of dim.)",
        ha="center", va="center", fontsize=7, style="italic",
        transform=ax6.transAxes,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "hypothesis_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {os.path.abspath(out_path)}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_data()

    # ── Compute statistics ────────────────────────────────────────────────
    print("\nComputing hypothesis-relevant statistics ...")
    imb     = class_imbalance_stats(df)
    fisch   = fischer_ratios(df)
    elev    = elevation_ranges_by_class(df)
    pca_var = pca_overlap_score(df)
    sparse  = binary_sparsity(df)

    # ── Derive and print hypotheses ───────────────────────────────────────
    hyps = derive_hypotheses(imb, fisch, elev, pca_var, sparse)
    print_hypothesis_report(df, imb, fisch, elev, pca_var, sparse, hyps)

    # ── Generate figure ───────────────────────────────────────────────────
    print("Generating hypothesis summary figure ...")
    plot_hypothesis_summary(df, imb, fisch, elev, pca_var, hyps)
    print("\nDone.")


if __name__ == "__main__":
    main()
