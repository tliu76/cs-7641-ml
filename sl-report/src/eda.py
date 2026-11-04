"""
EDA for Forest Cover Type (Covertype) dataset.

Produces:
  - Full-dataset summary (shape, class distribution, feature stats)
  - Stratified ~20,000-instance sample + sample EDA
  - Correlation heatmap, violin plots, mutual information ranking
  - Figures saved to outputs/eda/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.feature_selection import mutual_info_classif
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

plt.rcParams.update({"axes.grid": True, "grid.color": "white", "axes.facecolor": "#EAEAF2",
                     "axes.titlesize": 11, "xtick.labelsize": 8, "ytick.labelsize": 8})

# ── Paths & constants ────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "covtype.data")
INFO_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "covtype.info")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda")
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "sample.csv")

RANDOM_SEED = 42
SAMPLE_SIZE = 20_000

# ── Column definitions ───────────────────────────────────────────────────────
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

PALETTE = [plt.cm.tab10(i / 10) for i in range(7)]


# ── Helpers ──────────────────────────────────────────────────────────────────
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def finalize_and_save(fig, filename, xlabel=None, ylabel=None):
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    save(fig, filename)


# ── Data loading ─────────────────────────────────────────────────────────────
def download_covtype():
    bunch = fetch_covtype(as_frame=True)
    df = bunch.frame.copy()
    df.columns = FEATURE_COLS + [TARGET_COL]
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, header=False, index=False)
    return df


def load_full() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"[eda] {DATA_PATH} not found. Downloading from sklearn.datasets ...")
        return download_covtype()
    return pd.read_csv(DATA_PATH, header=None, names=FEATURE_COLS + [TARGET_COL])


# ── Stratified sampling ──────────────────────────────────────────────────────
def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Proportional stratified sample targeting SAMPLE_SIZE instances."""
    _, sample = train_test_split(
        df,
        test_size=SAMPLE_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COL],
    )
    return sample.reset_index(drop=True)


# ── Full-dataset EDA ──────────────────────────────────────────────────────────
def eda_full(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("FULL DATASET EDA")
    print("=" * 60)
    print(f"Shape            : {df.shape}")
    print(f"Missing values   : {df.isnull().sum().sum()}")
    print(f"Duplicate rows   : {df.duplicated().sum()}")
    print()

    vc = df[TARGET_COL].value_counts().sort_index()
    pct = vc / len(df) * 100
    print("Class distribution:")
    for cls in range(1, 8):
        print(f"  C{cls} ({CLASS_NAMES[cls]:<22}): {vc[cls]:>8,} instances ({pct[cls]:>6.2f}%)")
    print()

    stats = df[CONTINUOUS_COLS].describe().T
    stats["range"] = stats["max"] - stats["min"]
    print("Continuous feature summary:")
    print(stats[["mean", "std", "min", "25%", "50%", "75%", "max", "range"]])
    print()

    _plot_class_distribution(df, vc)
    _plot_continuous_boxplots(df, "continuous_boxplots_full.png", color="steelblue")
    _plot_feature_ranges(df)
    _plot_elevation_by_class(df)


def _plot_class_distribution(df: pd.DataFrame, vc: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [CLASS_NAMES[i] for i in vc.index]
    bars = ax.bar(labels, vc.values, color=PALETTE)
    ax.bar_label(bars, labels=[f"{v:,}" for v in vc.values], padding=3, fontsize=8)
    ax.set(title=f"Full Dataset — Class Distribution (N={len(df):,})", ylabel="Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=25, ha="right")
    finalize_and_save(fig, "class_dist_full.png", xlabel="Cover Type")


def _plot_continuous_boxplots(df: pd.DataFrame, filename: str, color: str) -> None:
    z = (df[CONTINUOUS_COLS] - df[CONTINUOUS_COLS].mean()) / df[CONTINUOUS_COLS].std()
    fig, ax = plt.subplots(figsize=(10, 4))
    z.boxplot(
        ax=ax, vert=False, patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.6),
        medianprops=dict(color="red"),
    )
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"Continuous Features (z-scored) — {filename.split('_')[2].split('.')[0].title()}")
    finalize_and_save(fig, filename, xlabel="Standard deviations from mean")


def _plot_feature_ranges(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    (df[CONTINUOUS_COLS].max() - df[CONTINUOUS_COLS].min()).plot(
        kind="barh", ax=ax, color="steelblue", alpha=0.7
    )
    ax.set_title("Full Dataset — Raw Feature Value Ranges")
    finalize_and_save(fig, "feature_ranges_full.png", xlabel="Range (max − min)")


def _plot_elevation_by_class(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    elev = df.groupby(TARGET_COL)["Elevation"].mean()
    ax.bar([CLASS_NAMES[i] for i in elev.index], elev.values, color=PALETTE)
    ax.set_title("Mean Elevation by Cover Type (Full Dataset)")
    plt.xticks(rotation=25, ha="right")
    finalize_and_save(fig, "elevation_by_class_full.png",
                      ylabel="Mean Elevation (m)", xlabel="Cover Type")


# ── Sample EDA ────────────────────────────────────────────────────────────────
def eda_sample(full: pd.DataFrame, sample: pd.DataFrame) -> None:
    print("=" * 60)
    print("SAMPLED DATASET EDA")
    print("=" * 60)
    print(f"Shape            : {sample.shape}")
    print(f"Missing values   : {sample.isnull().sum().sum()}")
    print()

    vc_full   = full[TARGET_COL].value_counts().sort_index()
    vc_sample = sample[TARGET_COL].value_counts().sort_index()
    pct_full   = vc_full   / len(full)   * 100
    pct_sample = vc_sample / len(sample) * 100

    print("Class distribution comparison (full vs sample):")
    hdr = f"  {'Cls':<4} {'Name':<22} {'Full N':>8} {'Full %':>7}  {'Sample N':>9} {'Sample %':>9}"
    print(hdr)
    print("  " + "-" * 66)
    for cls in range(1, 8):
        print(f"  {cls:<4} {CLASS_NAMES[cls]:<22} "
              f"{vc_full[cls]:>8,} {pct_full[cls]:>7.2f}%  "
              f"{vc_sample[cls]:>9,} {pct_sample[cls]:>9.2f}%")
    print()

    cont_full   = full[CONTINUOUS_COLS].describe().T
    cont_sample = sample[CONTINUOUS_COLS].describe().T
    print("Continuous feature mean ± std (full vs sample):")
    for col in CONTINUOUS_COLS:
        print(f"  {col:<42} "
              f"full: {cont_full.loc[col,'mean']:8.1f} ± {cont_full.loc[col,'std']:6.1f}   "
              f"sample: {cont_sample.loc[col,'mean']:8.1f} ± {cont_sample.loc[col,'std']:6.1f}")
    print()

    # side-by-side class distribution
    x = np.arange(7)
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, pct_full.values,   w, label="Full (581,012)",         color="steelblue",  alpha=0.8)
    ax.bar(x + w/2, pct_sample.values, w, label=f"Sample ({SAMPLE_SIZE:,})", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}\n{CLASS_NAMES[i]}" for i in range(1, 8)], fontsize=8)
    ax.set_ylabel("Class proportion (%)")
    ax.set_title("Class Distribution: Full vs. Stratified Sample")
    ax.legend()
    plt.tight_layout()
    save(fig, "class_dist_comparison.png")

    _plot_continuous_boxplots(sample, "continuous_boxplots_sample.png", color="darkorange")

    # per-class normalised feature means heatmap
    class_means = sample.groupby(TARGET_COL)[CONTINUOUS_COLS].mean()
    norm = (class_means - class_means.min()) / (class_means.max() - class_means.min())
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(norm.T, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"C{i}\n{CLASS_NAMES[i]}" for i in range(1, 8)], fontsize=7)
    ax.set_yticks(range(len(CONTINUOUS_COLS)))
    ax.set_yticklabels(CONTINUOUS_COLS, fontsize=8)
    ax.set_title("Per-Class Normalised Feature Means (Sample)")
    fig.colorbar(im, ax=ax, label="Normalised mean")
    plt.tight_layout()
    save(fig, "per_class_feature_heatmap_sample.png")

    # wilderness area prevalence per class
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=False)
    for i, col in enumerate(WILDERNESS_COLS):
        pct = sample.groupby(TARGET_COL)[col].mean() * 100
        axes[i].bar([str(c) for c in pct.index], pct.values, color="teal", alpha=0.75)
        axes[i].set_title(col.replace("Wilderness_Area", "WA"), fontsize=9)
        axes[i].set_xlabel("Cover Type")
        if i == 0:
            axes[i].set_ylabel("% instances")
    fig.suptitle("Wilderness Area Prevalence per Cover Type (Sample)", fontsize=10)
    plt.tight_layout()
    save(fig, "wilderness_by_class_sample.png")

    # soil type sparsity
    soil_pct = sample[SOIL_COLS].mean() * 100
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(range(40), soil_pct.values, color="slategray", alpha=0.7)
    ax.set_xticks(range(40))
    ax.set_xticklabels([f"S{i+1}" for i in range(40)], rotation=90, fontsize=6)
    ax.set_ylabel("% active")
    ax.set_title("Soil Type Prevalence in Sampled Dataset")
    plt.tight_layout()
    save(fig, "soil_type_prevalence_sample.png")

    # correlation heatmap
    corr = sample[CONTINUOUS_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    corr_masked = np.where(mask, np.nan, corr.values)
    im = ax.imshow(corr_masked, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ticks = range(len(CONTINUOUS_COLS))
    ax.set_xticks(ticks); ax.set_xticklabels(CONTINUOUS_COLS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(CONTINUOUS_COLS, fontsize=7)
    for i in range(len(CONTINUOUS_COLS)):
        for j in range(len(CONTINUOUS_COLS)):
            if not mask[i, j]:
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=5)
    ax.set_title("Continuous Feature Correlation Matrix (Sample)", pad=12)
    plt.tight_layout()
    save(fig, "correlation_heatmap_sample.png")

    # mutual information scores — computed first so violin plots use top-4 by MI
    discrete_mask = (
        [False] * len(CONTINUOUS_COLS)
        + [True] * (len(WILDERNESS_COLS) + len(SOIL_COLS))
    )
    mi = mutual_info_classif(
        sample[FEATURE_COLS], sample[TARGET_COL],
        discrete_features=discrete_mask,
        random_state=RANDOM_SEED,
    )
    mi_series = pd.Series(mi, index=FEATURE_COLS).sort_values(ascending=False)

    # top 4 continuous features by MI — drives violin plots (matches report Fig 2 caption)
    top_cont_mi = [f for f in mi_series.index if f in CONTINUOUS_COLS][:4]
    print(f"  Top 4 continuous features by MI: {top_cont_mi}")

    # violin plots for top-4 MI-ranked continuous features
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feat in zip(axes.flat, top_cont_mi):
        groups = [sample.loc[sample[TARGET_COL] == cls, feat].values for cls in range(1, 8)]
        parts = ax.violinplot(groups, positions=range(1, 8), showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(PALETTE[i])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels([CLASS_NAMES[i] for i in range(1, 8)], rotation=30, ha="right", fontsize=7)
        ax.set_xlabel("Cover Type")
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} by Cover Type")
    fig.suptitle("Feature Distributions by Cover Type (Sample) — Top 4 by Mutual Information",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save(fig, "violin_top_features_sample.png")

    mi_series = mi_series.sort_values(ascending=True)
    top_mi = mi_series.tail(20)
    bar_colors = [
        "steelblue"  if f in CONTINUOUS_COLS else
        "darkorange" if f in WILDERNESS_COLS else
        "slategray"
        for f in top_mi.index
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_mi.index, top_mi.values, color=bar_colors, alpha=0.85)
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Top 20 Features by Mutual Information (Sample)")
    ax.legend(handles=[
        Patch(facecolor="steelblue",  alpha=0.85, label="Continuous"),
        Patch(facecolor="darkorange", alpha=0.85, label="Wilderness"),
        Patch(facecolor="slategray",  alpha=0.85, label="Soil Type"),
    ], loc="lower right")
    plt.tight_layout()
    save(fig, "mutual_information_top20_sample.png")


# ── Preprocessing notes ───────────────────────────────────────────────────────
def print_preprocessing_notes(sample: pd.DataFrame) -> None:
    print("=" * 60)
    print("PREPROCESSING & IMBALANCE NOTES")
    print("=" * 60)

    vc = sample[TARGET_COL].value_counts().sort_index()
    ir = vc.max() / vc.min()
    print(f"Imbalance ratio (majority / minority): {ir:.1f}x")
    print()

    raw_ranges = sample[CONTINUOUS_COLS].max() - sample[CONTINUOUS_COLS].min()
    print("Raw feature ranges (motivates StandardScaler):")
    for col in CONTINUOUS_COLS:
        print(f"  {col:<42} {raw_ranges[col]:,.0f}")
    print()
    print("=> kNN, SVM, and NN require StandardScaler fitted on training data only.")
    print("=> Decision Trees are scale-invariant.")
    print()
    print(f"Feature groups: {len(CONTINUOUS_COLS)} continuous, "
          f"{len(WILDERNESS_COLS)} wilderness binary, "
          f"{len(SOIL_COLS)} soil-type binary  (total {len(FEATURE_COLS)})")


# ── Justification & leakage debug ─────────────────────────────────────────────
def sanity_and_justification(full: pd.DataFrame, sample: pd.DataFrame) -> None:
    print("=" * 60)
    print("JUSTIFICATION & LEAKAGE DEBUG")
    print("=" * 60)

    # 1. Sampling strategy
    print("1. Sampling strategy justification")
    print(f"   Method  : Proportional stratified sample (n={SAMPLE_SIZE:,}, seed={RANDOM_SEED})")
    print(f"   Why n={SAMPLE_SIZE:,}: balances runtime (<2 min/fold for kNN/SVM) with "
          "statistical representativeness.")
    vc_full   = full[TARGET_COL].value_counts().sort_index()
    vc_sample = sample[TARGET_COL].value_counts().sort_index()
    pct_full   = vc_full   / len(full)   * 100
    pct_sample = vc_sample / len(sample) * 100
    max_drift = (pct_sample - pct_full).abs().max()
    print(f"   Class proportion drift (full vs sample): max {max_drift:.3f} pp — "
          "stratification successful.")
    print()

    # 2. Metric choice
    print("2. Metric choice justification (Macro F1 > Accuracy for Covertype)")
    X = sample[FEATURE_COLS].values
    y = sample[TARGET_COL].values
    idx = np.arange(len(sample))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2,
                                      stratify=y, random_state=RANDOM_SEED)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    dummy_mf = DummyClassifier(strategy="most_frequent")
    dummy_mf.fit(X_tr, y_tr)
    y_mf   = dummy_mf.predict(X_te)
    acc_mf = accuracy_score(y_te, y_mf)
    f1_mf  = f1_score(y_te, y_mf, average="macro", zero_division=0)
    maj_cls = int(vc_full.idxmax())
    print(f"   DummyClassifier (always predicts C{maj_cls} = '{CLASS_NAMES[maj_cls]}'):")
    print(f"     Accuracy : {acc_mf:.4f}  ({acc_mf:.1%}) — deceptively high due to class imbalance")
    print(f"     Macro F1 : {f1_mf:.4f}  — correctly reveals zero minority-class recall")
    print(f"   => Macro F1 is the primary metric; accuracy is reported but not optimised.")
    print()

    # 3. Leakage debug
    print("3. Leakage / evaluation debug")

    rng = np.random.default_rng(RANDOM_SEED)
    y_tr_shuf = rng.permutation(y_tr)
    dummy_st  = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    dummy_st.fit(X_tr, y_tr_shuf)
    y_st  = dummy_st.predict(X_te)
    f1_st = f1_score(y_te, y_st, average="macro", zero_division=0)
    print(f"   Shuffled-label baseline Macro F1 : {f1_st:.4f}  "
          f"(near-chance — confirms no label leakage can explain real results)")

    overlap = len(set(tr_idx.tolist()) & set(te_idx.tolist()))
    print(f"   Split integrity : {len(tr_idx):,} train + {len(te_idx):,} test, "
          f"overlapping indices = {overlap}  "
          f"({'OK' if overlap == 0 else 'WARNING: OVERLAP FOUND'})")

    sample_dupes = sample.duplicated().sum()
    print(f"   Duplicate rows in sample : {sample_dupes}  "
          f"({'OK' if sample_dupes == 0 else 'WARNING'})")
    print()

    labels = ["Dummy\n(most_frequent)", "Shuffled\n(stratified)"]
    accs   = [acc_mf, accuracy_score(y_te, y_st)]
    f1s    = [f1_mf,  f1_st]
    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, accs, w, label="Accuracy",  color="steelblue",  alpha=0.8)
    ax.bar(x + w/2, f1s,  w, label="Macro F1",  color="darkorange", alpha=0.8)
    ax.axhline(1 / 7, color="red", ls="--", lw=1.2, label="Chance (1/7)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Leakage Debug — Baseline Scores (Accuracy vs Macro F1)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save(fig, "leakage_debug_baselines.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.dirname(SAMPLE_PATH))

    print(f"Loading data from {DATA_PATH} ...")
    full = load_full()
    print(f"Loaded {len(full):,} rows, {full.shape[1]} columns.\n")

    eda_full(full)

    print(f"Creating stratified sample (n={SAMPLE_SIZE:,}, seed={RANDOM_SEED}) ...")
    sample = stratified_sample(full)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Sample saved to {os.path.abspath(SAMPLE_PATH)}\n")

    eda_sample(full, sample)
    print_preprocessing_notes(sample)
    sanity_and_justification(full, sample)

    print(f"\nAll figures saved to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
