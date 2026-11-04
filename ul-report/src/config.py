"""
Central configuration for the UL Report (CS7641 Summer 2026).

SL/OL continuity (assignment §3 "Connection to Prior Reports"):
  - the SAME ~20,000-instance stratified Covertype sample,
  - the SAME target definition (Cover_Type 1..7 -> 0..6),
  - the SAME preprocessing philosophy (StandardScaler on the continuous
    cartographic features only; binary wilderness/soil indicators left as 0/1),
  - the SAME train/val/test split strategy for the downstream neural network.

Everything that must stay fixed for a controlled comparison lives here so that
differences in results are causal rather than accidental. Any deviation must be
disclosed in the report (FAQ §4 "Dataset, Labels, and Preprocessing").

Fast smoke test: set UL_FAST=1 to shrink sweeps/seeds/epochs for a quick check.
"""

from __future__ import annotations

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
UL_DIR    = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(UL_DIR)


def _first_existing(*paths: str) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]  # fall back to the first (error surfaces later with context)


# Reuse the EXACT Covertype data from the SL Report. The SL data dir is "Data"
# on this checkout, but tolerate a lowercase "data" too for portability.
DATA_PATH = _first_existing(
    os.path.join(REPO_ROOT, "sl-report", "Data", "covtype.data"),
    os.path.join(REPO_ROOT, "sl-report", "data", "covtype.data"),
)
INFO_PATH = _first_existing(
    os.path.join(REPO_ROOT, "sl-report", "Data", "covtype.info"),
    os.path.join(REPO_ROOT, "sl-report", "data", "covtype.info"),
)

# If the SL EDA already wrote a stratified sample, point here to reuse it
# verbatim. The SL Report built this file with n=20,000, seed=42, proportional
# stratified sampling (confirmed in the SL run log). If the file on disk is a
# truncated smoke-test copy (< ~0.8*SAMPLE_SIZE rows), data.py regenerates the
# identical 20k sample from the full Covertype data using the SAME procedure and
# seed, so SL continuity is preserved without editing the SL committed file.
SL_SAMPLE_PATH = _first_existing(
    os.path.join(REPO_ROOT, "sl-report", "outputs", "sample.csv"),
    os.path.join(REPO_ROOT, "ol-report", "outputs", "sample.csv"),
)

OUT_DIR    = os.path.join(UL_DIR, "outputs")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
RESULT_DIR = os.path.join(OUT_DIR, "results")
LOG_DIR    = os.path.join(OUT_DIR, "logs")
for _d in (OUT_DIR, FIG_DIR, RESULT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# UL-owned cache for a regenerated sample (used only if the SL sample on disk is
# a truncated smoke copy). Writing here never touches the SL Report's file.
UL_SAMPLE_PATH = os.path.join(OUT_DIR, "sample.csv")

# ── Reproducibility ───────────────────────────────────────────────────────────
FAST = os.environ.get("UL_FAST", "0") == "1"

# Multiple seeds are required wherever a method is stochastic (ICA init, RP
# projection, K-Means/GMM init, NN training). RP especially must be evaluated
# across seeds (FAQ §8 "What should I report for Randomized Projections").
SEEDS       = [int(s) for s in os.environ.get("UL_SEEDS", "0,1,2").split(",")]
SAMPLE_SEED = 42   # matches the seed used to build the SL sample.
SPLIT_SEED  = 42   # train/val/test split seed (fixed, matches OL).

# ── Dataset / sample / split ──────────────────────────────────────────────────
SAMPLE_SIZE = 2_000 if FAST else 20_000
TEST_FRAC   = 0.20   # held-out test set, touched ONLY at final NN evaluation.
VAL_FRAC    = 0.20   # of the remaining train pool -> validation.

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
BINARY_COLS     = WILDERNESS_COLS + SOIL_COLS
TARGET_COL      = "Cover_Type"

N_FEATURES = len(FEATURE_COLS)   # 54
N_CLASSES  = 7                   # Cover_Type 1..7 -> mapped to 0..6 internally

CLASS_NAMES = {
    0: "Spruce/Fir", 1: "Lodgepole Pine", 2: "Ponderosa Pine",
    3: "Cottonwood/Willow", 4: "Aspen", 5: "Douglas-fir", 6: "Krummholz",
}

# Preprocessing choice for the EXPLORATORY unsupervised analysis (Parts 1-3):
# scale continuous features only, leave binary indicators as 0/1. This keeps the
# 0/1 categorical meaning of wilderness/soil while preventing large-range
# cartographic variables (elevation, distances) from dominating Euclidean
# geometry and PCA variance (FAQ §4). Set SCALE_ALL=1 to scale everything
# instead (disclose the change if you do).
SCALE_ALL = os.environ.get("UL_SCALE_ALL", "0") == "1"

# ── Part 1: Clustering on original features ───────────────────────────────────
# k=7 is a reference (seven cover types) but NOT assumed correct; sweep around it.
KMEANS_KS      = [2, 3, 5] if FAST else [2, 3, 4, 5, 6, 7, 8, 10, 12]
KMEANS_N_INIT  = 10
KMEANS_MAX_ITER = 300

# Upper end extended past k=7 so BIC/AIC can plateau or turn instead of pinning
# at the grid ceiling (a min-BIC that always lands on the largest n is a red flag).
GMM_COMPONENTS = [2, 3, 5] if FAST else [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
GMM_COV_TYPES  = ["full", "diag"] if FAST else ["full", "tied", "diag", "spherical"]
GMM_N_INIT     = 3
GMM_MAX_ITER   = 200
GMM_REG_COVAR  = 1e-6

# Number of points used for silhouette (O(n^2) memory); subsample for speed.
SILHOUETTE_SAMPLE = 2_000 if FAST else 5_000

# ── Part 2: Dimensionality reduction ──────────────────────────────────────────
# Each method's component count is chosen by METHOD-APPROPRIATE evidence, not a
# single shared number (FAQ §8): PCA by cumulative explained variance, ICA by the
# retained non-Gaussian rank, RP by the distance-distortion / JL reference. The
# per-method justified dims are computed at runtime in dr.justified_components().
PCA_VAR_TARGET = 0.90
# Candidate component counts for the DR sweeps (scree/reconstruction/distortion
# curves + the NN performance-vs-dim curve). Wide enough to show "how low can we
# go" and to let RP at high dim approach PCA at low dim.
DR_COMPONENTS  = [2, 5, 10, 15, 20, 30] if not FAST else [2, 5, 10]
# NN-after-DR performance-vs-n_components sweep (Part 4). One curve answers
# "how many components are enough" far better than a single-dim bar chart.
NN_DR_SWEEP_DIMS = [5, 10, 20, 30] if not FAST else [5, 10]
# RP JL-distortion tolerance used to justify the RP component count (eps in the
# Johnson-Lindenstrauss sense: target ~+/-eps multiplicative distance error).
RP_JL_EPS = 0.30
# Optional manual overrides (else resolved per method at runtime).
ICA_N_COMPONENTS = None
RP_N_COMPONENTS  = None
ICA_MAX_ITER     = 500
ICA_TOL          = 1e-3
ICA_FUN          = "logcosh"      # FastICA contrast function.
RP_KIND          = "gaussian"     # "gaussian" or "sparse" Random Projection.
DIST_PAIRS       = 5_000          # random pairs for RP distance-preservation check.
# RP is the least stable method, so it gets the MOST seeds -- seed variability is
# the whole point of the RP section (FAQ §8). Others stay at SEEDS.
RP_SEEDS = [0, 1, 2] if FAST else list(range(10))

# ── Part 4: Neural network after DR (leakage-safe) ────────────────────────────
# Best justified recipe informed by SL (150-unit ReLU MLP, class-weighted CE) and
# OL (Adam optimizer family, light weight decay, early stopping). Topology is held
# fixed across all representations; only the INPUT dimension changes with DR.
HIDDEN_SIZES    = [150]
DROPOUT_P       = 0.0
CLASS_WEIGHTED  = True
NN_OPTIMIZER    = "adam"          # OL finding: Adam > plain SGD for this problem.
NN_LR           = 1e-3
NN_WEIGHT_DECAY = 1e-4            # OL regularization finding (modest).
NN_BATCH_SIZE   = 128
NN_EPOCHS       = 20 if FAST else 100
# OL Part 3 measured early stopping on VALIDATION LOSS as the single most harmful
# regularizer for Macro-F1 (0.630 -> 0.573), which is the same val-loss/Macro-F1
# dissociation that was OL's headline finding. The best-justified recipe from SL+OL
# is therefore Adam + light L2 on a FIXED epoch budget (OL Part 2/3 protocol), not
# val-loss early stopping. Set UL_EARLY_STOP=1 to restore the stopping variant.
NN_EARLY_STOP_PATIENCE = 10 if os.environ.get("UL_EARLY_STOP", "0") == "1" else None
DEVICE          = os.environ.get("UL_DEVICE", "cpu")   # "cuda"/"mps" if available.

# ── Extra credit: nonlinear manifold learning ─────────────────────────────────
MANIFOLD_METHOD     = os.environ.get("UL_MANIFOLD", "tsne")  # tsne|umap|isomap|lle|spectral
MANIFOLD_SAMPLE     = 2_000 if FAST else 4_000
MANIFOLD_COMPONENTS = 2
