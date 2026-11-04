"""
Central configuration for the OL Report.

Everything that must stay FIXED across comparisons (dataset sample, split,
preprocessing, backbone, seeds, batch size, metric definitions) lives here so
that differences in results are causal rather than accidental (assignment §
"Compute accounting and fairness"). Any per-part deviation must be documented.
"""

from __future__ import annotations

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# config.py lives in <repo>/ol-report/src/, so the repo root is three levels up.
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
OL_DIR     = os.path.dirname(SRC_DIR)
REPO_ROOT  = os.path.dirname(OL_DIR)

# Reuse the EXACT Covertype data from the SL Report (SL continuity requirement).
SL_DATA_DIR = os.path.join(REPO_ROOT, "sl-report", "data")
DATA_PATH   = os.path.join(SL_DATA_DIR, "covtype.data")
INFO_PATH   = os.path.join(SL_DATA_DIR, "covtype.info")

# If your SL EDA already wrote a stratified sample, point here to reuse it
# verbatim. If missing, data.py rebuilds it from SAMPLE_SIZE + SAMPLE_SEED and
# warns that you must confirm it matches SL.
SL_SAMPLE_PATH = os.path.join(REPO_ROOT, "sl-report", "outputs", "sample.csv")

OUT_DIR     = os.path.join(OL_DIR, "outputs")
FIG_DIR     = os.path.join(OUT_DIR, "figures")
RESULT_DIR  = os.path.join(OUT_DIR, "results")   # csv/json trajectories + tables
LOG_DIR     = os.path.join(OUT_DIR, "logs")

for _d in (OUT_DIR, FIG_DIR, RESULT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
# >= 3 seeds required for all main claims (assignment § "Do I need multiple seeds").
# Sizes may be shrunk via env vars for a fast smoke test (see OL_* below); the
# defaults are the values used for the reported runs.
SEEDS       = [int(s) for s in os.environ.get("OL_SEEDS", "0,1,2").split(",")]
SAMPLE_SEED = 42         # matches the seed used to build the SL sample.csv.
SPLIT_SEED  = 42         # train/val/test split seed (fixed across all parts).

# ── Dataset / sample / split ──────────────────────────────────────────────────
# ~20,000-instance stratified sample, matching the SL Report unless justified.
SAMPLE_SIZE = 20_000
TEST_FRAC   = 0.20       # held-out test set, touched ONLY at final evaluation.
VAL_FRAC    = 0.20       # of the remaining train pool -> validation.

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

N_FEATURES = len(FEATURE_COLS)   # 54
N_CLASSES  = 7                   # Cover_Type 1..7  -> mapped to 0..6 internally

CLASS_NAMES = {
    0: "Spruce/Fir", 1: "Lodgepole Pine", 2: "Ponderosa Pine",
    3: "Cottonwood/Willow", 4: "Aspen", 5: "Douglas-fir", 6: "Krummholz",
}

# ── PyTorch backbone (matches the finalized SL architecture) ──────────────────
# Finalized SL PyTorch MLP: one hidden layer of 150 ReLU units (54 -> 150 -> 7),
# class-weighted cross-entropy. This is the single source of truth for the OL
# backbone; the topology stays fixed across all parts (only Part 3 alters
# regularization such as dropout).
HIDDEN_SIZES = [150]             # SL finalized PyTorch MLP (single hidden layer).
DROPOUT_P    = 0.0               # base backbone has NO dropout (Part 3 varies it).
CLASS_WEIGHTED = True            # SL used balanced class-weighted CrossEntropyLoss.

# ── Shared training budget (kept fixed for fair comparison) ───────────────────
# Baseline / Part-2 / Part-3 use a FIXED epoch budget (no early stopping outside
# the Part-3 early-stopping condition) so all optimizers get identical compute.
BATCH_SIZE  = 128
EPOCHS      = int(os.environ.get("OL_EPOCHS", "100"))   # gradient-based budget.
BASELINE_LR = 0.1                # SL PyTorch SGD-only learning rate.
DEVICE      = "cpu"              # set "cuda"/"mps" if available; report hardware.

# ── Part 1: Randomized Optimization on final layers ───────────────────────────
# The SL backbone has two Linear layers (54->150, 150->7). We freeze the learned
# representation (first layer) and run RO over ONLY the final classifier layer,
# i.e. 150*7 + 7 = 1,057 trainable parameters -- well under the 50k cap.
RO_TRAINABLE_LAYERS = 1          # final classifier layer unfrozen for RO.
RO_PARAM_CAP        = 50_000     # hard cap on trainable RO parameters.
RO_FUNC_EVAL_BUDGET = int(os.environ.get("OL_RO_BUDGET", "3000"))  # 1 objective call == 1 func eval.
RO_VAL_SUBSET       = None       # None => full val set; else fixed subset size (disclose).

# ── Part 2: Adam-family ablations ─────────────────────────────────────────────
# Coarse, comparable search per optimizer family (fair-protocol requirement).
LR_GRID_SGD  = [1e-2, 3e-2, 1e-1]
LR_GRID_ADAM = [3e-4, 1e-3, 3e-3]
# Coarse sensitivity heatmaps: (alpha, beta1) and (alpha, beta2). 3x3/4x4 only.
ALPHA_GRID = [3e-4, 1e-3, 3e-3]
BETA1_GRID = [0.0, 0.9, 0.99]
BETA2_GRID = [0.99, 0.999, 0.9999]
# ell: fixed validation-loss threshold for speed-to-target. Chosen from the
# baseline pilot on train/val ONLY (never test): the SGD-only baseline settles at
# validation loss ~0.69, so 0.70 is a modest improvement target that at least one
# optimizer reaches, exposing how quickly each optimizer gets there. Kept constant.
ELL_VAL_LOSS = 0.70

# ── Part 3: Regularization study (standard Adam, best Part-2 HPs, fixed) ──────
WEIGHT_DECAY_GRID   = [1e-5, 1e-4, 1e-3]
DROPOUT_GRID        = [0.1, 0.3, 0.5]
LABEL_SMOOTHING_GRID = [0.05, 0.1]
INPUT_NOISE_GRID    = [0.05, 0.1]     # std in scaled feature space, train-only.
EARLY_STOP_PATIENCE = 8               # monitored quantity = val loss; restore best.

# ── Part 4: Integrated best combination (extra credit) ────────────────────────
RO_FINETUNE_FRAC = 0.10          # <= 10% of Part-1 RO function-eval budget.
INTERACTION_TRIALS = 5           # <= 5 small interaction trials.
