# CS7641 Unsupervised Learning & Dimensionality Reduction Report (Summer 2026)

Covertype unsupervised study: clustering, dimensionality reduction, and a
leakage-safe rerun of the SL/OL PyTorch neural network on reduced representations.

## Data continuity with SL/OL

The working set is the **same ~20,000-instance stratified Covertype sample** used
in the SL Report (proportional stratified, `n=20000`, `seed=42`). `src/data.py`
reuses `sl-report/outputs/sample.csv` when it holds ~20k rows; on this checkout
that file is a 2k smoke truncation, so the code **regenerates the identical 20k
sample** from full Covertype using SL's exact `train_test_split(..., test_size=
20000, stratify=Cover_Type, random_state=42)` call and caches it to
`outputs/sample.csv` (the SL file is never modified). Target `Cover_Type` (1..7)
is mapped to 0..6 and used **only after fitting** as an external label.

Preprocessing: StandardScaler on the 10 continuous cartographic features only;
the 44 binary wilderness/soil indicators stay 0/1 (`UL_SCALE_ALL=1` to override).

## Pipeline

| Stage      | Module              | What it does |
|------------|---------------------|--------------|
| `part1`    | `src/clustering.py` | K-Means + GMM on original features; internal (silhouette, DB, CH, inertia, BIC/AIC/loglik) + external (ARI, NMI, homogeneity/completeness/V) + contingency table |
| `part2`    | `src/dr.py`         | PCA (scree, cumulative var, reconstruction, effective rank, loadings), ICA (kurtosis, convergence, seed stability), RP (distance preservation across seeds, gaussian+sparse) |
| `part3`    | `src/dr_clustering.py` | K-Means & GMM rerun after PCA/ICA/RP (six combos), reselecting k/components per space |
| `part4`    | `src/nn_dr.py`, `src/model.py` | Leakage-safe NN: scaler+DR fit on **train only**; original vs PCA/ICA/RP; Accuracy/Macro-F1/balanced-acc/per-class + runtime, RP across seeds |
| `manifold` | `src/manifold.py`   | Extra credit: nonlinear manifold (t-SNE default) vs PCA, visualization only |

NN recipe (best justified from SL + OL): single 150-unit ReLU hidden layer,
class-weighted cross-entropy (SL), Adam + light weight decay (OL Part 2/3), early
stopping on val loss. Only the input dimension changes across representations.

## Run

```bash
# Fast smoke test (shrinks sweeps/seeds/epochs, uses the 2k sample):
UL_FAST=1 python run_ul.py all

# Full run on the cluster:
sbatch submit_ul.sh              # writes ul_main_<jobid>.log

# Individual stages:
python run_ul.py part1 part2
```

Outputs: `outputs/figures/*.png`, `outputs/results/*.csv|json`,
`outputs/results/repro.json` (environment + git commit + config for the REPRO PDF).

## Key environment variables

- `UL_FAST=1` — smoke config (2k sample, tiny sweeps).
- `UL_SEEDS=0,1,2` — seeds for stochastic methods (ICA/RP/GMM/K-Means/NN).
- `UL_SCALE_ALL=1` — scale all 54 features instead of continuous-only.
- `UL_MANIFOLD=tsne|isomap|lle|spectral|umap` — extra-credit method.
- `UL_DEVICE=cpu|cuda` — NN device.
