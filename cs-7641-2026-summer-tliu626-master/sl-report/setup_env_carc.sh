#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env_carc.sh  —  CS 7641 SL: USC CARC (Discovery) environment setup
#
# Run ONCE from the login node before submitting any jobs:
#   bash setup_env_carc.sh
#
# Then submit experiments with:
#   sbatch submit_carc.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VENV_DIR="/scratch1/${USER}/envs/cs7641-sl"
# Keep pip cache and temp on scratch — home quota is too small for PyTorch wheels
export PIP_CACHE_DIR="/scratch1/${USER}/.pip_cache"
export TMPDIR="/scratch1/${USER}/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

echo "============================================================"
echo "  CS 7641 SL — Environment Setup on USC CARC (Discovery)"
echo "  Venv path : ${VENV_DIR}"
echo "============================================================"

# ── 1. Load modules ───────────────────────────────────────────────────────────
module purge
module load gcc/13.3.0
module load python/3.11.9

echo "[1/4] Python: $(python3 --version)"

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ -d "${VENV_DIR}" ]; then
    echo "[2/4] Venv already exists at ${VENV_DIR} — skipping creation."
else
    echo "[2/4] Creating virtual environment ..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet --no-cache-dir

# ── 3. Install packages ───────────────────────────────────────────────────────
echo "[3/4] Installing packages ..."

pip install --quiet --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    scipy

# PyTorch — CUDA 12.4 build for A100 GPU nodes on CARC Discovery
# CARC has cuda/12.4.0 and cuda/12.6.3; cu124 wheel works for both.
# Switch to CPU-only if you want a faster/smaller install:
#   pip install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Verify
echo "[3/4] Verifying installs ..."
python3 -c "
import sklearn, torch, numpy, pandas, matplotlib
print(f'  scikit-learn : {sklearn.__version__}')
print(f'  torch        : {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'  numpy        : {numpy.__version__}')
print(f'  pandas       : {pandas.__version__}')
print(f'  matplotlib   : {matplotlib.__version__}')
"

# ── 4. Verify data file ────────────────────────────────────────────────────────
echo "[4/4] Checking dataset ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${SCRIPT_DIR}/data/covtype.data"

if [ -f "${DATA}" ]; then
    echo "  Dataset found: ${DATA}"
    echo "  Rows: $(wc -l < "${DATA}")"
else
    echo "  WARNING: ${DATA} not found."
    echo "  Upload it from your local machine with:"
    echo "    scp /path/to/covtype.data tanxinli@discovery.usc.edu:/scratch1/tanxinli/7641/cs-7641-2026-summer-tliu626/sl-report/data/"
fi

echo ""
echo "============================================================"
echo "  Setup complete."
echo "  Submit your job with:  sbatch submit_carc.sh"
echo "============================================================"
