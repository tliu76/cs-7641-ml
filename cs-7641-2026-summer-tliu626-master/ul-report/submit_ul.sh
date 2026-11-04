#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_ul.sh — CS 7641 UL Report experiments on USC CARC (Discovery cluster)
#
# Submit:       sbatch submit_ul.sh
# Check status: squeue -u $USER
# View log:     tail -f ul_main_<jobid>.log
# Cancel:       scancel <jobid>
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=desmith_128
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=ul7641
#SBATCH --output=ul_main_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tanxinli@usc.edu

set -euo pipefail

echo "============================================================"
echo "  UL job started : $(date)"
echo "  Node           : $(hostname)"
echo "  Working dir     : $(pwd)"
echo "  Job ID          : ${SLURM_JOB_ID}"
echo "  CPUs            : ${SLURM_CPUS_PER_TASK}"
echo "============================================================"

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load gcc/13.3.0
module load python/3.11.9

# ── Activate the shared virtual environment (built for the SL report) ─────────
VENV_DIR="/scratch1/${USER}/envs/cs7641-sl"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Virtual env not found at ${VENV_DIR} — run sl-report/setup_env_carc.sh first."
    exit 1
fi
source "${VENV_DIR}/bin/activate"
echo "Python: $(which python3)  $(python3 --version)"

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1

# scipy is needed for ICA kurtosis; install into the env if missing (no-op if present).
python3 -c "import scipy" 2>/dev/null || pip install --quiet "scipy>=1.11"

# ── Threading: small models, cap threads to avoid dispatch overhead ───────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export UL_TORCH_THREADS=4

python3 -c "import torch,sklearn,scipy;print(f'torch {torch.__version__}  sklearn {sklearn.__version__}  scipy {scipy.__version__}')" \
    || echo "package check skipped"

# ── Run the full UL pipeline ──────────────────────────────────────────────────
echo ""
echo "[UL] Running full pipeline (part1 -> part2 -> part3 -> part4 -> manifold) ..."
python3 run_ul.py all

echo ""
echo "============================================================"
echo "  UL job finished : $(date)"
echo "  Figures  → $(pwd)/outputs/figures/"
echo "  Results  → $(pwd)/outputs/results/"
echo "============================================================"
