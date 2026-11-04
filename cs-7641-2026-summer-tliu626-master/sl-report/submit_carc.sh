#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_carc.sh  —  CS 7641 SL experiments on USC CARC (Discovery cluster)
#
# Submit:       sbatch submit_carc.sh
# Check status: squeue -u $USER
# View log:     tail -f sl_main_<jobid>.log
# Cancel:       scancel <jobid>
# ─────────────────────────────────────────────────────────────────────────────

# ── CPU partition (main) — no GPU needed for 20k-sample sklearn/PyTorch job ──
#SBATCH --account=desmith_128
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=sl7641
#SBATCH --output=sl_main_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tanxinli@usc.edu

# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "============================================================"
echo "  Job started : $(date)"
echo "  Node        : $(hostname)"
echo "  Working dir : $(pwd)"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "============================================================"

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load gcc/13.3.0
module load python/3.11.9

# ── Activate virtual environment ──────────────────────────────────────────────
VENV_DIR="/scratch1/${USER}/envs/cs7641-sl"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Virtual env not found — run setup_env_carc.sh first."
    exit 1
fi
source "${VENV_DIR}/bin/activate"
echo "Python: $(which python3)  $(python3 --version)"

# ── Navigate to project ────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR}"

# Force Python stdout to flush immediately so SLURM log is readable in real time
export PYTHONUNBUFFERED=1

# ── Parallelism hints ─────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ── PyTorch check (CPU-only is fine for 20k samples) ─────────────────────────
python3 -c "
import torch
print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')
" 2>/dev/null || echo "PyTorch check skipped"

# ── Step 0a: EDA ──────────────────────────────────────────────────────────────
echo ""
echo "[Step 0a] EDA ..."
python3 src/eda.py

# ── Step 0b: Hypothesis figures ───────────────────────────────────────────────
echo ""
echo "[Step 0b] Hypothesis ..."
python3 src/hypothesis.py

# ── Step 1-5: All models ──────────────────────────────────────────────────────
echo ""
echo "[Step 1-5] Running all models ..."
python3 run_all.py

echo ""
echo "============================================================"
echo "  Job finished : $(date)"
echo "  Figures  → $(pwd)/outputs/figures/"
echo "  Results  → $(pwd)/outputs/results/"
echo "============================================================"
