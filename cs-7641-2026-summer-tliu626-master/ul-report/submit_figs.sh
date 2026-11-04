#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_figs.sh — regenerate every report figure (no experiment is re-run;
# only the fitted-transform figures refit PCA/ICA/RP + K-Means/GMM at the
# settings the saved results already selected).
#
# Submit:  sbatch submit_figs.sh [figure-name ...]
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=desmith_128
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=ulfigs
#SBATCH --output=ul_figs_%j.log

set -euo pipefail

module purge
module load gcc/13.3.0
module load python/3.11.9

VENV_DIR="/scratch1/${USER}/envs/cs7641-sl"
source "${VENV_DIR}/bin/activate"

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch1/${USER}/tmp"
mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "[figs] $(date)  node=$(hostname)"
python3 make_figures.py "$@"
echo "[figs] done $(date)"
