#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_stability.sh — Part 3 add-on: clustering stability per representation.
# Submit: sbatch submit_stability.sh    Log: ul_stab_<jobid>.log
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=desmith_128
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=ul_stab
#SBATCH --output=ul_stab_%j.log

set -euo pipefail

echo "============================================================"
echo "  Stability job started : $(date)"
echo "  Node                  : $(hostname)"
echo "  Job ID                : ${SLURM_JOB_ID}"
echo "============================================================"

module purge
module load gcc/13.3.0
module load python/3.11.9
source "/scratch1/${USER}/envs/cs7641-sl/bin/activate"
echo "Python: $(which python3)  $(python3 --version)"

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch1/${USER}/tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python3 run_stability.py

echo "============================================================"
echo "  Stability job finished : $(date)"
echo "============================================================"
