#!/bin/bash
# Part 4b sweep: NN Macro-F1 vs component count, fixed-epoch recipe, on GPU.
# Submit: sbatch submit_sweep.sh     Progress: squeue -u $USER   Log: ul_sw_<jobid>.log

#SBATCH --account=desmith_128
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=ul_sw
#SBATCH --output=ul_sw_%j.log

set -euo pipefail
echo "============================================================"
echo "  Sweep job started : $(date)"
echo "  Node              : $(hostname)"
echo "  Job ID            : ${SLURM_JOB_ID}"
echo "============================================================"

module purge
module load gcc/13.3.0
module load python/3.11.9
source "/scratch1/${USER}/envs/cs7641-sl/bin/activate"

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch1/${USER}/tmp"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export UL_DEVICE=cpu
export UL_TORCH_THREADS=4

python3 run_sweep.py

echo "  Sweep job finished : $(date)"
