#!/usr/bin/env bash
#SBATCH --job-name=s81-unsw512
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=short
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:TERM@120
set -euo pipefail
umask 077
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
: "${NCD_CODE_ROOT:?}" "${NCD_DATA_ROOT:?}" "${NCD_OUTPUT_ROOT:?}" "${NCD_PROTECTED_ROOT:?}"
: "${NCD_PYTHON:?}" "${NCD_SOURCE_COMMIT:?}"
test "${SLURM_CPUS_PER_TASK}" = 16
cd "$NCD_CODE_ROOT"
# Allocation uses sbatch --hint=nomultithread; srun retains the same policy.
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B ncd/hpc/section81_capacity_job.py \
  --dataset UNSW-NB15 --width 512 --data-root "$NCD_DATA_ROOT" \
  --output-dir "$NCD_OUTPUT_ROOT" --protected-dir "$NCD_PROTECTED_ROOT" --wall-seconds 2100
