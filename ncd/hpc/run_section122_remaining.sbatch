#!/usr/bin/env bash
#SBATCH --job-name=s122-capacity
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=normal
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
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
# Retain the preflight's measured 16-CPU cap: four bounded four-thread workers.
# Section122 fallback: 321.0853 s * 4 * ceil(60/4) * 1.3 = 25044.7 s.
# Round to seven hours including the supervisor's 120-second checkpoint margin.
# Memory: observed CIC1024 peak 1.853 GiB per child * 4 * 1.3 = 9.64 GiB.
# This is a measured extrapolation, not a completed-cell memory upper bound.
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B ncd/hpc/section81_capacity_job.py \
  --remaining --workers 4 --memory-limit-gib 9.5 --data-root "$NCD_DATA_ROOT" \
  --output-dir "$NCD_OUTPUT_ROOT" --protected-dir "$NCD_PROTECTED_ROOT" --wall-seconds 25200
