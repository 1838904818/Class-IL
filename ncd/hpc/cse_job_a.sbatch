#!/usr/bin/env bash
#SBATCH --job-name=cse-s1-recovery
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:TERM@120
set -euo pipefail
umask 077
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
: "${NCD_CODE_ROOT:?}" "${NCD_FROZEN_ROOT:?}" "${NCD_DATA_ROOT:?}"
: "${NCD_OUTPUT_ROOT:?}" "${NCD_CACHE_ROOT:?}" "${NCD_PYTHON:?}" "${NCD_SOURCE_COMMIT:?}"
export TMPDIR="$NCD_CACHE_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
cd "$NCD_CODE_ROOT"
# Six pending cells, two CPU threads each; local observed per-cell peaks 6-9 GiB.
# Completed BiC11/12 and iCaRL10 never enter the training queue.
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u ncd/hpc/cse_job_a.py
