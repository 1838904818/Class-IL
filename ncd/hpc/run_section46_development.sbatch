#!/usr/bin/env bash
#SBATCH --job-name=ncd-s46-dev
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
umask 077
: "${NCD_PYTHON:?Existing research Python}"
: "${NCD_CODE_ROOT:?New immutable development code package}"
: "${NCD_DATA_ROOT:?Complete existing scratch data}"
: "${NCD_OUTPUT_ROOT:?New single-dataset output}"
: "${NCD_PROTECTED_ROOT:?New HOME small-result copy}"
: "${NCD_SOURCE_COMMIT:?Published adapter commit}"
: "${NCD_DEVELOPMENT_DATASET:?NSL-KDD or CIC-IDS-2017}"
[[ "$NCD_DEVELOPMENT_DATASET" = NSL-KDD || "$NCD_DEVELOPMENT_DATASET" = CIC-IDS-2017 ]]
[[ "${NCD_RESUME:-0}" =~ ^[01]$ ]]
cd "$NCD_CODE_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
args=(ncd/hpc/section46_development_job.py --code-root "$NCD_CODE_ROOT"
  --data-root "$NCD_DATA_ROOT" --output-root "$NCD_OUTPUT_ROOT"
  --protected-root "$NCD_PROTECTED_ROOT" --methods aacil_nogkd_wa --seeds 1 2 3 4 42)
if [ "${NCD_RESUME:-0}" = 1 ]; then args+=(--resume); fi
: "${SLURM_JOB_ID:?Requires Slurm allocation}"
test "$SLURM_NNODES" = 1
test "${SLURM_NTASKS:-1}" = 1
test "$SLURM_CPUS_PER_TASK" = 2
srun --ntasks=1 --cpus-per-task=2 "$NCD_PYTHON" -I -B -u "${args[@]}"
