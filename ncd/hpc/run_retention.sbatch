#!/usr/bin/env bash
#SBATCH --job-name=ncd-retention
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# GPU/CPU/RAM/QoS/walltime requests are supplied only after live policy and
# measured-resource checks. This entry never submits another job.
set -euo pipefail
umask 077
: "${NCD_PYTHON:?Existing research Python executable}"
: "${NCD_CODE_ROOT:?Versioned HOME code directory}"
: "${NCD_DATA_ROOT:?Existing prepared full scratch cache}"
: "${NCD_OUTPUT_ROOT:?Independent active scratch results}"
: "${NCD_PROTECTED_ROOT:?HOME small-evidence copies}"
: "${NCD_SOURCE_COMMIT:?Actual wrapper commit}"
: "${NCD_TRAINING_SOURCE_COMMIT:?Actual frozen retention training commit}"
: "${NCD_DEADLINE_UTC:?Owner hard deadline no later than 19Oct21:50:26 MYT}"
: "${NCD_METHODS:?Space-separated assigned methods}"
[[ "$NCD_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$NCD_TRAINING_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "${NCD_LOCAL_SMOKE:-0}" =~ ^[01]$ ]]
cd "$NCD_CODE_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
read -r -a methods <<< "$NCD_METHODS"
args=(ncd/hpc/retention_job.py --code-root "$NCD_CODE_ROOT" --data-root "$NCD_DATA_ROOT"
  --output-root "$NCD_OUTPUT_ROOT" --protected-root "$NCD_PROTECTED_ROOT"
  --dataset cic17 --methods "${methods[@]}" --training-source-commit "$NCD_TRAINING_SOURCE_COMMIT"
  --deadline-utc "$NCD_DEADLINE_UTC")
if [ -n "${NCD_SELECTION:-}" ]; then
  args+=(--selection "$NCD_SELECTION")
fi
if [ -n "${NCD_SELECTION_SHA256:-}" ]; then
  args+=(--selection-sha256 "$NCD_SELECTION_SHA256")
fi
if [ "${NCD_LOCAL_SMOKE:-0}" = 1 ]; then
  exec "$NCD_PYTHON" -I "${args[@]}" --local-smoke
fi
: "${NCD_SELECTION:?Immutable completed Section29.8 selection.json}"
: "${NCD_SELECTION_SHA256:?SHA256 of the immutable selection.json}"
: "${SLURM_JOB_ID:?Formal execution requires Slurm}"
test "$SLURM_NNODES" = 1
test "${SLURM_NTASKS:-1}" = 1
test "$SLURM_CPUS_PER_TASK" = 2
test "$(id -un)" = "$USER"
exec srun --ntasks=1 --cpus-per-task=2 "$NCD_PYTHON" -I "${args[@]}"
