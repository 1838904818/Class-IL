#!/usr/bin/env bash
#SBATCH --job-name=ncd-aacil
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Resource requests are supplied after live account checks and measured sizing.
set -euo pipefail
umask 077
: "${NCD_PYTHON:?Existing research Python executable}"
: "${NCD_CODE_ROOT:?Versioned code directory}"
: "${NCD_DATA_ROOT:?Existing prepared full cache}"
: "${NCD_OUTPUT_ROOT:?Independent active results}"
: "${NCD_PROTECTED_ROOT:?HOME small-evidence copies}"
: "${NCD_SOURCE_COMMIT:?Actual wrapper commit}"
: "${NCD_TRAINING_SOURCE_COMMIT:?Frozen AA-CIL training commit}"
: "${NCD_DATASET:?nsl, unsw or cic17}"
: "${NCD_METHODS:?Space-separated allocated methods}"
: "${NCD_SEEDS:?Space-separated allocated seeds}"
[[ "$NCD_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test "$NCD_TRAINING_SOURCE_COMMIT" = 414136b418cbb5f682875309b00e4f8151240bfe
[[ "${NCD_LOCAL_SMOKE:-0}" =~ ^[01]$ ]]
[[ "${NCD_NO_OWNER_DEADLINE:-0}" =~ ^[01]$ ]]
if [ "${NCD_NO_OWNER_DEADLINE:-0}" = 1 ]; then
  if [ -n "${NCD_DEADLINE_UTC:-}" ]; then
    printf '%s\n' 'NCD_NO_OWNER_DEADLINE and NCD_DEADLINE_UTC are mutually exclusive' >&2
    exit 2
  fi
  deadline_args=(--no-owner-deadline)
else
  : "${NCD_DEADLINE_UTC:?Owner absolute hard deadline or NCD_NO_OWNER_DEADLINE=1 required}"
  deadline_args=(--deadline-utc "$NCD_DEADLINE_UTC")
fi
cd "$NCD_CODE_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
read -r -a methods <<< "$NCD_METHODS"
read -r -a seeds <<< "$NCD_SEEDS"
args=(ncd/hpc/aacil_job.py --code-root "$NCD_CODE_ROOT" --data-root "$NCD_DATA_ROOT"
  --output-root "$NCD_OUTPUT_ROOT" --protected-root "$NCD_PROTECTED_ROOT"
  --dataset "$NCD_DATASET" --methods "${methods[@]}" --seeds "${seeds[@]}"
  "${deadline_args[@]}")
if [ "${NCD_LOCAL_SMOKE:-0}" = 1 ]; then
  exec "$NCD_PYTHON" -I "${args[@]}" --local-smoke
fi
: "${SLURM_JOB_ID:?Formal execution requires Slurm}"
test "$SLURM_NNODES" = 1
test "${SLURM_NTASKS:-1}" = 1
test "$SLURM_CPUS_PER_TASK" = 2
test "$(id -un)" = "$USER"
exec srun --ntasks=1 --cpus-per-task=2 "$NCD_PYTHON" -I "${args[@]}"
