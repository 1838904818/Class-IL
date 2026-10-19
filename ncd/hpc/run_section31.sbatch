#!/usr/bin/env bash
#SBATCH --job-name=ncd-s31-cse18
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# The submitter supplies measured GPU/CPU/RAM/QoS/walltime requests after current
# account/policy checks, and keeps total live owned jobs within the two-GPU limit.
# No job arrays, fan-out, recursive submission, or owner absolute deadline.
set -euo pipefail
umask 077
: "${NCD_PYTHON:?Existing research Python executable}"
: "${NCD_CODE_ROOT:?Immutable HOME code package}"
: "${NCD_DATA_ROOT:?Existing full Section21 scratch cache}"
: "${NCD_OUTPUT_ROOT:?Shared independent Section31 scratch reference root}"
: "${NCD_PROTECTED_ROOT:?HOME small-evidence copies}"
: "${NCD_SOURCE_COMMIT:?Actual new adapter and wrapper commit}"
: "${NCD_METHODS:?Disjoint space-separated physical methods assigned to this job}"
[[ "$NCD_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "${NCD_LOCAL_SMOKE:-0}" =~ ^[01]$ ]]
[[ "${NCD_RESUME:-0}" =~ ^[01]$ ]]
cd "$NCD_CODE_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
read -r -a methods <<< "$NCD_METHODS"
read -r -a seeds <<< "${NCD_SEEDS:-1 2 3 4 42}"
args=(ncd/hpc/section31_job.py --code-root "$NCD_CODE_ROOT" --data-root "$NCD_DATA_ROOT"
  --output-root "$NCD_OUTPUT_ROOT" --protected-root "$NCD_PROTECTED_ROOT"
  --methods "${methods[@]}" --seeds "${seeds[@]}")
if [ "${NCD_RESUME:-0}" = 1 ]; then
  args+=(--resume)
fi
if [ "${NCD_LOCAL_SMOKE:-0}" = 1 ]; then
  exec "$NCD_PYTHON" -I -B -u "${args[@]}" --local-smoke --smoke-dataset "${NCD_SMOKE_DATASET:-CSE-CIC-IDS2018}"
fi
: "${SLURM_JOB_ID:?Formal execution requires Slurm}"
test "$SLURM_NNODES" = 1
test "${SLURM_NTASKS:-1}" = 1
test "$SLURM_CPUS_PER_TASK" = 2
test "$(id -un)" = "$USER"
srun --ntasks=1 --cpus-per-task=2 "$NCD_PYTHON" -I -B -u "${args[@]}"
