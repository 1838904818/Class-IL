#!/usr/bin/env bash
#SBATCH --job-name=ncd-s32-kdwa
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Caller supplies measured resources/current QoS after live own-account checks.
# One GPU child at a time; no submission, arrays or background fan-out here.
set -euo pipefail
umask 077
: "${NCD_PYTHON:?Existing research Python executable}"
: "${NCD_CODE_ROOT:?New immutable HOME selected-main code package}"
: "${NCD_DATA_ROOT:?Existing full Section21 scratch cache}"
: "${NCD_OUTPUT_ROOT:?Independent selected_main_v1 scratch output}"
: "${NCD_PROTECTED_ROOT:?Independent HOME small-evidence copies}"
: "${NCD_SOURCE_COMMIT:?New wrapper commit, not the frozen training commit}"
: "${NCD_SELECTION_FILE:?Exact immutable development selection JSON}"
[[ "$NCD_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "${NCD_LOCAL_SMOKE:-0}" =~ ^[01]$ ]]
[[ "${NCD_RESUME:-0}" =~ ^[01]$ ]]
cd "$NCD_CODE_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
read -r -a seeds <<< "${NCD_SEEDS:-1 2 3 4 42}"
args=(ncd/hpc/section32_job.py --code-root "$NCD_CODE_ROOT" --data-root "$NCD_DATA_ROOT"
  --output-root "$NCD_OUTPUT_ROOT" --protected-root "$NCD_PROTECTED_ROOT"
  --methods aacil_kd_wa --seeds "${seeds[@]}")
if [ "${NCD_RESUME:-0}" = 1 ]; then
  args+=(--resume)
fi
if [ "${NCD_LOCAL_SMOKE:-0}" = 1 ]; then
  export CUDA_VISIBLE_DEVICES=-1
  exec "$NCD_PYTHON" -I -B -u "${args[@]}" --local-smoke --smoke-dataset "${NCD_SMOKE_DATASET:-NSL-KDD}"
fi
: "${SLURM_JOB_ID:?Formal execution requires Slurm}"
test "$SLURM_NNODES" = 1
test "${SLURM_NTASKS:-1}" = 1
test "$SLURM_CPUS_PER_TASK" = 2
test "$(id -un)" = "$USER"
srun --ntasks=1 --cpus-per-task=2 "$NCD_PYTHON" -I -B -u "${args[@]}"
