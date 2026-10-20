#!/usr/bin/env bash
#SBATCH --job-name=ncd-s44-cse
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Submitter checks current limits; each job owns one new seed, eight serial cells.
# No arrays, scheduler submissions, fan-out, tuning or automatic retry here.
set -euo pipefail
umask 077
: "${NCD_PYTHON:?Existing research Python}"
: "${NCD_CODE_ROOT:?New immutable Section44 HOME package}"
: "${NCD_DATA_ROOT:?Existing complete scratch cache}"
: "${NCD_OUTPUT_ROOT:?New Section44 scratch output}"
: "${NCD_PROTECTED_ROOT:?New Section44 HOME small-result copy}"
: "${NCD_SOURCE_COMMIT:?Published Section44 wrapper commit}"
: "${NCD_SEED:?One explicit new seed}"
[[ "$NCD_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$NCD_SEED" =~ ^[5-9]$ ]]
[[ "${NCD_RESUME:-0}" =~ ^[01]$ ]]
cd "$NCD_CODE_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
args=(ncd/hpc/section44_job.py --code-root "$NCD_CODE_ROOT"
  --data-root "$NCD_DATA_ROOT" --output-root "$NCD_OUTPUT_ROOT"
  --protected-root "$NCD_PROTECTED_ROOT" --seeds "$NCD_SEED"
  --methods aacil_nogkd_wa aacil_kd_wa aacil replay_bs replay derpp wacb freq)
if [ "${NCD_RESUME:-0}" = 1 ]; then
  args+=(--resume)
fi
: "${SLURM_JOB_ID:?Formal execution requires Slurm}"
test "$SLURM_NNODES" = 1
test "${SLURM_NTASKS:-1}" = 1
test "$SLURM_CPUS_PER_TASK" = 2
test "$(id -un)" = "$USER"
srun --ntasks=1 --cpus-per-task=2 "$NCD_PYTHON" -I -B -u "${args[@]}"
