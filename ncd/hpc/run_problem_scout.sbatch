#!/usr/bin/env bash
#SBATCH --job-name=ncd-s21
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Resource flags MUST be supplied to sbatch from current account checks and
# measured workload requirements. No arrays, dependencies, or nested submissions.
set -euo pipefail
umask 077

: "${NCD_PYTHON:?Set the existing research Python executable}"
: "${NCD_CODE_ROOT:?Set the versioned code directory}"
: "${NCD_DATA_ROOT:?Set the prepared full Section21 cache directory}"
: "${NCD_OUTPUT_ROOT:?Set the result root, separate from the local serial run}"
: "${NCD_PROTECTED_ROOT:?Set the small-artifact backup root}"
: "${NCD_SOURCE_COMMIT:?Set the actual complete execution source commit}"
: "${NCD_DATASET:?Use nsl, unsw, cic17, or cic18}"
: "${NCD_SEED:?Use 1, 2, 3, 4, or 42}"

case "$NCD_DATASET" in nsl|unsw|cic17|cic18) ;; *) exit 64 ;; esac
case "$NCD_SEED" in 1|2|3|4|42) ;; *) exit 64 ;; esac
case "${NCD_METHOD_GROUP:-all}" in all) ;; joint|nonjoint) test "$NCD_DATASET" = cic18 ;; *) exit 64 ;; esac
[[ "$NCD_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "${NCD_LOCAL_SMOKE:-0}" =~ ^[01]$ ]] || exit 64

cd "$NCD_CODE_ROOT"
test -f ncd/run_problem_scout.py
test -f ncd/hpc/problem_scout_job.py
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1

args=(ncd/hpc/problem_scout_job.py --code-root "$NCD_CODE_ROOT"
  --data-root "$NCD_DATA_ROOT" --output-root "$NCD_OUTPUT_ROOT"
  --protected-root "$NCD_PROTECTED_ROOT" --dataset "$NCD_DATASET"
  --seed "$NCD_SEED" --method-group "${NCD_METHOD_GROUP:-all}")
if [ "${NCD_LOCAL_SMOKE:-0}" = 1 ]; then
  # This is the identical entry and training runner; --smoke makes the runner use
  # seed 1, one epoch, and at most 64 rows/class, including final summary writing.
  exec "$NCD_PYTHON" -I "${args[@]}" --local-smoke
fi

: "${SLURM_JOB_ID:?Formal execution requires a Slurm allocation}"
: "${SLURM_CPUS_PER_TASK:?Supply measured --cpus-per-task to sbatch}"
test "$SLURM_NNODES" = 1
test "${SLURM_NTASKS:-1}" = 1
test "$SLURM_CPUS_PER_TASK" -ge 2
test "$(id -un)" = "$USER"
# The Python entry independently checks resolved HOME/scratch containment.
exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$NCD_PYTHON" -I "${args[@]}"
