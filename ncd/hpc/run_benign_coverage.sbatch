#!/usr/bin/env bash
#SBATCH --job-name=ncd-s25-d1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Supply measured resources and current-account-compatible QoS on sbatch's CLI.
# Exactly one GPU, one task, and serial CIC17 D1/D2 seeds; no nested submission.
set -euo pipefail
umask 077
: "${NCD_PYTHON:?Set the existing research Python executable}"
: "${NCD_CODE_ROOT:?Set the committed versioned code directory}"
: "${NCD_DATA_ROOT:?Set the existing full Section21 clean cache directory}"
: "${NCD_OUTPUT_ROOT:?Set the independent HPC active-result root}"
: "${NCD_PROTECTED_ROOT:?Set the HOME small-evidence copy root}"
: "${NCD_SOURCE_COMMIT:?Set the complete wrapper execution commit}"
[[ "$NCD_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "${NCD_LOCAL_SMOKE:-0}" =~ ^[01]$ ]] || exit 64
NCD_TREATMENT="${NCD_TREATMENT:-D1}"
case "$NCD_TREATMENT" in
  D1)
    test -z "${NCD_D2_BUDGET_MODE:-}" || exit 64
    # Original D1 callers retain their original child source, including 5a5a882.
    NCD_TRAINING_SOURCE_COMMIT="${NCD_TRAINING_SOURCE_COMMIT:-$NCD_SOURCE_COMMIT}"
    ;;
  D2)
    : "${NCD_D2_BUDGET_MODE:?D2 requires explicit fixed_final}"
    : "${NCD_TRAINING_SOURCE_COMMIT:?D2 requires the frozen training commit}"
    test "$NCD_D2_BUDGET_MODE" = fixed_final || exit 64
    test "$NCD_TRAINING_SOURCE_COMMIT" = 5657b468d0c83af58ec60dcd60ecf83bf61594f5 || exit 64
    ;;
  *) exit 64 ;;
esac
[[ "$NCD_TRAINING_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 64
cd "$NCD_CODE_ROOT"
test -f ncd/run_benign_coverage.py
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
args=(ncd/hpc/benign_coverage_job.py --code-root "$NCD_CODE_ROOT"
  --data-root "$NCD_DATA_ROOT" --output-root "$NCD_OUTPUT_ROOT"
  --protected-root "$NCD_PROTECTED_ROOT" --treatment "$NCD_TREATMENT"
  --training-source-commit "$NCD_TRAINING_SOURCE_COMMIT")
if [ "$NCD_TREATMENT" = D2 ]; then
  args+=(--d2-budget-mode "$NCD_D2_BUDGET_MODE")
fi
if [ "${NCD_LOCAL_SMOKE:-0}" = 1 ]; then
  # Identical entry/runner; one seed, one epoch, <=64/class and final summaries.
  exec "$NCD_PYTHON" -I "${args[@]}" --local-smoke
fi
: "${SLURM_JOB_ID:?Formal execution requires a Slurm allocation}"
: "${SLURM_CPUS_PER_TASK:?Supply measured CPU allocation}"
test "$SLURM_NNODES" = 1
test "${SLURM_NTASKS:-1}" = 1
test "$SLURM_CPUS_PER_TASK" = 2
test "$(id -un)" = "$USER"
exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$NCD_PYTHON" -I "${args[@]}"
