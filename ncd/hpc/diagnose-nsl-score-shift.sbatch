#!/usr/bin/env bash
#SBATCH --job-name=nsl-score-shift
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
umask 077
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
: "${NCD_PYTHON:?}" "${NCD_VALIDATION_CODE:?}" "${NCD_OUTPUT_ROOT:?}" "${NCD_PROTECTED_ROOT:?}"
: "${NCD_FROZEN_ROOT:?}" "${NCD_SKLEARN_ROOT:?}"
mkdir "$NCD_OUTPUT_ROOT" "$NCD_PROTECTED_ROOT"
sha256sum "$NCD_VALIDATION_CODE/diagnose-nsl-score-shift.py"
set +e
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u "$NCD_VALIDATION_CODE/diagnose-nsl-score-shift.py" \
  --source "$NCD_FROZEN_ROOT" \
  --data "/scr/user/$USER/classil-b-v2-20261102/data/nsl-kdd" \
  --stateless-root "/scr/user/$USER/classil-b-v2-20261102/cells/confirmation/nsl-kdd/R3-A" \
  --derpp-root "/scr/user/$USER/classil-b-v2-20261102/cells/nsl-kdd/derpp" \
  --output "$NCD_OUTPUT_ROOT/results" --require-exact \
  > "$NCD_OUTPUT_ROOT/diagnosis.stdout.log" 2> "$NCD_OUTPUT_ROOT/diagnosis.stderr.log"
rc=$?
set -e
cp "$NCD_OUTPUT_ROOT/diagnosis.stdout.log" "$NCD_OUTPUT_ROOT/diagnosis.stderr.log" "$NCD_PROTECTED_ROOT/"
if [ -d "$NCD_OUTPUT_ROOT/results" ]; then
  cp -r "$NCD_OUTPUT_ROOT/results" "$NCD_PROTECTED_ROOT/"
fi
cat "$NCD_PROTECTED_ROOT/diagnosis.stdout.log"
cat "$NCD_PROTECTED_ROOT/diagnosis.stderr.log" >&2
printf 'NSL_SCORE_DIAGNOSTIC_EXIT=%s\n' "$rc"
exit "$rc"
