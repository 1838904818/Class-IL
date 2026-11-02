#!/usr/bin/env bash
#SBATCH --job-name=cse-wacb-score
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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
printf '%s  %s\n' '736c77dc6a926a4c1591b1caf25b646c235e384237952dcbed960ae3e522296c' "$NCD_VALIDATION_CODE/diagnose-cse-wacb-score-distributions.py" | sha256sum -c -
mkdir "$NCD_OUTPUT_ROOT" "$NCD_PROTECTED_ROOT"
set +e
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u "$NCD_VALIDATION_CODE/diagnose-cse-wacb-score-distributions.py" \
  --cells "/scr/user/$USER/classil-b-v2-20261102/cells/cic-ids-2018/wacb" \
  --data "/scr/user/$USER/classil-a-v2-20261102/data/cic-ids-2018" \
  --output "$NCD_OUTPUT_ROOT/results" \
  > "$NCD_OUTPUT_ROOT/diagnosis.stdout.log" 2> "$NCD_OUTPUT_ROOT/diagnosis.stderr.log"
rc=$?
set -e
cp "$NCD_OUTPUT_ROOT/diagnosis.stdout.log" "$NCD_OUTPUT_ROOT/diagnosis.stderr.log" "$NCD_PROTECTED_ROOT/"
if [ -d "$NCD_OUTPUT_ROOT/results" ]; then
  cp -r "$NCD_OUTPUT_ROOT/results" "$NCD_PROTECTED_ROOT/"
fi
cat "$NCD_PROTECTED_ROOT/diagnosis.stdout.log"
cat "$NCD_PROTECTED_ROOT/diagnosis.stderr.log" >&2
printf 'CSE_WACB_SCORE_EXIT=%s\n' "$rc"
exit "$rc"
