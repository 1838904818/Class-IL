#!/usr/bin/env bash
#SBATCH --job-name=cse-wacb-startup-trace
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=short
#SBATCH --time=00:05:00
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
printf '%s  %s\n' '3ade9812504ade534df25597e6dbedab6856b66a0d82cf3be51585c4f5ffb09f' "$NCD_VALIDATION_CODE/trace-cse-wacb-startup.py" | sha256sum -c -
mkdir "$NCD_OUTPUT_ROOT" "$NCD_PROTECTED_ROOT"
set +e
srun --hint=nomultithread --cpu-bind=cores timeout --signal=TERM --kill-after=4s 95s \
  "$NCD_PYTHON" -I -B -u -X importtime "$NCD_VALIDATION_CODE/trace-cse-wacb-startup.py" \
  --target "$HOME/classil-20261102/cse-wacb-score-20261102/diagnose-cse-wacb-score-distributions.py" -- \
  --cells "/scr/user/$USER/classil-b-v2-20261102/cells/cic-ids-2018/wacb" \
  --data "/scr/user/$USER/classil-a-v2-20261102/data/cic-ids-2018" \
  --output "$NCD_OUTPUT_ROOT/startup-artifacts" --seeds 10 \
  > "$NCD_OUTPUT_ROOT/trace.stdout.log" 2> "$NCD_OUTPUT_ROOT/trace.stderr.log"
rc=$?
set -e
cp "$NCD_OUTPUT_ROOT/trace.stdout.log" "$NCD_OUTPUT_ROOT/trace.stderr.log" "$NCD_PROTECTED_ROOT/"
cat "$NCD_PROTECTED_ROOT/trace.stdout.log"
cat "$NCD_PROTECTED_ROOT/trace.stderr.log" >&2
printf 'STARTUP_TRACE_EXIT=%s_NO_MODEL_EVALUATION\n' "$rc"
exit "$rc"
