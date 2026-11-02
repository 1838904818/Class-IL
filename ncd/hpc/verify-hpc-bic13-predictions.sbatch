#!/usr/bin/env bash
#SBATCH --job-name=bic13-prediction-check
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
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
mkdir "$NCD_OUTPUT_ROOT" "$NCD_PROTECTED_ROOT"
sha256sum "$NCD_VALIDATION_CODE/check-hpc-cse-bic-predictions.py"
set +e
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u "$NCD_VALIDATION_CODE/check-hpc-cse-bic-predictions.py" \
  --directory "/scr/user/$USER/classil-a-v2-20261102/cells/bic/seed13" \
  --data "/scr/user/$USER/classil-a-v2-20261102/data/cic-ids-2018" \
  > "$NCD_OUTPUT_ROOT/check.stdout.log" 2> "$NCD_OUTPUT_ROOT/check.stderr.log"
rc=$?
set -e
cp "$NCD_OUTPUT_ROOT/check.stdout.log" "$NCD_OUTPUT_ROOT/check.stderr.log" "$NCD_PROTECTED_ROOT/"
cat "$NCD_PROTECTED_ROOT/check.stdout.log"
cat "$NCD_PROTECTED_ROOT/check.stderr.log" >&2
printf 'PREDICTION_CHECK_EXIT=%s\n' "$rc"
exit "$rc"
