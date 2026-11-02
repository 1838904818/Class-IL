#!/usr/bin/env bash
#SBATCH --job-name=cse-original-runtime
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
: "${NCD_PYTHON:?}" "${NCD_OUTPUT_ROOT:?}" "${NCD_PROTECTED_ROOT:?}"
: "${NCD_RUNTIME_SOURCE_SHA256:?}" "${NCD_DEFAULT_REFERENCE_SHA256:?}"
root="$HOME/classil-20261103/cse-original-runtime"
printf '%s  %s\n' "$NCD_RUNTIME_SOURCE_SHA256" "$root/reevaluate-cse-original-runtime.py" | sha256sum -c -
printf '%s  %s\n' "$NCD_DEFAULT_REFERENCE_SHA256" "$root/icarl10-default-stages.json" | sha256sum -c -
mkdir "$NCD_OUTPUT_ROOT" "$NCD_PROTECTED_ROOT"
protect() {
  rc=$?
  trap - EXIT
  if [ -f "$NCD_OUTPUT_ROOT/diagnosis.stdout.log" ]; then cp "$NCD_OUTPUT_ROOT/diagnosis.stdout.log" "$NCD_PROTECTED_ROOT/"; fi
  if [ -f "$NCD_OUTPUT_ROOT/diagnosis.stderr.log" ]; then cp "$NCD_OUTPUT_ROOT/diagnosis.stderr.log" "$NCD_PROTECTED_ROOT/"; fi
  if [ -d "$NCD_OUTPUT_ROOT/results" ]; then cp -r "$NCD_OUTPUT_ROOT/results" "$NCD_PROTECTED_ROOT/"; fi
  exit "$rc"
}
trap protect EXIT
old="/scr/user/$USER/classil-same-score-validation-20261102/inputs/cic-ids-2018"
inputs="$NCD_OUTPUT_ROOT/inputs/cic-ids-2018"
mkdir -p "$inputs/icarl/seed10" "$inputs/cer-bic-common"
for name in result.json calibrated_stages.json checkpoints; do
  ln -s "$old/icarl/seed10/$name" "$inputs/icarl/seed10/$name"
done
ln -s "$root/icarl10-default-stages.json" "$inputs/icarl/seed10/default_stages.json"
for seed in 10 11 12; do ln -s "$old/cer-bic-common/seed$seed" "$inputs/cer-bic-common/seed$seed"; done
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u "$root/reevaluate-cse-original-runtime.py" \
  --inputs "$NCD_OUTPUT_ROOT/inputs" \
  --data "/scr/user/$USER/classil-a-v2-20261102/data/cic-ids-2018" \
  --output "$NCD_OUTPUT_ROOT/results" \
  --forward-source "$HOME/classil-20261102/same-score-validation-20261102/check-section178-same-score.py" \
  > "$NCD_OUTPUT_ROOT/diagnosis.stdout.log" 2> "$NCD_OUTPUT_ROOT/diagnosis.stderr.log"
