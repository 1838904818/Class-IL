#!/usr/bin/env bash
#SBATCH --job-name=s178-same-score
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --dependency=afterok:493652
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
: "${NCD_INPUT_ARCHIVE:?}" "${NCD_INPUT_ARCHIVE_SHA256:?}" "${NCD_SMALL_DATA_ARCHIVE:?}" "${NCD_SMALL_DATA_ARCHIVE_SHA256:?}"
: "${NCD_SAME_SCORE_SOURCE_SHA256:?}"
: "${NCD_BIC_VERIFIER_SHA256:?}"
# All extraction and dataset/model computation is inside this allocation.
printf '%s  %s\n' "$NCD_SAME_SCORE_SOURCE_SHA256" "$NCD_VALIDATION_CODE/check-section178-same-score.py" | sha256sum -c -
printf '%s  %s\n' "$NCD_BIC_VERIFIER_SHA256" "$NCD_VALIDATION_CODE/check-hpc-cse-bic-predictions.py" | sha256sum -c -
printf '%s  %s\n' "$NCD_INPUT_ARCHIVE_SHA256" "$NCD_INPUT_ARCHIVE" | sha256sum -c -
printf '%s  %s\n' "$NCD_SMALL_DATA_ARCHIVE_SHA256" "$NCD_SMALL_DATA_ARCHIVE" | sha256sum -c -
bic14="/scr/user/$USER/classil-a-v2-20261102/cells/bic/seed14"
# Do not reserve an allocation to wait for an unfinished training cell.
test -f "$bic14/result.json"
mkdir "$NCD_OUTPUT_ROOT" "$NCD_PROTECTED_ROOT"
mkdir "$NCD_OUTPUT_ROOT/inputs" "$NCD_OUTPUT_ROOT/data"
tar -xf "$NCD_INPUT_ARCHIVE" -C "$NCD_OUTPUT_ROOT/inputs"
tar -xf "$NCD_SMALL_DATA_ARCHIVE" -C "$NCD_OUTPUT_ROOT/data"
ln -s "/scr/user/$USER/classil-a-v2-20261102/data/cic-ids-2018" "$NCD_OUTPUT_ROOT/data/cic-ids-2018"
ln -s "$bic14" "$NCD_OUTPUT_ROOT/inputs/cic-ids-2018/cer-bic-common/seed14"
set +e
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u "$NCD_VALIDATION_CODE/check-hpc-cse-bic-predictions.py" \
  --directory "$bic14" --data "$NCD_OUTPUT_ROOT/data/cic-ids-2018" \
  > "$NCD_OUTPUT_ROOT/bic14-original.stdout.log" 2> "$NCD_OUTPUT_ROOT/bic14-original.stderr.log"
bic_rc=$?
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u "$NCD_VALIDATION_CODE/check-section178-same-score.py" \
  --inputs "$NCD_OUTPUT_ROOT/inputs" --data "$NCD_OUTPUT_ROOT/data" --output "$NCD_OUTPUT_ROOT/results" \
  > "$NCD_OUTPUT_ROOT/diagnosis.stdout.log" 2> "$NCD_OUTPUT_ROOT/diagnosis.stderr.log"
rc=$?
set -e
cp "$NCD_OUTPUT_ROOT/diagnosis.stdout.log" "$NCD_OUTPUT_ROOT/diagnosis.stderr.log" "$NCD_PROTECTED_ROOT/"
cp "$NCD_OUTPUT_ROOT/bic14-original.stdout.log" "$NCD_OUTPUT_ROOT/bic14-original.stderr.log" "$NCD_PROTECTED_ROOT/"
if [ -d "$NCD_OUTPUT_ROOT/results" ]; then cp -r "$NCD_OUTPUT_ROOT/results" "$NCD_PROTECTED_ROOT/"; fi
cat "$NCD_PROTECTED_ROOT/diagnosis.stdout.log"
cat "$NCD_PROTECTED_ROOT/diagnosis.stderr.log" >&2
printf 'BIC14_ORIGINAL_EXIT=%s SAME_SCORE_EXIT=%s\n' "$bic_rc" "$rc"
if [ "$bic_rc" != 0 ]; then exit "$bic_rc"; fi
exit "$rc"
