#!/usr/bin/env bash
#SBATCH --job-name=cse-bic14-resume
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:TERM@120
set -euo pipefail
umask 077
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
: "${NCD_PYTHON:?}" "${NCD_CODE_ROOT:?}" "${NCD_FROZEN_ROOT:?}" "${NCD_DATA_ROOT:?}"
: "${NCD_CACHE_ROOT:?}" "${NCD_OUTPUT_ROOT:?}" "${NCD_PROTECTED_ROOT:?}" "${NCD_SOURCE_COMMIT:?}"
: "${NCD_RESUME_ROOT:?}" "${NCD_PROGRESS_SHA256:?}" "${NCD_PORTABLE_ENTRY_SHA256:?}"
test -z "$(squeue -h -j 493533 -o '%i')"
attempt="$NCD_OUTPUT_ROOT"
mkdir "$attempt" "$NCD_PROTECTED_ROOT"
export TMPDIR="$NCD_CACHE_ROOT/tmp" TEMP="$NCD_CACHE_ROOT/tmp" TMP="$NCD_CACHE_ROOT/tmp"
progress="$NCD_RESUME_ROOT/bic/seed14/progress/backbone_000.pt"
printf '%s  %s\n' "$NCD_PROGRESS_SHA256" "$progress" | sha256sum -c -
printf '%s  %s\n' "$NCD_PORTABLE_ENTRY_SHA256" "$NCD_CODE_ROOT/ncd/hpc/cse_bic_portable.py" | sha256sum -c -
# The guard directory is new; actual scientific identity/path stays unchanged.
export NCD_OUTPUT_ROOT="$NCD_RESUME_ROOT"
"$NCD_PYTHON" -I -B -c 'import json,os,torch; from pathlib import Path; p=Path(os.environ["NCD_RESUME_ROOT"])/"bic/seed14/progress/backbone_000.pt"; s=torch.load(p,map_location="cpu",weights_only=True); assert s["identity"]["seed"]==14 and s["next_epoch"]>=5; assert s["steps"]==sum(x["optimizer_steps"] for x in s["history"]); print(json.dumps(dict(event="EPOCH_RESUME_BOUNDARY",next_epoch=s["next_epoch"],optimizer_steps=s["steps"],training_seconds=s["seconds"],model_and_optimizer_restored_by_original_entry=True)),flush=True)'
cd "$NCD_CODE_ROOT"
set +e
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u "$NCD_CODE_ROOT/ncd/hpc/cse_bic_portable.py" \
  --dataset CSE-CIC-IDS2018 --seed 14 --data-root "$NCD_DATA_ROOT" \
  --output-dir "$NCD_RESUME_ROOT/bic/seed14" --resume \
  --log-prefix "$NCD_RESUME_ROOT/logs/bic-14-attempt0" \
  > "$attempt/train.stdout.log" 2> "$attempt/train.stderr.log"
rc=$?
set -e
cp "$attempt/train.stdout.log" "$attempt/train.stderr.log" "$NCD_PROTECTED_ROOT/"
cat "$attempt/train.stdout.log"
cat "$attempt/train.stderr.log" >&2
if [ "$rc" = 0 ]; then
  test -f "$NCD_RESUME_ROOT/bic/seed14/summary.json"
  mkdir -p "$NCD_PROTECTED_ROOT/bic/seed14"
  find "$NCD_RESUME_ROOT/bic/seed14" -maxdepth 1 -type f -name '*.json' -exec cp -t "$NCD_PROTECTED_ROOT/bic/seed14" -- {} +
fi
printf 'BIC14_RESUME_EXIT=%s\n' "$rc"
exit "$rc"
