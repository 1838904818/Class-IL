#!/usr/bin/env bash
#SBATCH --job-name=ncd-crr-s14-gate
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00

set -euo pipefail
umask 077
# DICC_CHECKPOINT: final gate rows are copied to protected storage.

: "${NCD_SOURCE_COMMIT:?NCD_SOURCE_COMMIT is required}"
: "${NCD_ARCHIVE:?NCD_ARCHIVE is required}"
: "${NCD_ARCHIVE_SHA256:?NCD_ARCHIVE_SHA256 is required}"
: "${NCD_SCRATCH_ROOT:?NCD_SCRATCH_ROOT is required}"
: "${NCD_PROTECTED_ROOT:?NCD_PROTECTED_ROOT is required}"

test "$(id -un)" = "$USER"
test "$HOME" = "/home/user/$USER"
test -f "$NCD_ARCHIVE" && test ! -L "$NCD_ARCHIVE"
printf '%s  %s\n' "$NCD_ARCHIVE_SHA256" "$NCD_ARCHIVE" | sha256sum -c --status

run_root="$NCD_SCRATCH_ROOT/gates-$SLURM_JOB_ID"
repo_root="$run_root/repo"
protected="$NCD_PROTECTED_ROOT"
mkdir -p "$repo_root" "$protected"
chmod 700 "$run_root" "$protected"

archive_evidence() {
  rc=$?
  mkdir -p "$protected/results" "$protected/logs"
  if [ -d "$repo_root/ncd/results/crr/section14" ]; then
    cp -a "$repo_root/ncd/results/crr/section14/." "$protected/results/"
  fi
  if [ -d "$repo_root/ncd/logs" ]; then cp -a "$repo_root/ncd/logs/." "$protected/logs/"; fi
  printf '{"job_id":"%s","source_commit":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$NCD_SOURCE_COMMIT" "$rc" > "$protected/JOB_EXIT.json"
  return "$rc"
}
trap archive_evidence EXIT

tar -xf "$NCD_ARCHIVE" -C "$repo_root"
test -f "$repo_root/ncd/run_crr_section14_gates.py"
test -f "$repo_root/ncd/crr_benchmarks.py"
test -f "$repo_root/src/__init__.py"
mkdir -p "$run_root/kaggle_leon_release/datasets"
for dataset in nsl-kdd unsw-nb15; do
  source_path="/scr/user/$USER/ofra-etg/authorized-raw-20260820/$dataset"
  test -d "$source_path"
  ln -s "$source_path" "$run_root/kaggle_leon_release/datasets/$dataset"
  test "$(realpath -e "$repo_root/../kaggle_leon_release/datasets/$dataset")" = "$source_path"
done

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export NCD_SOURCE_COMMIT
python_bin="$HOME/.venvs/ofra-etg/bin/python"
test -x "$python_bin"

printf 'CRR_S14_GATE_START id=%s source=%s host=%s utc=%s\n' \
  "$SLURM_JOB_ID" "$NCD_SOURCE_COMMIT" "$(hostname -s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi
cd "$repo_root"
"$python_bin" -I -c \
  'import sys; sys.path.insert(0, sys.argv[1]); import ncd.crr_benchmarks, src; print("IMPORT_SMOKE_OK")' \
  "$repo_root"

output=ncd/results/crr/section14/gates_seed1_2
mkdir -p "$output" ncd/logs
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$python_bin" -I ncd/run_crr_section14_gates.py \
  --config ncd/configs/crr_section12.json --device cuda --output-dir "$output" \
  2>&1 | tee ncd/logs/crr_section14_gates_seed1_2.log

result="$output/gate_result.json"
verdict="$output/GATE_VERDICT.json"
rows="$output/gate_rows.csv"
completed="$output/COMPLETED.json"
"$python_bin" -I -c \
  'import csv,json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); v=json.load(open(sys.argv[2],encoding="utf-8")); c=json.load(open(sys.argv[4],encoding="utf-8")); r=list(csv.DictReader(open(sys.argv[3],encoding="utf-8"))); assert x["git_commit"]==sys.argv[5]; assert c["git_commit"]==sys.argv[5]; assert len(x["rows"])==48; assert len(r)==48; assert v==x["verdict"]' \
  "$result" "$verdict" "$rows" "$completed" "$NCD_SOURCE_COMMIT"

mkdir -p "$protected/results" "$protected/logs" "$protected/configs"
cp -a "$output/." "$protected/results/"
cp -a ncd/logs/crr_section14_gates_seed1_2.log "$protected/logs/"
cp -a ncd/configs/crr_section12.json "$protected/configs/"
find "$protected/results" "$protected/logs" "$protected/configs" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$protected/SHA256SUMS.txt"
printf '{"job_id":"%s","source_commit":"%s","completed_utc":"%s","passed":%s}\n' \
  "$SLURM_JOB_ID" "$NCD_SOURCE_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$("$python_bin" -I -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' "$verdict")" \
  > "$protected/COMPLETED.json"
printf 'CRR_S14_GATE_COMPLETE id=%s protected=%s\n' "$SLURM_JOB_ID" "$protected"
trap - EXIT
