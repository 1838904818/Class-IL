#!/usr/bin/env bash
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00

set -euo pipefail
umask 077

: "${NCD_DATASET:?NCD_DATASET must be nsl or unsw}"
: "${NCD_SOURCE_COMMIT:?NCD_SOURCE_COMMIT is required}"
: "${NCD_ARCHIVE:?NCD_ARCHIVE is required}"
: "${NCD_ARCHIVE_SHA256:?NCD_ARCHIVE_SHA256 is required}"
: "${NCD_OPERATION_ROOT:?NCD_OPERATION_ROOT is required}"
: "${NCD_SCRATCH_ROOT:?NCD_SCRATCH_ROOT is required}"
: "${NCD_PROTECTED_ROOT:?NCD_PROTECTED_ROOT is required}"

case "$NCD_DATASET" in nsl) dataset_slug=nsl-kdd ;; unsw) dataset_slug=unsw-nb15 ;; *) exit 64 ;; esac
test "$(id -un)" = leon12138
test "$HOME" = /home/user/leon12138
test -f "$NCD_ARCHIVE" && test ! -L "$NCD_ARCHIVE"
printf '%s  %s\n' "$NCD_ARCHIVE_SHA256" "$NCD_ARCHIVE" | sha256sum -c --status

run_root="$NCD_SCRATCH_ROOT/$NCD_DATASET"
repo_root="$run_root/repo"
protected="$NCD_PROTECTED_ROOT/$NCD_DATASET"
mkdir -p "$run_root" "$protected"
chmod 700 "$run_root" "$protected"

archive_failure_evidence() {
  rc=$?
  mkdir -p "$protected/logs"
  if [ -d "$repo_root/ncd/logs" ]; then cp -a "$repo_root/ncd/logs/." "$protected/logs/"; fi
  if [ -f "$repo_root/ncd/results/exp3/${NCD_DATASET}_phase1_hpc_state.json" ]; then
    cp -a "$repo_root/ncd/results/exp3/${NCD_DATASET}_phase1_hpc_state.json" "$protected/"
  fi
  printf '{"job_id":"%s","dataset":"%s","source_commit":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$rc" > "$protected/JOB_EXIT.json"
  return "$rc"
}
trap archive_failure_evidence EXIT

if [ ! -d "$repo_root" ]; then
  mkdir -m 700 "$repo_root"
  tar -xf "$NCD_ARCHIVE" -C "$repo_root"
fi
test -f "$repo_root/ncd/run_exp3_phase1_dataset.py"
test -f "$repo_root/ncd/source_commit.py"
mkdir -p "$run_root/kaggle_leon_release/datasets"
if [ ! -e "$run_root/kaggle_leon_release/datasets/$dataset_slug" ]; then
  ln -s "/scr/user/leon12138/ofra-etg/authorized-raw-20260820/$dataset_slug" \
    "$run_root/kaggle_leon_release/datasets/$dataset_slug"
fi
test "$(realpath -e "$repo_root/../kaggle_leon_release/datasets/$dataset_slug")" = \
  "/scr/user/leon12138/ofra-etg/authorized-raw-20260820/$dataset_slug"

if [ "$NCD_DATASET" = nsl ]; then
  printf '%s  %s\n' fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84 \
    "$repo_root/../kaggle_leon_release/datasets/nsl-kdd/KDDTest+.txt" | sha256sum -c
  printf '%s  %s\n' 1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95 \
    "$repo_root/../kaggle_leon_release/datasets/nsl-kdd/KDDTrain+.txt" | sha256sum -c
else
  printf '%s  %s\n' 734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559 \
    "$repo_root/../kaggle_leon_release/datasets/unsw-nb15/UNSW_NB15_testing-set.csv" | sha256sum -c
  printf '%s  %s\n' bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa \
    "$repo_root/../kaggle_leon_release/datasets/unsw-nb15/UNSW_NB15_training-set.csv" | sha256sum -c
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export NCD_SOURCE_COMMIT
python="$HOME/.venvs/ofra-etg/bin/python"
test -x "$python"

printf 'JOB_START id=%s dataset=%s source=%s host=%s utc=%s\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$(hostname -s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi
cd "$repo_root"
"$python" -I ncd/run_exp3_phase1_dataset.py \
  --dataset "$NCD_DATASET" \
  --state-path "ncd/results/exp3/${NCD_DATASET}_phase1_hpc_state.json"

state="$repo_root/ncd/results/exp3/${NCD_DATASET}_phase1_hpc_state.json"
"$python" -c 'import json,sys; x=json.load(open(sys.argv[1], encoding="utf-8")); assert x["complete"] and len(x["completed_runs"]) == x["expected_runs"] == 25' "$state"
mkdir -p "$protected/results" "$protected/logs" "$protected/configs"
cp -a "$repo_root/ncd/results/exp3/runs/$dataset_slug" "$protected/results/"
cp -a "$repo_root/ncd/logs/." "$protected/logs/"
cp -a "$repo_root/ncd/configs/exp3_${NCD_DATASET}_"*.json "$protected/configs/"
cp -a "$state" "$protected/"
find "$protected/results" "$protected/logs" "$protected/configs" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$protected/SHA256SUMS.txt"
printf '{"job_id":"%s","dataset":"%s","dataset_slug":"%s","source_commit":"%s","runs":25,"completed_utc":"%s"}\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$dataset_slug" "$NCD_SOURCE_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$protected/COMPLETED.json"
printf 'JOB_COMPLETE id=%s dataset=%s protected=%s\n' "$SLURM_JOB_ID" "$NCD_DATASET" "$protected"
trap - EXIT
