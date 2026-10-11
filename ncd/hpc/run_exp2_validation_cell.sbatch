#!/usr/bin/env bash
#SBATCH --job-name=ncd-e2-valid
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

: "${NCD_DATASET:?NCD_DATASET must be nsl or unsw}"
: "${NCD_METHOD:?NCD_METHOD must be replay or icarl}"
: "${NCD_SOURCE_COMMIT:?NCD_SOURCE_COMMIT is required}"
: "${NCD_ARCHIVE:?NCD_ARCHIVE is required}"
: "${NCD_ARCHIVE_SHA256:?NCD_ARCHIVE_SHA256 is required}"
: "${NCD_SCRATCH_ROOT:?NCD_SCRATCH_ROOT is required}"
: "${NCD_PROTECTED_ROOT:?NCD_PROTECTED_ROOT is required}"

case "$NCD_DATASET" in
  nsl) dataset_dir=nsl-kdd; dataset_source="/scr/user/$USER/ofra-etg/authorized-raw-20260820/nsl-kdd" ;;
  unsw) dataset_dir=unsw-nb15; dataset_source="/scr/user/$USER/ofra-etg/authorized-raw-20260820/unsw-nb15" ;;
  *) exit 64 ;;
esac
case "$NCD_METHOD" in replay|icarl) ;; *) exit 64 ;; esac

test "$(id -un)" = "$USER"
test "$HOME" = "/home/user/$USER"
test -f "$NCD_ARCHIVE" && test ! -L "$NCD_ARCHIVE"
printf '%s  %s\n' "$NCD_ARCHIVE_SHA256" "$NCD_ARCHIVE" | sha256sum -c --status

run_root="$NCD_SCRATCH_ROOT/$NCD_DATASET-$NCD_METHOD"
repo_root="$run_root/repo"
protected="$NCD_PROTECTED_ROOT/$NCD_DATASET/$NCD_METHOD"
mkdir -p "$repo_root" "$protected"
chmod 700 "$run_root" "$protected"

archive_evidence() {
  rc=$?
  mkdir -p "$protected/logs"
  if [ -d "$repo_root/ncd/logs" ]; then cp -a "$repo_root/ncd/logs/." "$protected/logs/"; fi
  if [ -d "$repo_root/ncd/results/exp2_validation_v1/$dataset_dir/$NCD_METHOD" ]; then
    mkdir -p "$protected/results"
    cp -a "$repo_root/ncd/results/exp2_validation_v1/$dataset_dir/$NCD_METHOD/." "$protected/results/"
  fi
  printf '{"job_id":"%s","dataset":"%s","method":"%s","source_commit":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_METHOD" "$NCD_SOURCE_COMMIT" "$rc" > "$protected/JOB_EXIT.json"
  return "$rc"
}
trap archive_evidence EXIT

if [ ! -f "$repo_root/ncd/groundtruth.py" ]; then
  tar -xf "$NCD_ARCHIVE" -C "$repo_root"
fi
test -f "$repo_root/ncd/groundtruth.py"
mkdir -p "$run_root/kaggle_leon_release/datasets"
if [ ! -e "$run_root/kaggle_leon_release/datasets/$dataset_dir" ]; then
  ln -s "$dataset_source" "$run_root/kaggle_leon_release/datasets/$dataset_dir"
fi
test "$(realpath -e "$repo_root/../kaggle_leon_release/datasets/$dataset_dir")" = "$dataset_source"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export NCD_SOURCE_COMMIT
python_bin="$HOME/.venvs/ofra-etg/bin/python"
test -x "$python_bin"

printf 'VALIDATION_START id=%s dataset=%s method=%s source=%s host=%s utc=%s\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_METHOD" "$NCD_SOURCE_COMMIT" "$(hostname -s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi
cd "$repo_root"
for seed in 5 6 7 8 9; do
  config="ncd/configs/exp2_validation_v1_${NCD_DATASET}_${NCD_METHOD}_seed${seed}.json"
  result="ncd/results/exp2_validation_v1/$dataset_dir/$NCD_METHOD/seed${seed}/exp2_groundtruth_result.json"
  if [ -f "$result" ]; then
    "$python_bin" -c 'import json,sys; x=json.load(open(sys.argv[1], encoding="utf-8")); assert x["git_commit"] == sys.argv[2]; assert int(x["effective_config"]["seed"]) == int(sys.argv[3]); assert len(x["scenario_records"]) >= 8' "$result" "$NCD_SOURCE_COMMIT" "$seed"
    printf 'SKIP_COMPLETE config=%s\n' "$config"
    continue
  fi
  srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$python_bin" -I ncd/groundtruth.py --config "$config"
  "$python_bin" -c 'import json,sys; x=json.load(open(sys.argv[1], encoding="utf-8")); assert x["git_commit"] == sys.argv[2]; assert int(x["effective_config"]["seed"]) == int(sys.argv[3]); assert len(x["scenario_records"]) >= 8' "$result" "$NCD_SOURCE_COMMIT" "$seed"
done

mkdir -p "$protected/results" "$protected/logs" "$protected/configs"
cp -a "$repo_root/ncd/results/exp2_validation_v1/$dataset_dir/$NCD_METHOD/." "$protected/results/"
cp -a "$repo_root/ncd/logs/." "$protected/logs/"
cp -a "$repo_root/ncd/configs/exp2_validation_v1_${NCD_DATASET}_${NCD_METHOD}_seed"*.json "$protected/configs/"
find "$protected/results" "$protected/logs" "$protected/configs" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$protected/SHA256SUMS.txt"
printf '{"job_id":"%s","dataset":"%s","method":"%s","source_commit":"%s","seeds":[5,6,7,8,9],"completed_utc":"%s"}\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_METHOD" "$NCD_SOURCE_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$protected/COMPLETED.json"
printf 'VALIDATION_COMPLETE id=%s dataset=%s method=%s protected=%s\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_METHOD" "$protected"
trap - EXIT
