#!/usr/bin/env bash
#SBATCH --job-name=ncd-crr-s15
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=gpu-a100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00

set -euo pipefail
umask 077

# The same batch entry runs an end-to-end local GPU smoke before submission.
if [ "${NCD_LOCAL_SMOKE:-0}" = 1 ]; then
  exec "$NCD_PYTHON" -I ncd/run_crr_real_matrix.py --dataset "$NCD_DATASET_NAME" --lambda-file "$NCD_LAMBDA_FILE" --output-dir "$NCD_LOCAL_OUTPUT" --smoke
fi

: "${NCD_MODE:?NCD_MODE must be necessity or real}"
: "${NCD_DATASET:?NCD_DATASET must be nsl or unsw}"
: "${NCD_SOURCE_COMMIT:?NCD_SOURCE_COMMIT is required}"
: "${NCD_ARCHIVE:?NCD_ARCHIVE is required}"
: "${NCD_ARCHIVE_SHA256:?NCD_ARCHIVE_SHA256 is required}"
: "${NCD_SCRATCH_ROOT:?NCD_SCRATCH_ROOT is required}"
: "${NCD_PROTECTED_ROOT:?NCD_PROTECTED_ROOT is required}"

case "$NCD_MODE" in necessity|real) ;; *) exit 64 ;; esac
case "$NCD_DATASET" in nsl|unsw) ;; *) exit 64 ;; esac
if [ "$NCD_MODE" = real ]; then
  : "${NCD_LAMBDA_ROOT:?NCD_LAMBDA_ROOT is required for real mode}"
fi

test "$(id -un)" = "$USER"
test "$HOME" = "/home/user/$USER"
test -f "$NCD_ARCHIVE" && test ! -L "$NCD_ARCHIVE"
printf '%s  %s\n' "$NCD_ARCHIVE_SHA256" "$NCD_ARCHIVE" | sha256sum -c --status

run_root="$NCD_SCRATCH_ROOT/$NCD_MODE-$NCD_DATASET-$SLURM_JOB_ID"
repo_root="$run_root/repo"
protected="$NCD_PROTECTED_ROOT"
mkdir -p "$repo_root" "$protected"
chmod 700 "$run_root" "$protected"

archive_evidence() {
  rc=$?
  mkdir -p "$protected/results" "$protected/logs"
  if [ -d "$repo_root/ncd/results/crr/section15" ]; then
    cp -a "$repo_root/ncd/results/crr/section15/." "$protected/results/"
  fi
  if [ -d "$repo_root/ncd/logs/section15" ]; then
    cp -a "$repo_root/ncd/logs/section15/." "$protected/logs/"
  fi
  printf '{"job_id":"%s","mode":"%s","dataset":"%s","source_commit":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$NCD_MODE" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$rc" > "$protected/JOB_EXIT.json"
  return "$rc"
}
trap archive_evidence EXIT

tar -xf "$NCD_ARCHIVE" -C "$repo_root"
test -f "$repo_root/ncd/run_crr_experiment.py"
test -f "$repo_root/ncd/run_crr_real_sequence.py"
test -f "$repo_root/src/__init__.py"
mkdir -p "$run_root/kaggle_leon_release/datasets"

case "$NCD_DATASET" in
  nsl) dataset_name=NSL-KDD; data_name=nsl-kdd ;;
  unsw) dataset_name=UNSW-NB15; data_name=unsw-nb15 ;;
esac
source_path="/scr/user/$USER/ofra-etg/authorized-raw-20260820/$data_name"
test -d "$source_path"
ln -s "$source_path" "$run_root/kaggle_leon_release/datasets/$data_name"
test "$(realpath -e "$repo_root/../kaggle_leon_release/datasets/$data_name")" = "$source_path"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export NCD_SOURCE_COMMIT
python_bin="$HOME/.venvs/ofra-etg/bin/python"
test -x "$python_bin"

printf 'CRR_S15_START id=%s mode=%s dataset=%s source=%s host=%s utc=%s\n' \
  "$SLURM_JOB_ID" "$NCD_MODE" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$(hostname -s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi
cd "$repo_root"
"$python_bin" -I -c \
  'import sys; sys.path.insert(0, sys.argv[1]); import ncd.crr_benchmarks, ncd.run_crr_real_sequence, src; print("IMPORT_SMOKE_OK")' \
  "$repo_root"

mkdir -p ncd/results/crr/section15 ncd/logs/section15
# Resume completed shards only from the same source commit; interrupted cells rerun.
if [ -n "${NCD_RESUME_ROOT:-}" ]; then
  test -f "$NCD_RESUME_ROOT/JOB_EXIT.json"
  "$python_bin" -I -c 'import json,sys; assert json.load(open(sys.argv[1]))["source_commit"]==sys.argv[2]' "$NCD_RESUME_ROOT/JOB_EXIT.json" "$NCD_SOURCE_COMMIT"
  cp -a "$NCD_RESUME_ROOT/results/." ncd/results/crr/section15/
  cp -a "$NCD_RESUME_ROOT/logs/." ncd/logs/section15/
fi
methods=(replay icarl replay_lwf kd_perturbed rcl_tab global_ir oldpair_kd crr)
seeds=(1 2 3 4 42)

if [ "$NCD_MODE" = necessity ]; then
  if [ -n "${NCD_CALIBRATION_ROOT:-}" ]; then
    test -d "$NCD_CALIBRATION_ROOT/results/$NCD_DATASET/calibration"
    cp -a "$NCD_CALIBRATION_ROOT/results/$NCD_DATASET" "ncd/results/crr/section15/"
    cp -a "$NCD_CALIBRATION_ROOT/logs/." ncd/logs/section15/
    cp "$NCD_CALIBRATION_ROOT/JOB_EXIT.json" "ncd/results/crr/section15/$NCD_DATASET/calibration_source_job.json"
  fi
  regularized=(icarl replay_lwf kd_perturbed rcl_tab global_ir oldpair_kd crr)
  selections=()
  for method in "${regularized[@]}"; do
    candidates=()
    for lambda in 0.25 0.5 1.0 2.0; do
      safe_lambda=${lambda/./p}
      output="ncd/results/crr/section15/$NCD_DATASET/calibration/$method/lambda_$safe_lambda"
      log="ncd/logs/section15/${NCD_DATASET}_${method}_lambda_${safe_lambda}.log"
      if [ -f "$output/crr_result.json" ]; then
        "$python_bin" -I -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["git_commit"]==sys.argv[2]; assert len(x["rows"])==8; assert {r["dataset"] for r in x["rows"]}=={sys.argv[3]}' "$output/crr_result.json" "${NCD_CALIBRATION_SOURCE_COMMIT:?}" "$dataset_name"
      else
      srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        "$python_bin" -I ncd/run_crr_experiment.py \
        --config ncd/configs/crr_section15.json --device cuda --dataset "$dataset_name" \
        --seeds 5,6 --methods "replay,$method" --benchmarks necessity \
        --regularization "$lambda" --output-dir "$output" --log-path "$log"
      fi
      candidates+=("$output/crr_result.json")
    done
    selection="ncd/results/crr/section15/$NCD_DATASET/calibration/$method/lambda_selection.json"
    "$python_bin" -I ncd/select_crr_section15_lambda.py "${candidates[@]}" \
      --method "$method" --output "$selection"
    selections+=("$selection")
  done
  lambda_file="ncd/results/crr/section15/$NCD_DATASET/lambda_selections.json"
  "$python_bin" -I ncd/collect_crr_section15_lambdas.py "${selections[@]}" \
    --dataset "$dataset_name" --source-commit "$NCD_SOURCE_COMMIT" --output "$lambda_file"

  shards=()
  for method in "${methods[@]}"; do
    lambda=$(
      "$python_bin" -I -c \
        'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["lambdas"][sys.argv[2]])' \
        "$lambda_file" "$method"
    )
    output="ncd/results/crr/section15/$NCD_DATASET/necessity/shards/$method"
    log="ncd/logs/section15/${NCD_DATASET}_necessity_${method}.log"
    if [ -f "$output/COMPLETED.json" ]; then
      "$python_bin" -I -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["git_commit"]==sys.argv[2]; assert len(x["rows"])==10' "$output/crr_result.json" "$NCD_SOURCE_COMMIT"
    else
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      "$python_bin" -I ncd/run_crr_experiment.py \
      --config ncd/configs/crr_section15.json --device cuda --dataset "$dataset_name" \
      --seeds 1,2,3,4,42 --methods "$method" --benchmarks necessity \
      --regularization "$lambda" --output-dir "$output" --log-path "$log"
    fi
    "$python_bin" -I -c \
      'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); assert x["git_commit"]==sys.argv[2]; assert len(x["rows"])==10' \
      "$output/crr_result.json" "$NCD_SOURCE_COMMIT"
    shards+=("$output/crr_result.json")
  done
  merged="ncd/results/crr/section15/$NCD_DATASET/necessity/necessity_result.json"
  "$python_bin" -I ncd/merge_crr_section15.py "${shards[@]}" \
    --mode necessity --dataset "$dataset_name" --source-commit "$NCD_SOURCE_COMMIT" --output "$merged"
else
  lambda_file="$NCD_LAMBDA_ROOT/lambda_selections.json"
  test -f "$lambda_file" && test ! -L "$lambda_file"
  "$python_bin" -I -c \
    'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); assert x["git_commit"]==sys.argv[2]; assert x["dataset"]==sys.argv[3]' \
    "$lambda_file" "${NCD_LAMBDA_SOURCE_COMMIT:-$NCD_SOURCE_COMMIT}" "$dataset_name"
  mkdir -p "ncd/results/crr/section15/$NCD_DATASET/real"
  cp "$lambda_file" "ncd/results/crr/section15/$NCD_DATASET/real/lambda_selections.json"
  srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$python_bin" -I ncd/run_crr_real_matrix.py \
    --dataset "$dataset_name" --lambda-file "$lambda_file" \
    --output-dir "ncd/results/crr/section15/$NCD_DATASET/real"
fi

mkdir -p "$protected/results" "$protected/logs" "$protected/configs"
cp -a "$repo_root/ncd/results/crr/section15/." "$protected/results/"
cp -a "$repo_root/ncd/logs/section15/." "$protected/logs/"
cp -a "$repo_root/ncd/configs/crr_section15.json" "$protected/configs/"
find "$protected/results" "$protected/logs" "$protected/configs" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$protected/SHA256SUMS.txt"
printf '{"job_id":"%s","mode":"%s","dataset":"%s","source_commit":"%s","completed_utc":"%s"}\n' \
  "$SLURM_JOB_ID" "$NCD_MODE" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$protected/COMPLETED.json"
printf 'CRR_S15_COMPLETE id=%s mode=%s dataset=%s protected=%s\n' \
  "$SLURM_JOB_ID" "$NCD_MODE" "$NCD_DATASET" "$protected"
