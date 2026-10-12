#!/usr/bin/env bash
#SBATCH --job-name=ncd-crr-s13
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=gpu-a100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00

set -euo pipefail
umask 077
# DICC_CHECKPOINT: every completed calibration/formal stage is copied to
# protected storage and restored on a same-commit resume.

: "${NCD_MODE:?NCD_MODE must be learnability or corrected}"
: "${NCD_DATASET:?NCD_DATASET must be all, nsl, or unsw}"
: "${NCD_SOURCE_COMMIT:?NCD_SOURCE_COMMIT is required}"
: "${NCD_ARCHIVE:?NCD_ARCHIVE is required}"
: "${NCD_ARCHIVE_SHA256:?NCD_ARCHIVE_SHA256 is required}"
: "${NCD_SCRATCH_ROOT:?NCD_SCRATCH_ROOT is required}"
: "${NCD_PROTECTED_ROOT:?NCD_PROTECTED_ROOT is required}"

case "$NCD_MODE" in learnability|corrected) ;; *) exit 64 ;; esac
case "$NCD_DATASET" in all|nsl|unsw) ;; *) exit 64 ;; esac
if [ "$NCD_MODE" = learnability ] && [ "$NCD_DATASET" != all ]; then
  printf 'learnability mode must cover both datasets\n' >&2
  exit 64
fi
if [ "$NCD_MODE" = corrected ] && [ "$NCD_DATASET" = all ]; then
  printf 'corrected mode must use one dataset per job\n' >&2
  exit 64
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
  if [ -d "$repo_root/ncd/results/crr" ]; then cp -a "$repo_root/ncd/results/crr/." "$protected/results/"; fi
  if [ -d "$repo_root/ncd/logs" ]; then cp -a "$repo_root/ncd/logs/." "$protected/logs/"; fi
  printf '{"job_id":"%s","mode":"%s","dataset":"%s","source_commit":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$NCD_MODE" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$rc" > "$protected/JOB_EXIT.json"
  return "$rc"
}
trap archive_evidence EXIT

tar -xf "$NCD_ARCHIVE" -C "$repo_root"
test -f "$repo_root/ncd/run_crr_experiment.py"
test -f "$repo_root/ncd/crr_benchmarks.py"
test -f "$repo_root/ncd/validate_crr_learnability.py"
test -f "$repo_root/ncd/select_section13_lambda.py"
test -f "$repo_root/src/__init__.py"
if [ -d "$protected/results" ]; then
  mkdir -p "$repo_root/ncd/results/crr"
  cp -a "$protected/results/." "$repo_root/ncd/results/crr/"
fi
mkdir -p "$run_root/kaggle_leon_release/datasets"

link_dataset() {
  key=$1
  case "$key" in
    nsl) name=nsl-kdd ;;
    unsw) name=unsw-nb15 ;;
    *) return 64 ;;
  esac
  source_path="/scr/user/$USER/ofra-etg/authorized-raw-20260820/$name"
  test -d "$source_path"
  ln -s "$source_path" "$run_root/kaggle_leon_release/datasets/$name"
  test "$(realpath -e "$repo_root/../kaggle_leon_release/datasets/$name")" = "$source_path"
}

if [ "$NCD_DATASET" = all ]; then
  link_dataset nsl
  link_dataset unsw
else
  link_dataset "$NCD_DATASET"
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export NCD_SOURCE_COMMIT
python_bin="$HOME/.venvs/ofra-etg/bin/python"
test -x "$python_bin"

printf 'CRR_S13_START id=%s mode=%s dataset=%s source=%s host=%s utc=%s\n' \
  "$SLURM_JOB_ID" "$NCD_MODE" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$(hostname -s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi
cd "$repo_root"
"$python_bin" -I -c \
  'import sys; sys.path.insert(0, sys.argv[1]); import ncd.crr_benchmarks, src; print("IMPORT_SMOKE_OK")' \
  "$repo_root"

result_valid() {
  output=$1
  expected_rows=$2
  result="$output/crr_result.json"
  completed="$output/COMPLETED.json"
  test -f "$result" && test -f "$completed" && \
    "$python_bin" -I -c \
      'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); c=json.load(open(sys.argv[2],encoding="utf-8")); assert x["git_commit"]==sys.argv[3]; assert c["git_commit"]==sys.argv[3]; assert len(x["rows"])==int(sys.argv[4])' \
      "$result" "$completed" "$NCD_SOURCE_COMMIT" "$expected_rows"
}

checkpoint_output() {
  output=$1
  relative=${output#ncd/results/crr/}
  target="$protected/results/$relative"
  mkdir -p "$target"
  cp -a "$output/." "$target/"
}

if [ "$NCD_MODE" = learnability ]; then
  output=ncd/results/crr/section13/learnability_seed1
  log=ncd/logs/crr_section13_learnability_seed1.log
  if ! result_valid "$output" 6; then
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      "$python_bin" -I ncd/run_crr_experiment.py \
      --config ncd/configs/crr_section12.json --device cuda \
      --seeds 1 --methods replay --benchmarks necessity \
      --output-dir "$output" --log-path "$log"
    checkpoint_output "$output"
  fi
  result="$output/crr_result.json"
  verdict="$output/LEARNABILITY_VERDICT.json"
  "$python_bin" -I ncd/validate_crr_learnability.py "$result" --output "$verdict"
  "$python_bin" -I -c \
    'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); v=json.load(open(sys.argv[2],encoding="utf-8")); assert x["git_commit"]==sys.argv[3]; assert len(x["rows"])==6; assert v["passed"] is True' \
    "$result" "$verdict" "$NCD_SOURCE_COMMIT"
else
  case "$NCD_DATASET" in
    nsl) dataset_name=NSL-KDD ;;
    unsw) dataset_name=UNSW-NB15 ;;
  esac
  declare -A selected
  for method in global_ir rcl_tab; do
    calibration_inputs=()
    for lambda in 0.25 0.5 1.0 2.0; do
      safe_lambda=${lambda/./p}
      output="ncd/results/crr/section13/calibration/$NCD_DATASET/$method/lambda_$safe_lambda"
      log="ncd/logs/crr_section13_calibration_${NCD_DATASET}_${method}_${safe_lambda}.log"
      if ! result_valid "$output" 28; then
        srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
          "$python_bin" -I ncd/run_crr_experiment.py \
          --config ncd/configs/crr_section12.json --device cuda --dataset "$dataset_name" \
          --seeds 5,6 --methods "replay,$method" --regularization "$lambda" \
          --output-dir "$output" --log-path "$log"
        checkpoint_output "$output"
      fi
      calibration_inputs+=("$output/crr_result.json")
    done
    selection="ncd/results/crr/section13/calibration/$NCD_DATASET/$method/lambda_selection.json"
    "$python_bin" -I ncd/select_section13_lambda.py "${calibration_inputs[@]}" \
      --method "$method" --output "$selection"
    selected[$method]=$("$python_bin" -I -c \
      'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["selected_lambda"])' "$selection")
  done

  for method in global_ir rcl_tab rcl_tab_nogate; do
    if [ "$method" = rcl_tab_nogate ]; then lambda=${selected[rcl_tab]}; else lambda=${selected[$method]}; fi
    output="ncd/results/crr/section13/formal/$NCD_DATASET/$method"
    log="ncd/logs/crr_section13_formal_${NCD_DATASET}_${method}.log"
    if ! result_valid "$output" 70; then
      srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        "$python_bin" -I ncd/run_crr_experiment.py \
        --config ncd/configs/crr_section12.json --device cuda --dataset "$dataset_name" \
        --methods "replay,$method" --regularization "$lambda" \
        --output-dir "$output" --log-path "$log"
      checkpoint_output "$output"
    fi
    result="$output/crr_result.json"
    "$python_bin" -I -c \
      'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); assert x["git_commit"]==sys.argv[2]; assert len(x["rows"])==70; assert sorted(x["effective_config"]["seeds"])==[1,2,3,4,42]' \
      "$result" "$NCD_SOURCE_COMMIT"
  done
fi

mkdir -p "$protected/results" "$protected/logs" "$protected/configs"
cp -a "$repo_root/ncd/results/crr/." "$protected/results/"
cp -a "$repo_root/ncd/logs/." "$protected/logs/"
cp -a "$repo_root/ncd/configs/crr_section12.json" "$protected/configs/"
find "$protected/results" "$protected/logs" "$protected/configs" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$protected/SHA256SUMS.txt"
printf '{"job_id":"%s","mode":"%s","dataset":"%s","source_commit":"%s","completed_utc":"%s"}\n' \
  "$SLURM_JOB_ID" "$NCD_MODE" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$protected/COMPLETED.json"
printf 'CRR_S13_COMPLETE id=%s mode=%s dataset=%s protected=%s\n' \
  "$SLURM_JOB_ID" "$NCD_MODE" "$NCD_DATASET" "$protected"
trap - EXIT
