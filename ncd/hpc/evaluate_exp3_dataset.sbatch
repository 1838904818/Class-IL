#!/usr/bin/env bash
#SBATCH --job-name=ncd-e3-perf
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

: "${NCD_DATASET:?NCD_DATASET must be nsl, unsw, cic17, or cic18}"
: "${NCD_SOURCE_COMMIT:?NCD_SOURCE_COMMIT is required}"
: "${NCD_ARCHIVE:?NCD_ARCHIVE is required}"
: "${NCD_ARCHIVE_SHA256:?NCD_ARCHIVE_SHA256 is required}"
: "${NCD_SCRATCH_ROOT:?NCD_SCRATCH_ROOT is required}"
: "${NCD_PROTECTED_ROOT:?NCD_PROTECTED_ROOT is required}"

case "$NCD_DATASET" in
  nsl) dataset_slug=nsl-kdd ;;
  unsw) dataset_slug=unsw-nb15 ;;
  cic17) dataset_slug=cic-ids-2017 ;;
  cic18) dataset_slug=cic-ids-2018 ;;
  *) exit 64 ;;
esac
test "$(id -un)" = "$USER"
test "$HOME" = "/home/user/$USER"
test -f "$NCD_ARCHIVE" && test ! -L "$NCD_ARCHIVE"
printf '%s  %s\n' "$NCD_ARCHIVE_SHA256" "$NCD_ARCHIVE" | sha256sum -c --status

upstream_root="$NCD_SCRATCH_ROOT/$NCD_DATASET"
upstream_repo="$upstream_root/repo"
data_root="$upstream_root/kaggle_leon_release/datasets"
state="$upstream_repo/ncd/results/exp3/${NCD_DATASET}_phase1_hpc_state.json"
run_root="$NCD_SCRATCH_ROOT/performance-$NCD_DATASET-$SLURM_JOB_ID"
analysis_repo="$run_root/repo"
protected="$NCD_PROTECTED_ROOT/$NCD_DATASET/performance-$SLURM_JOB_ID"
mkdir -m 700 -p "$analysis_repo" "$protected"

archive_evidence() {
  rc=$?
  if [ -f "$run_root/performance_metrics.json" ]; then
    cp -a "$run_root/performance_metrics.json" "$protected/"
  fi
  printf '{"job_id":"%s","dataset":"%s","source_commit":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$rc" > "$protected/JOB_EXIT.json"
  return "$rc"
}
trap archive_evidence EXIT

tar -xf "$NCD_ARCHIVE" -C "$analysis_repo"
test -f "$analysis_repo/ncd/evaluate_exp3_performance.py"
test -f "$state"
test -d "$data_root/$dataset_slug"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export NCD_SOURCE_COMMIT
python_bin="$HOME/.venvs/ofra-etg/bin/python"
test -x "$python_bin"

printf 'PERFORMANCE_JOB_START id=%s dataset=%s source=%s host=%s utc=%s\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$(hostname -s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi
cd "$analysis_repo"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$python_bin" -I ncd/evaluate_exp3_performance.py \
  --run-repo "$upstream_repo" \
  --state-path "ncd/results/exp3/${NCD_DATASET}_phase1_hpc_state.json" \
  --data-root "$data_root" \
  --device cuda \
  --batch-size 4096 \
  --output "$run_root/performance_metrics.json"

cp -a "$run_root/performance_metrics.json" "$protected/"
sha256sum "$protected/performance_metrics.json" > "$protected/SHA256SUMS.txt"
printf '{"job_id":"%s","dataset":"%s","source_commit":"%s","completed_utc":"%s"}\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$protected/COMPLETED.json"
printf 'PERFORMANCE_JOB_COMPLETE id=%s dataset=%s protected=%s\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$protected"
trap - EXIT
