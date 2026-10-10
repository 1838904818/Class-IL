#!/usr/bin/env bash
#SBATCH --job-name=ncd-e3-drift
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=cpu-epyc
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00

set -euo pipefail
umask 077

: "${NCD_DATASET:?NCD_DATASET must be nsl, unsw, cic17, or cic18}"
: "${NCD_SOURCE_COMMIT:?NCD_SOURCE_COMMIT is required}"
: "${NCD_ARCHIVE:?NCD_ARCHIVE is required}"
: "${NCD_ARCHIVE_SHA256:?NCD_ARCHIVE_SHA256 is required}"
: "${NCD_SCRATCH_ROOT:?NCD_SCRATCH_ROOT is required}"
: "${NCD_PROTECTED_ROOT:?NCD_PROTECTED_ROOT is required}"

case "$NCD_DATASET" in
  nsl|unsw|cic17|cic18) ;;
  *) exit 64 ;;
esac
test "$(id -un)" = "$USER"
test "$HOME" = "/home/user/$USER"
test -f "$NCD_ARCHIVE" && test ! -L "$NCD_ARCHIVE"
printf '%s  %s\n' "$NCD_ARCHIVE_SHA256" "$NCD_ARCHIVE" | sha256sum -c --status

upstream_repo="$NCD_SCRATCH_ROOT/$NCD_DATASET/repo"
state="$upstream_repo/ncd/results/exp3/${NCD_DATASET}_phase1_hpc_state.json"
run_root="$NCD_SCRATCH_ROOT/drift-$NCD_DATASET-$SLURM_JOB_ID"
analysis_repo="$run_root/repo"
output_dir="$run_root/analysis"
protected="$NCD_PROTECTED_ROOT/$NCD_DATASET/drift-$SLURM_JOB_ID"
mkdir -m 700 -p "$analysis_repo" "$output_dir" "$protected"

archive_evidence() {
  rc=$?
  if [ -d "$output_dir" ]; then
    cp -a "$output_dir/." "$protected/"
  fi
  printf '{"job_id":"%s","dataset":"%s","source_commit":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$rc" > "$protected/JOB_EXIT.json"
  return "$rc"
}
trap archive_evidence EXIT

tar -xf "$NCD_ARCHIVE" -C "$analysis_repo"
test -f "$analysis_repo/ncd/analyze_exp3.py"
test -f "$state"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export NCD_SOURCE_COMMIT
python_bin="$HOME/.venvs/ofra-etg/bin/python"
test -x "$python_bin"

printf 'DRIFT_JOB_START id=%s dataset=%s source=%s host=%s utc=%s\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$(hostname -s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$analysis_repo"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$python_bin" -I ncd/analyze_exp3.py \
  --state-path "$state" \
  --run-repo "$upstream_repo" \
  --output-dir "$output_dir" \
  --permutations 1000 \
  --log-path "$output_dir/analysis.log"

test -f "$output_dir/exp3_result.json"
cp -a "$output_dir/." "$protected/"
find "$protected" -maxdepth 1 -type f ! -name 'SHA256SUMS.txt' -print0 \
  | sort -z | xargs -0 sha256sum > "$protected/SHA256SUMS.txt"
printf '{"job_id":"%s","dataset":"%s","source_commit":"%s","completed_utc":"%s"}\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$NCD_SOURCE_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$protected/COMPLETED.json"
printf 'DRIFT_JOB_COMPLETE id=%s dataset=%s protected=%s\n' \
  "$SLURM_JOB_ID" "$NCD_DATASET" "$protected"
trap - EXIT
