#!/usr/bin/env bash
#SBATCH --partition=cpu-epyc
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077

: "${NCD_SOURCE_COMMIT:?NCD_SOURCE_COMMIT is required}"
: "${NCD_ARCHIVE:?NCD_ARCHIVE is required}"
: "${NCD_ARCHIVE_SHA256:?NCD_ARCHIVE_SHA256 is required}"
: "${NCD_SCRATCH_ROOT:?NCD_SCRATCH_ROOT is required}"
: "${NCD_PROTECTED_ROOT:?NCD_PROTECTED_ROOT is required}"

test "$(id -un)" = leon12138
test "$HOME" = /home/user/leon12138
test -f "$NCD_ARCHIVE" && test ! -L "$NCD_ARCHIVE"
printf '%s  %s\n' "$NCD_ARCHIVE_SHA256" "$NCD_ARCHIVE" | sha256sum -c --status

run_root="$NCD_SCRATCH_ROOT/cache-validation-$SLURM_JOB_ID"
repo_root="$run_root/repo"
data_root="$run_root/kaggle_leon_release/datasets"
protected="$NCD_PROTECTED_ROOT/cache-validation-$SLURM_JOB_ID"
mkdir -m 700 -p "$repo_root" "$data_root" "$protected"

archive_evidence() {
  rc=$?
  if [ -f "$run_root/cache_validation.json" ]; then
    cp -a "$run_root/cache_validation.json" "$protected/"
  fi
  printf '{"job_id":"%s","source_commit":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$NCD_SOURCE_COMMIT" "$rc" > "$protected/JOB_EXIT.json"
  return "$rc"
}
trap archive_evidence EXIT

tar -xf "$NCD_ARCHIVE" -C "$repo_root"
test -f "$repo_root/ncd/validate_cache_data.py"

cache_base=/scr/user/leon12138/ofra-etg/transfers/20260730/core/ofra_formal_v3_cache_20260716
ln -s "$cache_base/cic-ids-2017" "$data_root/cic-ids-2017"
ln -s "$cache_base/cic-ids-2018" "$data_root/cic-ids-2018"
test "$(realpath -e "$data_root/cic-ids-2017")" = "$cache_base/cic-ids-2017"
test "$(realpath -e "$data_root/cic-ids-2018")" = "$cache_base/cic-ids-2018"

printf '%s  %s\n' 3995c35da82934b7fe8986f54fbd3c8dcc64787f44c8a54731880cd9b883f61d \
  "$data_root/cic-ids-2017/manifest.json" | sha256sum -c
printf '%s  %s\n' d33e23185c2b9111b26b965cbef55d708db3def2d18dbec6b699a6d353f00c20 \
  "$data_root/cic-ids-2017/feature_schema.json" | sha256sum -c
printf '%s  %s\n' c9de770b59a06821b25b38249217461c657a08c1f156b89a7e65376ac82b9fed \
  "$data_root/cic-ids-2018/manifest.json" | sha256sum -c
printf '%s  %s\n' ac2fa15c57546f320a414e258fae16ce928f457c75a91d357179e0c99458fca0 \
  "$data_root/cic-ids-2018/feature_schema.json" | sha256sum -c

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export NCD_SOURCE_COMMIT
python="$HOME/.venvs/ofra-etg/bin/python"
test -x "$python"

printf 'CACHE_JOB_START id=%s source=%s host=%s utc=%s\n' \
  "$SLURM_JOB_ID" "$NCD_SOURCE_COMMIT" "$(hostname -s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$repo_root"
"$python" -I ncd/validate_cache_data.py \
  --data-root "$data_root" \
  --datasets CIC-IDS-2017 CSE-CIC-IDS2018 \
  --output "$run_root/cache_validation.json"

cp -a "$run_root/cache_validation.json" "$protected/"
sha256sum "$protected/cache_validation.json" > "$protected/SHA256SUMS.txt"
printf '{"job_id":"%s","source_commit":"%s","completed_utc":"%s"}\n' \
  "$SLURM_JOB_ID" "$NCD_SOURCE_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$protected/COMPLETED.json"
printf 'CACHE_JOB_COMPLETE id=%s protected=%s\n' "$SLURM_JOB_ID" "$protected"
trap - EXIT
