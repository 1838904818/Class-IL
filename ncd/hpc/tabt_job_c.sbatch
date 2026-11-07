#!/usr/bin/env bash
#SBATCH --job-name=cse-nsl-tabt
#SBATCH --partition=gpu-v100s
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:v100s:2
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:TERM@120
set -euo pipefail
umask 077
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DICC_CHECKPOINT=per_task
: "${NCD_CODE_ROOT:?}" "${NCD_FROZEN_ROOT:?}" "${NCD_DATA_ROOT:?}" "${NCD_OUTPUT_ROOT:?}"
: "${NCD_CACHE_ROOT:?}" "${NCD_TABT_DEPENDENCY_ROOT:?}" "${NCD_PROTECTED_ROOT:?}" "${NCD_PYTHON:?}"
export TMPDIR="$NCD_CACHE_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
cd "$NCD_CODE_ROOT"
: "${NCD_INPUT_ARCHIVE:?}" "${NCD_INPUT_ARCHIVE_SHA256:?}"
# Fresh verified data are unpacked only inside the allocated compute session.
# Some compute images do not expose the zstd executable even when libzstd is
# available. Prefer the executable when present; otherwise use the bounded
# stdlib/ctypes streaming decoder, without installing software or staging a
# second copy of the archive.
srun --hint=nomultithread --cpu-bind=cores bash -c 'set -euo pipefail
printf "%s  %s\n" "$NCD_INPUT_ARCHIVE_SHA256" "$NCD_INPUT_ARCHIVE" | sha256sum -c -
zstd_bin="$(command -v zstd || true)"
if [[ -z "$zstd_bin" ]]; then
  for candidate in /usr/bin/zstd /bin/zstd; do
    if [[ -x "$candidate" ]]; then zstd_bin="$candidate"; break; fi
  done
fi
if [[ -n "$zstd_bin" ]]; then
  echo "ARCHIVE_DECODER=$zstd_bin" >&2
  "$zstd_bin" -dc "$NCD_INPUT_ARCHIVE" | tar -xf - -C "$NCD_DATA_ROOT"
else
  echo "ARCHIVE_DECODER=python-ctypes-libzstd" >&2
  "$NCD_PYTHON" -I -B ncd/hpc/extract_zstd_stream.py "$NCD_INPUT_ARCHIVE" | tar -xf - -C "$NCD_DATA_ROOT"
fi'
# Five large CSE seeds share two GPUs, at most three workers on each GPU.
# Task checkpoints survive an allocation limit; no automatic scheduler retry.
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u ncd/hpc/tabt_job_c.py
