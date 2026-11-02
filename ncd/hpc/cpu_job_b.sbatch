#!/usr/bin/env bash
#SBATCH --job-name=cse-nsl-cpu
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:TERM@120
set -euo pipefail
umask 077
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
: "${NCD_CODE_ROOT:?}" "${NCD_DATA_ROOT:?}" "${NCD_FROZEN_ROOT:?}"
: "${NCD_CACHE_ROOT:?}" "${NCD_OUTPUT_ROOT:?}" "${NCD_PROTECTED_ROOT:?}" "${NCD_PYTHON:?}"
export TMPDIR="$NCD_CACHE_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
cd "$NCD_CODE_ROOT"
: "${NCD_INPUT_ARCHIVE:?}" "${NCD_INPUT_ARCHIVE_SHA256:?}"
# Fresh verified data are unpacked only inside the allocated compute session.
srun --hint=nomultithread --cpu-bind=cores bash -c 'set -euo pipefail; printf "%s  %s\n" "$NCD_INPUT_ARCHIVE_SHA256" "$NCD_INPUT_ARCHIVE" | sha256sum -c -; zstd -dc "$NCD_INPUT_ARCHIVE" | tar -xf - -C "$NCD_DATA_ROOT"'
# At most 16 bounded workers; at most two full CSE mappings concurrently.
# Resume task checkpoints; no automatic scheduler submissions.
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u ncd/hpc/cpu_job_b.py
