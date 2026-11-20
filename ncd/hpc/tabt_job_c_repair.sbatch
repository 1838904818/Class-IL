#!/usr/bin/env bash
#SBATCH --job-name=cse-tabt-repair
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8 DICC_CHECKPOINT=per_task
: "${NCD_PYTHON:?}" "${NCD_CODE_ROOT:?}" "${NCD_FROZEN_ROOT:?}" "${NCD_DATA_ROOT:?}"
: "${NCD_OUTPUT_ROOT:?}" "${NCD_PROTECTED_ROOT:?}" "${NCD_CACHE_ROOT:?}"
: "${NCD_EXECUTION_SOURCE_COMMIT:?}" "${NCD_SOURCE_COMMIT:?}"
export TMPDIR="$NCD_CACHE_ROOT/tmp" TEMP="$NCD_CACHE_ROOT/tmp" TMP="$NCD_CACHE_ROOT/tmp"
test -d "$TMPDIR"
cd "$NCD_CODE_ROOT"
printf '%s  %s\n' 659d7d7c018265aac0258d027a63133ce69eb97e21345dd6eb71f3a82538f3c2 tabt_extension_cell.py | sha256sum -c -
sha256sum tabt_job_c.py tabt_cpu_gpu_portable.py tabt_extension_cell.py tabt_checkpoint_repair.py tabt_feature_extension.py
srun --hint=nomultithread --cpu-bind=cores "$NCD_PYTHON" -I -B -u tabt_job_c.py \
  --datasets CSE-CIC-IDS2018 --repair-from "/scr/user/$USER/classil-c-v3-20261110"
