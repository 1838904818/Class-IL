#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$HOME/classil-20261102"
cd "$root/cse-wacb-score-guarded-20261102"
export NCD_PYTHON="$HOME/.venvs/ofra-etg/bin/python"
export NCD_PREFLIGHT="$root/dicc_preflight.py"
export NCD_VALIDATION_CODE="$PWD"
export NCD_WORK_ORDER="$root/job-a-bic14-rightsize/WORK_ORDER.md"
export NCD_OUTPUT_ROOT="/scr/user/$USER/classil-cse-wacb-score-guarded-20261102"
export NCD_PROTECTED_ROOT="$root/protected-cse-wacb-score-guarded-20261102"
printf '%s  %s\n' '42de11bc4e09eeffb0bbfb413104d40bb89a233238c7450b66b8db4b4d27a9d8' guard-cse-wacb-evaluation.sbatch | sha256sum -c -
bash -n guard-cse-wacb-evaluation.sbatch
"$NCD_PYTHON" -I -B "$NCD_PREFLIGHT" guard-cse-wacb-evaluation.sbatch --account-status full
sbatch --test-only --hint=nomultithread guard-cse-wacb-evaluation.sbatch
bash "$root/submit_checked_next.sh" guard-cse-wacb-evaluation.sbatch
