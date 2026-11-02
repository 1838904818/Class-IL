#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$HOME/classil-20261102"
cd "$root/cse-wacb-startup-trace-20261102"
export NCD_PYTHON="$HOME/.venvs/ofra-etg/bin/python"
export NCD_PREFLIGHT="$root/dicc_preflight.py"
export NCD_VALIDATION_CODE="$PWD"
export NCD_WORK_ORDER="$root/job-a-bic14-rightsize/WORK_ORDER.md"
export NCD_OUTPUT_ROOT="/scr/user/$USER/classil-cse-wacb-startup-trace-20261102"
export NCD_PROTECTED_ROOT="$root/protected-cse-wacb-startup-trace-20261102"
printf '%s  %s\n' '2cf976e336f18b6877120cc539f782d09540eb64cc489bc7f16eab2799b6dadc' trace-cse-wacb-startup.sbatch | sha256sum -c -
bash -n trace-cse-wacb-startup.sbatch
"$NCD_PYTHON" -I -B "$NCD_PREFLIGHT" trace-cse-wacb-startup.sbatch --account-status full
sbatch --test-only --hint=nomultithread trace-cse-wacb-startup.sbatch
bash "$root/submit_checked_next.sh" trace-cse-wacb-startup.sbatch
