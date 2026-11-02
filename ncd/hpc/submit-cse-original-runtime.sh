#!/usr/bin/env bash
set -euo pipefail
umask 077
old="$HOME/classil-20261102"
cd "$HOME/classil-20261103/cse-original-runtime"
export NCD_PYTHON="$HOME/.venvs/ofra-etg/bin/python"
export NCD_PREFLIGHT="$old/dicc_preflight.py"
export NCD_WORK_ORDER="$old/job-a-bic14-rightsize/WORK_ORDER.md"
export NCD_OUTPUT_ROOT="/scr/user/$USER/classil-cse-original-runtime-20261103"
export NCD_PROTECTED_ROOT="$HOME/classil-20261103/protected-cse-original-runtime-20261103"
export NCD_RUNTIME_SOURCE_SHA256="316e8795a90b9848bfea6c79315c8853ed593ba73c6dfbad746b038f24fc17a6"
export NCD_DEFAULT_REFERENCE_SHA256="36f486daa6a7afe9f94fc1c95a50e233988492e921a1b9d63b432d7de7179b47"
printf '%s  %s\n' "$NCD_RUNTIME_SOURCE_SHA256" reevaluate-cse-original-runtime.py | sha256sum -c -
printf '%s  %s\n' '340a7382fee559492f1d4103f6bf47b81923b1bcae1e74e8d0841e61cf663539' reevaluate-cse-original-runtime.sbatch | sha256sum -c -
bash -n reevaluate-cse-original-runtime.sbatch
"$NCD_PYTHON" -I -B "$NCD_PREFLIGHT" reevaluate-cse-original-runtime.sbatch --account-status full
sbatch --test-only --hint=nomultithread reevaluate-cse-original-runtime.sbatch
bash "$old/submit_checked_next.sh" reevaluate-cse-original-runtime.sbatch
