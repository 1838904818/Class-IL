#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$HOME/classil-20261102"
cd "$root/nsl-remaining-score-20261102"
export NCD_PYTHON="$HOME/.venvs/ofra-etg/bin/python"
export NCD_PREFLIGHT="$root/dicc_preflight.py"
export NCD_VALIDATION_CODE="$PWD"
export NCD_WORK_ORDER="$root/job-a-bic14-rightsize/WORK_ORDER.md"
export NCD_OUTPUT_ROOT="/scr/user/$USER/classil-nsl-remaining-score-20261102"
export NCD_PROTECTED_ROOT="$root/protected-nsl-remaining-score-20261102"
export NCD_FROZEN_ROOT="$root/frozen"
export NCD_SKLEARN_ROOT="$root/linux311-sklearn"
export NCD_DIAGNOSTIC_SHA256="6cbb6eba765305e55b8fc0bd40b924f64625fbc66252d29935e99e7110c2d7a7"
printf '%s  %s\n' '15b958508522f11bd1b84cf3b553540def00dc4c62307aa6bfff9fa5b96ad833' diagnose-nsl-remaining-score-points.sbatch | sha256sum -c -
bash -n diagnose-nsl-remaining-score-points.sbatch
"$NCD_PYTHON" -I -B "$NCD_PREFLIGHT" diagnose-nsl-remaining-score-points.sbatch --account-status full
sbatch --test-only --hint=nomultithread diagnose-nsl-remaining-score-points.sbatch
bash "$root/submit_checked_next.sh" diagnose-nsl-remaining-score-points.sbatch
