#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$HOME/classil-20261102"
cd "$root/cse-wacb-score-20261102"
export NCD_PYTHON="$HOME/.venvs/ofra-etg/bin/python"
export NCD_PREFLIGHT="$root/dicc_preflight.py"
export NCD_VALIDATION_CODE="$PWD"
export NCD_WORK_ORDER="$root/job-a-bic14-rightsize/WORK_ORDER.md"
export NCD_OUTPUT_ROOT="/scr/user/$USER/classil-cse-wacb-score-20261102"
export NCD_PROTECTED_ROOT="$root/protected-cse-wacb-score-20261102"
printf '%s  %s\n' 'c4ad9cb8d044b58057d3ac0e5f0e02ca12cb9ca101a922d624349bc36abd48ee' diagnose-cse-wacb-score-distributions.sbatch | sha256sum -c -
bash -n diagnose-cse-wacb-score-distributions.sbatch
"$NCD_PYTHON" -I -B "$NCD_PREFLIGHT" diagnose-cse-wacb-score-distributions.sbatch --account-status full
sbatch --test-only --hint=nomultithread diagnose-cse-wacb-score-distributions.sbatch
bash "$root/submit_checked_next.sh" diagnose-cse-wacb-score-distributions.sbatch
