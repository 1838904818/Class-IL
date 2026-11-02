#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$HOME/classil-20261102"
cd "$root/same-score-validation-20261102"
export NCD_PYTHON="$HOME/.venvs/ofra-etg/bin/python"
export NCD_PREFLIGHT="$PWD/dicc_preflight_section344.py"
export NCD_VALIDATION_CODE="$PWD"
export NCD_WORK_ORDER="$root/job-a-bic14-rightsize/WORK_ORDER.md"
export NCD_OUTPUT_ROOT="/scr/user/$USER/classil-same-score-validation-20261102"
export NCD_PROTECTED_ROOT="$root/protected-same-score-validation-20261102"
export NCD_INPUT_ARCHIVE="/scr/user/$USER/classil-same-score-inputs-20261102/section344-inputs-20261102.tar"
export NCD_INPUT_ARCHIVE_SHA256="ea7981c5799f674562a0e3a7e2b0f72d31c37f705ee6c59f3d9bf45a5cbc9263"
export NCD_SMALL_DATA_ARCHIVE="/scr/user/$USER/classil-same-score-inputs-20261102/section344-small-data-20261102.tar"
export NCD_SMALL_DATA_ARCHIVE_SHA256="74d4139b829eca96d205001e9fe6e52d15066c719854d84ac35d853f88cc5c03"
export NCD_SAME_SCORE_SOURCE_SHA256="3fe63c8f404d5217a6c63108e7169132bab7b75751a03146d17038ecc856ca2a"
export NCD_BIC_VERIFIER_SHA256="eb9e42e5816a1bb6640574b940048bb485f8374e9926435622eebfe5962b1d1d"
test "$(squeue -h -j 493652 -o '%u')" = "$USER"
printf '%s  %s\n' 'e9c016f753a2d89b3574537ac7b0f955939e92a038b3b54198903781d4da2a79' check-section178-same-score.sbatch | sha256sum -c -
printf '%s  %s\n' '76a436481f09fb8e4f2a9e02a6283bed0fb2149e78b12e3a38e6510f773199b2' dicc_preflight_section344.py | sha256sum -c -
bash -n check-section178-same-score.sbatch
"$NCD_PYTHON" -I -B "$NCD_PREFLIGHT" check-section178-same-score.sbatch --account-status full
sbatch --test-only --hint=nomultithread check-section178-same-score.sbatch
bash "$root/submit_checked_next.sh" check-section178-same-score.sbatch
