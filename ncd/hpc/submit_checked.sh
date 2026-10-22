#!/usr/bin/env bash
# Section116 single-submit wrapper; current repository/user authority applies.
set -euo pipefail
umask 077
test "$#" = 1
job="$1"
: "${NCD_PYTHON:?}" "${NCD_PREFLIGHT:?}" "${NCD_OUTPUT_ROOT:?}" "${NCD_PROTECTED_ROOT:?}"
test -z "${SLURM_JOB_ID:-}"
case "$(hostname -s)" in login*) ;; *) exit 77 ;; esac
account="$(account-info)"
printf '%s\n' "$account"
sed -E 's/\x1B\[[0-9;]*[mK]//g' <<< "$account" | grep -Eq 'Current Status[[:space:]]*:[[:space:]]*FULL'
cluster-info
association="$(sacctmgr -nP show assoc where user="$USER" format=User,Account,Partition,QOS,MaxTRES)"
printf '%s\n' "$association"
test "$association" = "$USER|free||long,normal,short|gres/gpu=2"
limits="$(sacctmgr -nP show qos short format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU)"
printf '%s\n' "$limits"
test "$limits" = 'short|01:00:00|gres/gpu=2||10'
queue="$(squeue --me -h -o '%i|%j|%T')"
printf '%s\n' "$queue"
# This initial twenty-cell matrix must not overlap any owned job.
test -z "$queue"
test ! -e "$NCD_OUTPUT_ROOT"
test ! -e "$NCD_PROTECTED_ROOT"
"$NCD_PYTHON" -I -B "$NCD_PREFLIGHT" "$job" --account-status full
if sbatch --help 2>&1 | grep -q -- '--test-only'; then
  sbatch --test-only --hint=nomultithread "$job"
else
  printf '%s\n' 'Client lacks --test-only; current project rules permit live/static checks.'
fi
test ! -e submitted.receipt
mkdir submit.lock
date -Is
sha256sum "$job"
jobid="$(sbatch --parsable --hint=nomultithread --no-requeue "$job")"
[[ "$jobid" =~ ^[0-9]+$ ]]
printf '%s\n' "$jobid" > submitted.receipt
printf 'SUBMITTED_JOB_ID=%s\n' "$jobid"
squeue --me -j "$jobid" -o '%i|%u|%j|%T|%M|%C|%R'
sacct -j "$jobid" -nP --format=JobID,User,JobName,State,Submit,Start,End,Elapsed,AllocTRES,ExitCode
