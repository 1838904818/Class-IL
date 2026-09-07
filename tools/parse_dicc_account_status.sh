#!/usr/bin/env bash
# Pure stdin parser. No account query, credential access, or remote mutation.
set -euo pipefail
clean="$(sed -E $'s/\x1B\\[[0-9;]*[[:alpha:]]//g' | tr -d '\r')"
if grep -Eqi '\b(SUSPENDED|DISABLED|EXPIRED|INACTIVE)\b' <<<"$clean"; then
  echo 'Refused: account status is not active.' >&2
  exit 77
fi
mapfile -t statuses < <(sed -nE 's/^[[:space:]]*Current Status[[:space:]]*:[[:space:]]*(.*)$/\1/ip' <<<"$clean")
[[ "${#statuses[@]}" == 1 ]] || {
  echo 'Refused: expected exactly one Current Status field.' >&2
  exit 77
}
status="$(printf '%s' "${statuses[0]}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
case "${status^^}" in
  FULL) printf 'full\n';;
  LIMITED) printf 'limited\n';;
  *) echo 'Refused: unrecognized Current Status value.' >&2; exit 77;;
esac
