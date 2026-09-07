# Account-status validation regression

7 September 2026. This is execution-preparation evidence, not a model result.

## Observed failure

The v5 fidelity-profile package was transferred and all 15 remote file hashes
matched its manifest. The validation-only guard then exited with code 77 while
reading an active FULL account status. The status was printed with ANSI colour
controls; an earlier word-boundary search treated the preceding colour suffix
as part of the word. No calculation was submitted or started. This was a
parser failure, not an expired account, numerical-fidelity failure or GPU job
failure. The uploaded v5 package is preserved unchanged.

## Local correction

The v6 guard verifies its execution manifest before invoking a bound, pure
stdin parser. The parser strips ANSI controls and carriage returns, requires
exactly one `Current Status` field, and accepts only the complete values FULL
or LIMITED. Missing, duplicate, conflicting, unknown or inactive statuses fail
closed. The parser neither queries an account nor grants submission authority.
Live account limits, partition/QoS checks, reviewed hashes and separate user
approval remain necessary.

The new parser has 16 passing regressions, including the observed coloured
FULL output, coloured LIMITED/CRLF, duplicate and conflicting fields, inactive
notices and a reproduction of the previous failure. Together with 23 existing
coordinator tests, local results are 38 PASS and one Windows symlink SKIP.
Static preflight and Bash syntax validation passed. Subsequently the exact
v6 guard also passed actual live validation, including coloured FULL status,
partition membership, own-account QoS and a 600-second request within the
259,200-second account limit. This client lacks the optional scheduler
test-only capability; the compatibility notice is preserved.

The candidate contains 17 bound files. It changes no scientific verifier,
checkpoint, input binding, coordinator, score tolerance or resource request.
Its GPU inference profile, resource suitability and remote tracking remain
unmeasured. No scientific review item is closed by these operational tests.

## Evidence identifiers and remaining gate

| Item | SHA-256 |
|---|---|
| Uploaded v5 sbatch | `49edb9f53c5527047ea5223b7f91b033fc3f747275ab11c58abbba73c6160fa2` |
| Uploaded v5 package manifest | `cddbd69727110f5c3ad99ecbb1ea83c9fcfc7fe3a020be34ca6c2b64223692ef` |
| Uploaded v6 sbatch | `b2eab1a663ddaad8643dec5e971cfc830a361ffc30671c02a4ec6644bbea1aa5` |
| Uploaded v6 package manifest | `0c1fe1ae0ba689466934d2ceba79d8b10ca52d2936ba9c4992ecd8edbf1fd4f7` |
| Pure parser | `a79089cfe79f971e88dd1bc973c85db6ea85ef1302b3dd9cf1c4982fce49946d` |

Authorized v6 staging is complete. All 17 remote operation files and all 91
bound input files match, the output parents are owned and empty, and the
guard explicitly returned that no job was submitted. See the
[redacted staging evidence](../results/replayids-gpu-fidelity/V6_STAGING_VALIDATION.json).
Independent review approved the exact v6 candidate. Fresh exact-command user
confirmation remains required; any relevant bound-file or live-state change
invalidates that approval. This is not a numerical-fidelity or Slurm execution result.

To reproduce the pure tests without a cluster connection:

```text
python -B -m unittest discover -s tests -p test_dicc_account_status.py -v
```

Bash, sed, grep and tr are required; Windows uses Git for Windows Bash.
