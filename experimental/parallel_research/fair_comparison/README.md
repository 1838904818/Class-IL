# Fair-comparison research preparation

**Research preparation only. Synthetic tests only. No new training, baseline
reproduction, resource profile, HPC upload or job submission is provided.**

This package supplies a prospective protocol, a source-grounded confound audit,
and a standard-library accounting validator. It does not change any existing
trainer, results, or checkpoint. A valid synthetic fixture is not experimental
evidence; the shipped real-study template is deliberately blocked.

Read [PROTOCOL.md](PROTOCOL.md), [CONFOUND_AUDIT.md](CONFOUND_AUDIT.md), and
[ACCOUNTING_SPEC.md](ACCOUNTING_SPEC.md). Source paths in the audit are relative
to their identified source tree; no private absolute paths are published.

From this directory:

```text
python -B -m unittest discover -s tests -v
python -B validator.py protocol.template.json
```

The second command must return exit code 2 with `BLOCKED`; all real run bindings,
phase plans and measured limits await selection or measurement. For a future
complete local evidence bundle:

```text
python -B validator.py protocol.json accounting.json --artifact-root evidence
```

Exit 0 means only `ACCOUNTING_CONSISTENT`: the declared hashes, counters, row
eligibility, memory inventory, limits, and registered matching assertions are
internally consistent. It is not scientific approval, proof that instrumentation
was honest/complete, a successful reconstruction, institutional permission, or
submission authority. Exit 2 fails closed. Inputs are read-only. The validator
never trains, downloads, starts services, or invokes Slurm.

Required next work: choose one dataset and the smallest contrast; freeze its
available-label stream and shared initial/checkpoint states; implement and
independently inspect instrumentation; obtain real bounded compute profiles;
then follow the project's fresh institutional review and exact-command human
confirmation gates. No executable job script or guessed resource request exists
here. TRIL3, adaptive-rank, and other literature-derived arms remain unimplemented
until separately adapted and validated.
