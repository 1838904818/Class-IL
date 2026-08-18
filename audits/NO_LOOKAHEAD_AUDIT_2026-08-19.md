# No-look-ahead preprocessing audit

Audit date: 19 August 2026
Scope: the five registered formal caches and the current `streaming_full`
normalisation implementation.

## Result

The numerical normalisation contract passes. `FrozenTask0Stats` reads only the
Task-0 training shards, freezes before Task-0 pretraining, and rejects any
subsequent statistics update. Test rows and later-task rows do not contribute
to the numerical mean or population-standard-deviation estimate.

The fixed one-hot schema has a narrower limitation. For NSL-KDD and UNSW-NB15,
the categorical vocabulary is collected from the complete official training
partition before the class-incremental stream starts. The early model does not
observe future row values, but the existence of some later-task-only categories
is represented by columns in the initial feature schema.

| Dataset | Current width | Task-0-only vocabulary width | Future-only columns | Later-task rows carrying one or more such values |
|---|---:|---:|---:|---:|
| NSL-KDD | 122 | 116 | 6 | 115 / 12,703 (0.905%) |
| UNSW-NB15 | 194 | 186 | 8 | 712 / 79,341 (0.897%) |
| CIC-IDS-2017 | 78 | 78 | 0 | not applicable |
| CSE-CIC-IDS2018 | 78 | 78 | 0 | not applicable |
| MalayaNetwork_GT | 77 | 77 | 0 | not applicable |

For NSL-KDD, the later-task-only values are five `service` values (`aol`,
`harvest`, `http_2784`, `http_8001`, and `pm_dump`) and the `RSTOS0` flag. For
UNSW-NB15, they are seven `proto` values (`argus`, `bbn-rcc`, `crtp`, `egp`,
`hmp`, `netblt`, and `rdp`) and the `irc` service.

## Interpretation and reporting boundary

This is a bounded transductive schema limitation, not row-level test leakage
and not future-label training. Nevertheless, the complete preprocessing
pipeline must not be called strictly no-look-ahead or fully oracle-free. A
strict replacement would build the vocabulary from Task 0 only, or use an
externally fixed vocabulary plus an `unknown` category, and then rerun affected
NSL-KDD and UNSW-NB15 experiments because their feature widths change.

The machine-readable result is
`results/audits/no_lookahead_audit_20260819.json`. Its canonical SHA-256 is
`97c366b3b05ab85dc1f07451600ff70ea540f0155376aa6741ee1f69c5e0757e`.

## Reproduction

```text
python scripts/audit_no_lookahead.py \
  --cache-root <formal-cache-root> \
  --project-root . \
  --output results/audits/no_lookahead_audit_20260819.json
```

The script verifies the bound raw-training-file hashes before calculating the
categorical counts. It does not modify the raw data or cached features.
