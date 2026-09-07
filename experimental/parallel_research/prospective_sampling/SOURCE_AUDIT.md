# Source audit and boundaries

Read-only inspection on 2026-09-07. Paths below are workspace-relative source
identifiers, not required dependencies of the standalone reference. Only
small source, protocol and JSON metadata files were inspected/hashed; no
source or derived `.npy` contents were loaded for this package.

## Exact bindings

| Source | SHA-256 |
| --- | --- |
| `work/replayids_score_readiness_20260907/build_train_protocol_v2.py` | `5762c7362dd6905e97e9407c02cf120e200f83c4f054351917e006deb388df75` |
| `work/research_directions_20260827/build_train_protocol.py` | `e6bacab1659b1be09eee169f2b35be86639e82b7ccf5b031e7d7b3c323fad1c4` |
| `work/research_directions_20260828/derived_replayids_normalcap_seed42_v2/sampling_audit.json` | `cef4214ec14fdaabbff08bb7bbd142d35840cb2757b89c1439a3840c3782c746` |
| `work/research_directions_20260828/derived_replayids_normalcap_seed42_v2/streaming_manifest.json` | `a89e981725e4384f64bcc9c9329ac84318c7d8045012e166211388b774597e27` |
| `work/hpc_transfer_20260730/operations/20260828-replayids-data-model-v2/DATA_PROTOCOL_V2_DICC.md` | `6bf94babf121c9493f5f8be07a5b29f04d4f7621b6a4f70339a497fd06394b40` |
| `work/review_remediation_20260907/EXPERIMENT_GATES.md` | `82edc233d1bd84ccb33a6a6616ba5fcc3b13f7f1366873da563cffa5733f257e` |
| `work/review_remediation_20260907/TECHNICAL_AUDIT_SUPPLEMENT.md` | `3785dc3764b66b7c357f3843d1e8054fc45ba81e9d0fbd9e3b9babc50779716e` |

The historical audit's builder digest exactly matches the existing readiness
copy above, and its helper digest exactly matches the 20260827 helper. The
20260828 research directory currently contains the derived audit/manifest,
not a builder source file. This content-hash match identifies the inspected
builder without treating a missing historical location as present.

## Exact code evidence

In the hash-bound `build_train_protocol_v2.py`:

- Lines 37-39: `derived_seed` includes `source_sha256` in the seed payload.
- Line 82: `source_sha` is computed from the complete source manifest.
- Lines 87-117: the preparation loop iterates **all** `classes`, resolving
  training and test shards and constructing every class's fit pool.
- Lines 93-99: test metadata belongs to the source structure, and per-class
  random permutations use the whole-manifest-derived seed.
- Lines 119-125: `attack_fit_reference` is the maximum of all prepared
  non-normal fit-pool sizes, without an increment-availability restriction.
- Lines 134-140: only the normal class is capped; attack fit pools are kept.
- Lines 262-278: audit configuration records this cap source and the
  train-calibration, attack-retention and official-test invariants.

The saved manifest declares tasks `[0, 1]`, `[2, 3]`, `[4, 5]`, `[6, 7]`.
The saved audit identifies class 1 (DoS GoldenEye) with 5,559 fit rows and
class 2 (DoS Hulk) with 124,780 fit rows; the normal cap is 124,780. These are
historical metadata observations, not rederived array evidence.

The full source-manifest hash quoted in that historical audit is
`99f6de7a6cdd09b91e9bc0e167304db4d4274e753adad5138ec7803f5338a15b`.
It is a recorded source binding, not independently rehashed here. The audit
also records byte-identical test preservation; actual official-test bytes
were not rechecked during this preparation.

## Interpretation limits

The cap has offline fitting-count lookahead. The seed has full-manifest
metadata coupling. Neither observation alone proves fitting on official-test
labels, test-metric optimization, or future-feature normalizer fitting.
Task-0-only numerical transforms remain a distinct requirement; this package
does not inspect or reproduce the numerical training implementation.

The original artifacts are not invalidated or silently replaced. Their
offline construction must be accurately labeled. The new prefix-local
sampler is not numerically identical to the old permutation generator, so
its effects must be separated using the O/P bridge in `PROTOCOL.md`.
