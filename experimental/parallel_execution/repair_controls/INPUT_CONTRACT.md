# Bound score and native-export contracts

## Direct runner bundle

`manifest.json` has schema `score-repair-bundle-v1`, a nonnegative integer
`increment`, unique nonempty string `classes`, a strict subset `old_classes`
with at least two members, `source_kind`, and exactly four `partitions`.
Each partition's metadata includes:

```text
file, sha256, row_ids, group_ids, observed_increment, labels_available_increment
```

Row/group order is authoritative. A row is unique, and groups cannot cross
partitions. Repeated group IDs inside one partition are allowed. Trigger, fit
and acceptance labels must be available by the current increment. Subsequent
observation is later and its label-availability field is null. A separate
`sealed_labels_sha256` binds the future labels without placing them in the
decision manifest. Paths are relative descendants, never symlinks, Windows
reparse points, alternate data streams, traversal paths or absolute paths.

Each score-partition JSON has:

```text
role, row_ids, group_ids,
new_head_logits [N,K], new_router_raw [N,K],
old_head_logits [N,Kold], old_router_raw [N,Kold]
```

All score arrays are finite numeric matrices, not boolean or string arrays.
The native head quantity is the independently scored two-logit family margin
`positive_logit - negative_logit`, NOT a multiclass-normalized probability.
Old columns follow `old_classes`; new columns follow `classes`. Routers are raw
pre-standardization affinities, NOT z scores. Do not infer missing raw arrays
from archived normalized values. Missing old raw arrays explicitly prevents
A/B/C/D diagnosis; it does not become a fabricated state effect.

The first three partitions additionally contain `labels`, one registered class
ID per row. Subsequent score JSON must not contain labels. Its separate sealed
file has `row_ids`, `group_ids`, `labels` and `observed_increment`, all checked
against the registered subsequent partition after decision lock.

Optional trigger attribution data is:

```text
explanations[class_id] = {
  method: registered method name,
  independent_draws_by_group: true,
  A: [draw, N, feature],
  D: [draw, N, feature]
}
```

There must be an even number of draws, at least four, finite nonzero vectors,
the same method, row/feature dimensions, and a valid group-independent draw
design. The runner computes distances/noise itself, not supplied outcome
intervals. Native score export does not provide these arrays. No synthetic
vector or absent array is substituted for real SHAP.

## Native adapter specification

The adapter accepts one JSON with EXACT top-level fields:

```text
schema: native-score-repair-adapter-v1
increment: integer
probability_margin_tolerance: positive prospective numerical tolerance
partitions: {trigger: entry, fit: entry, acceptance: entry, subsequent: entry}
```

Each entry has exactly:

```text
receipt: {path: relative EXPORT_RECEIPT.json path, sha256: file hash}
labels: {path: relative independent label JSON path, sha256: file hash}
origin: increment-available-training | prospective-heldout | retrospective-official-test
observed_increment: integer
labels_available_increment: integer or null for subsequent
```

The first three origins MUST be `increment-available-training`; official test
data cannot become trigger/calibration/acceptance data by relabeling its role.
The separate label JSON has exactly `row_ids`, `group_ids`, `labels`, `origin`.
Its origin must agree with the specification. These claims need actual
provenance/governance evidence for real inputs; a field name cannot prove origin.

Every native receipt must be COMPLETE, match its `COMPLETE.json`, and bind all
nine arrays: `old_head`, `old_head_logits`, `old_raw`, `old_row_ids`, their four
`new_` counterparts, and `group_ids`. `.npy` loading forbids pickle. Array hashes,
dtypes, dimensions, probabilities, row alignment, dataset/seed, checkpoint
chronology and identical checkpoint identities across the four receipts are
checked. The probability/margin consistency tolerance checks only the local
conversion contract; it is not a historical GPU numerical-fidelity certificate.

The adapter writes score partitions, a separate sealed label file, a manifest
and `ADAPTER_RECEIPT.json`. It computes no model or attribution. Real receipts
produce `score-bundle-pending-fidelity`; they cannot run until genuine external
evidence is supplied. The adapter preserves the native receipt's explicit
unverified historical-fidelity/lineage flags in provenance.

## External evidence required for real execution

A real manifest may use `externally-verified-score-bundle` only with actual
`score_fidelity_evidence` and `data_governance_evidence` references, each an
object `{file, sha256}`. Legacy truthy digest strings are not accepted.
If real attribution arrays are supplied, an additional
`attribution_fidelity_evidence` reference is mandatory.

Each bound evidence JSON must have the corresponding schema:

- `score-fidelity-evidence-v1`, status `PASSED`;
- `data-governance-evidence-v1`, status `APPROVED`;
- `attribution-fidelity-evidence-v1`, status `PASSED` when applicable.

All carry `source_binding_sha256` and a bound `report: {file, sha256}`.
`source_binding(manifest)` hashes the class axes, increment, all partition/data
identities, the sealed-label digest and native receipt provenance. Evidence
references and source-kind promotion are outside that digest to avoid a
circular self-hash. The underlying report must itself be a JSON
`external-research-review-report-v1` with the same status/source binding and a
nonempty `review_id`.

The score report must bind `fusion_sha256`, verify numerical fidelity and
training lineage. The governance report must bind `semantics_sha256`, affirm
`decision_partition_source: increment-available-training` and
`official_test_used_for_decisions: false`. An attribution report must bind the
method and verify the faithful target. These external files and underlying
reports are rehashed before acceptance and before future labels can be opened.

These checks validate artifact integrity/scope, not authenticity of a reviewer
or truth of an assertion. Do not write PASSED merely to satisfy the parser.
The actual numerical/governance/attribution evidence must be reviewed separately.
The local native exporter currently does not establish historical GPU fidelity
or training lineage, and this adapter must not upgrade those flags on its own.

## Frozen policy and recorded state

`policy.json` has `score-repair-policy-v1` and
`scientific_status: prospective-not-validated`; every field in trigger, fusion,
R1, acceptance, uncertainty, budget and semantics is explicit. Unknown fields
inside these sections are rejected. The generated demo policy documents a
complete schema instance, not scientifically recommended values.

The ledger binds canonical manifest/policy hashes and a source/runtime hash
covering both runner modules, Python version, NumPy version and float64
arithmetic. Candidate pools, reservations, progress, fitted state and decisions
are persisted. Changing a bound source, policy, file, runtime or evidence
invalidates later stages. The durable ledger is not a signed adversarial log;
keep its storage trusted and never reset it to reuse held-out observations.
