# Increment-available normal-cap comparison protocol

Date: 2026-09-07. Status: prospective follow-up **design and synthetic reference**.
No experiment execution, data derivation, upload, or submission is authorized
by this document. Historical D2 artifacts and claims remain separately named.

## 1. Estimand and scope

Question: under a fixed class-incremental stream and evaluation contract, how
does replacing an offline future-informed normal cap with a cap frozen using
only Task-0 training data change learning, retention, attack detection, and
resource cost?

This is prospective with respect to **class/cohort revelation**, not proof of
a chronological deployment or capture-independent split. It is not a claim
that removing future information improves accuracy, that the cap is optimal,
or that a sampler is novel. Report degradation and resource tradeoffs.

The minimal candidate is `task0_max_attack_fit`. Fix class order, training and
calibration source split, 10% calibration rule, normal-class definition,
stable source row IDs, seed schedule, feature contract, checkpoint-selection
rule and official evaluation support before candidate execution. A fixed-cap
alternative must have a recorded choice/rationale before the new campaign;
choosing a number after inspecting future counts does not make it prospective
merely because it is passed as a constant.

## 2. Information boundary and transition

At increment t, the selector receives only newly released source-train
cohorts for that increment and the immutable selections from earlier calls.
All rows have explicit source partition and availability metadata. It must
not receive future class identities/counts, official-test labels/features/
hashes/metrics, full-dataset manifests, or paths to those objects. Fixed
protocol definitions are allowed; their identity must not encode hidden
future/test data. Selection cannot control when a trusted producer releases
data: a governed, independently checked availability ledger is still needed.

For each available class c with n rows, calibration support is:

```text
k(c) = 0                                      if n <= 1 or fraction = 0
k(c) = min(n - 1, max(1, floor(n * 1 / 10)))   otherwise
```

Rank row IDs using SHA-256 of the fixed protocol domain, seed, stage, class ID
and row ID; take k(c) for calibration and use the remaining IDs as the fit
pool. Calibration and fitting partitions are deterministic and disjoint.
Their union with omitted normal-fit IDs equals the source cohort. The hash
ranking is a deterministic sampling mechanism, not a data-governance proof.

For the candidate, set C0 to the largest **Task-0** attack fit-pool count.
Freeze C0 forever. Select up to C0 normal fit rows using a separate hash-rank
stage. Retain all attack fit rows, with no replacement or minority
oversampling. Later cohorts cannot change an earlier calibration selection,
normal cap, or fitting selection. Task 0 must introduce exactly one declared
normal class. When it has no nonempty attack fit pool, stop this arm. The
separate fixed-cap arm can handle that setting if already declared; no
automatic fallback, silent zero cap or test-guided rescue is allowed.

Empty classes retain zero support and a warning; singleton classes retain
one fitting row and no calibration row. Two rows split into one fitting and
one calibration row. Do not claim class-specific calibration validity when
support is absent or too small. IDs must be globally unique across classes
and increments. Reject ID collisions, old-class rewrites, nonconsecutive
increments, future rows/cohorts, test inputs, invalid roles and malformed
train-only provenance hashes.

The reference accepts IDs, not actual arrays. Source digests are logged in
selections but never used as RNG seeds. A future ingestion layer must prove
that IDs and digests are derived only from the declared training cohort, not
from a whole-manifest digest, and must verify source bytes. Duplicate flow or
capture leakage needs content/group-aware checks beyond duplicate IDs.

## 3. Isolate cap availability from sampler changes

Changing the legacy seed/split generator and the cap simultaneously is not a
cap-only experiment. Use this minimal bridge before causal language:

| Arm | Normal cap | Sampling and calibration RNG | Interpretation |
| --- | --- | --- | --- |
| H | Historical D2 offline all-attack maximum | Archived whole-manifest-seeded permutation | Immutable historical reference |
| O | Same offline D2 cap (124,780 on the bound source) | New prefix-local hash ranks | Offline bridge, **not prospective** |
| P | Frozen Task-0 attack-fit maximum | Identical new prefix-local hash ranks | Candidate information boundary |
| F, optional | Independently predeclared fixed cap | Identical new prefix-local hash ranks | Separate fixed-budget comparator |

H versus O identifies the effect of the changed sampling/calibration
realization only when all other bound settings are matched. O versus P
changes cap availability/value with exactly the same new sampler, all attack
IDs and calibration IDs. Normal selections are nested under the same ranks.
The test suite checks this nested-selection contract on synthetic rows.
Do not pass the offline cap off as an independently predeclared budget. Arm O
is a scientific control whose design provenance explicitly includes offline
counts; it never gives the candidate access to future metadata. None of H,
O, P or F is newly trained or selected as a winner here.

All methods in each data arm use the same cohort identities, support, class
order, seeds, transform contract and evaluation rule. Keep the new sampler
fixed across O/P/F. Do not tune caps, thresholds or checkpoints on official
test performance. Statistical claims require the registered repeated-seed
comparison and uncertainty estimates, not the current seed-42 count audit.

## 4. Exposure and resource fairness

Historical count arithmetic illustrates why cost cannot be ignored. On the
saved D2 audit, Task-0 attack fitting support is 5,559, the offline normal cap
is 124,780, and total attack fit support is 143,917. Keeping the same support
counts and 10% calibration rule would yield 149,476 fit rows for the frozen
Task-0 count versus 268,697 historically: 119,221 fewer normal fit rows.
These are **count-only calculations**, not newly derived row hashes, training
results, a compute measurement or a guaranteed production output. The new
RNG also changes row membership relative to H.

Report both a fixed-training-recipe comparison and a separately declared
budget/exposure-matched control where feasible. Equal epochs or replay
capacity alone do not match computation. Record, per increment and class:

- source, calibration, selected fit and omitted supports, with row-ID hashes;
- unique old/new rows available and actually used, minibatch draws, repeat
  counts, replay admissions, normal-negative usage, and class weighting;
- optimizer updates, examples per update, replay/new and normal/attack ratios,
  batch size, losses, learning-rate schedule, early stopping and checkpoint;
- preprocessing/load/train/prediction time, allocated and peak memory, model
  bytes, replay bytes, and measured CPU/GPU utilization where applicable.

If a matching rule repeats rows or limits new-class examples, disclose that
intervention; do not call it identical data exposure. Match or bound examples,
steps and bytes explicitly, or report unmatched dimensions as limitations.
Keep Task-0 transform policy identical. Holding the policy fixed can still
change fitted scaler parameters when selected Task-0 rows change; either use
one declared, common Task-0 train-only transform-fit cohort in O/P, or label
scaler-parameter change as part of the policy effect. Never use calibration
or test rows to fit the transform. The exact transform-input cohort is a
required pre-execution design choice, not silently resolved by this reference.

Report all seeds, per-class support/confusion counts, accuracy, Macro-F1,
balanced accuracy, signed forgetting, attack recall and FPR when the dataset
has valid attack semantics. Include old-class retention and new-class
learning. No learning-effect or efficiency claim follows from local tests.

## 5. Official test and history preservation

The reference has no test-set parameter and performs no file I/O. Therefore
it neither modifies nor verifies any real test bytes. A production evaluator
must retain the complete original official-test support and class mapping,
independently bind original shard hashes, and verify bytes before evaluation.
No test resampling, calibration borrowing, balancing, threshold selection or
checkpoint selection is allowed. Any copying/materialization belongs to a
separate verified stage that cannot influence the selector's RNG.

Use new versioned output directories with exclusive creation. Never overwrite
historical D2 manifests, fitting/calibration arrays, audits, checkpoints or
results. A future writer must fail on an existing destination, preserve
source hashes, and write an auditable selection/availability ledger. The
pure reference does not supply a production filesystem writer.

## 6. Gates before real-data derivation or HPC work

1. **Local design gate:** synthetic tests and source hashes pass; approve the
   availability ledger, stable IDs, cap arm, O/P bridge, transform-fit cohort,
   training-exposure ledger, trainer interface and source/data authorization.
2. **CPU derivation candidate:** implement and separately review a production
   train-only reader and immutable writer; bind code, dependency, source and
   protocol hashes. Profile only in an authorized suitable allocation. Large
   preprocessing never runs on the login node or this local reference.
3. **DICC submission gate:** reread project rules and current official/live
   DICC evidence, use measured resources, pass local and remote preflight,
   obtain the dedicated independent exact-hash `VERDICT: APPROVED`, then show
   the exact sbatch path, hash, resources and command for one fresh user
   confirmation. No batch-chain or automatic second-stage submission.
4. **Derived-data gate:** verify the actual completed job, source and produced
   train/calibration/official-test bindings, availability/prefix-invariance
   checks, disjointness, immutable output closure, resource audit and backups.
   A local synthetic test or predicted hash is not remote derivation evidence.
5. **Training candidate:** only after that gate, separately bind new dataset
   manifests, matched comparison configuration, environments and measured
   resources. Numerical score reconstruction and attribution-fidelity gates
   remain dependencies for mechanism claims; this sampler does not clear them.
   Obtain new independent review and exact-command user confirmation.

Current status is gate 1 preparation only. No live DICC limits, remote source
state, submission readiness, completed derivation or model result is claimed.
