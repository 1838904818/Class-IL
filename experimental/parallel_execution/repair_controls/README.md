# One-stage score-level repair runner

Status: locally executable CPU research software, not validated real-data repair
utility. This package fits an actual two-parameter R1 calibration, computes
paired group-level simultaneous intervals and confusion counts, and persists
single-use decisions. It does not train an encoder, alter checkpoint weights,
extract real SHAP, establish historical GPU score fidelity, suppress original
alerts, or demonstrate novelty. Synthetic tests are software tests only.

## Source-only release allowlist

Include only `score_core.py`, `runner.py`, `make_demo.py`,
`adapt_native_exports.py`, `test_score_runner.py`, `requirements.txt`,
`README.md`, `PROTOCOL.md`, and `INPUT_CONTRACT.md`.
Exclude every SQLite database, `_test_scratch/`, `local_checks/`, `local_runs/`, generated demo/input/output
directory, bytecode file and local run receipt from a source release. No Git or
remote operation is part of this package preparation.

## Local commands

From this directory, with the locally tested Python 3.11.9 and NumPy 2.4.6:

```text
python -B -m unittest discover -s . -p test_score_runner.py -q

python -B make_demo.py --output local_runs/demo-input --seed 23 --groups 12
python -B runner.py decide --manifest local_runs/demo-input/manifest.json --policy local_runs/demo-input/policy.json --ledger local_runs/study.sqlite --output local_runs/results --study synthetic-demo-23
python -B runner.py evaluate --manifest local_runs/demo-input/manifest.json --policy local_runs/demo-input/policy.json --labels local_runs/demo-input/sealed-evaluation-labels.json --ledger local_runs/study.sqlite --output local_runs/results --study synthetic-demo-23
```

Use fresh output/input directories for demo generation. Preserve the SAME
ledger across increments. Repeating a study, reusing a row/group in another
study, reusing an acceptance partition, or evaluating twice is rejected, even
after restarting Python. A failed or interrupted attempt remains spent. Do not
delete the ledger or change identities to evade these controls.

The demo generator supplies random arrays, not prefilled repair outcomes. Its
numerical settings are explicit **prospective, unvalidated software-demo
choices**. They must not be copied into a real study as accepted thresholds.
The controlled test suite also contains deliberately artificial branch cases;
one uses a loose tolerance solely to exercise ACCEPT, while a separate ordinary
random demo can reject or abstain. Neither is evidence of useful repair.

## What executes

1. Strictly parse JSON, reject duplicate keys and nonfinite/overflowed numbers,
   reject symlink/reparse inputs, and validate the four-partition manifest.
2. Freeze the entire policy, source/runtime identity, row/group claims and all
   ten arm registrations in a durable SQLite transaction before loading labels.
3. Compute and persist each complete old-class candidate pool using trigger
   data only. All pools lock before any fitting or acceptance.
4. For an eligible arm, re-read hash-bound trigger data and recompute eligibility
   and ordering. Reserve one common R1 package before fit. Failed attempts are
   not refunded. An audit-only or abstaining arm reserves no repair.
5. Actually optimize R1 on fit labels only, with a fixed step count and projected
   bounds. Persist progress at every completed step. Freeze the fitted state
   before opening acceptance labels.
6. Compute baseline/proposed acceptance scores and group-level simultaneous
   intervals. Accept or roll back according to the frozen policy. Keep original
   scores/predictions and proposed outputs separately in either case.
7. Only after every arm decision locks, permit the separate `evaluate` command
   to open the sealed subsequent-label file. Report both proposed counterfactual
   and effective-policy outcomes without feeding them back into selection.

Primary controls are random, periodic, error-only, disagreement-only,
expansion-aware and audit-only. Additional arms are expansion-w0,
expansion-no-attribution, R1-reachability and expansion-R1-reachability. Simple
controls do not require attribution inputs. Missing attribution evidence blocks
the corresponding candidate arms; it does not invent SHAP or run it implicitly.

## Native checkpoint export adapter

The sibling native exporter produces `EXPORT_RECEIPT.json`, `COMPLETE.json`,
old/new independent head probabilities, native binary-logit margins, raw router
affinities and aligned row/group arrays. This adapter verifies those artifacts
and converts four separately specified partitions:

```text
python -B adapt_native_exports.py adapter-spec.json --output fresh-score-bundle
```

See [the input contract](INPUT_CONTRACT.md). The adapter reads no checkpoint and
performs no model inference. It never reconstructs raw affinities from z scores
or manufactures attributions. Real export completion becomes
`score-bundle-pending-fidelity`, not permission to run a real mechanism study.
Official test data cannot be assigned to trigger, fit or acceptance. A known
official test may be explicitly marked retrospective evaluation, but it must
not be described as previously unseen confirmation.

## Remaining boundary: real evidence and experiments

The fitter, simultaneous interval estimator, persistent single-spend ledger,
candidate-pool binding, rollback and subsequent evaluation are implemented.
The remaining work is evidence acquisition and actual registered experiments:

- Produce checkpoint-bound raw scores from authorized training-side partitions;
  close historical/native numerical fidelity and training-lineage validation.
- Establish method-specific faithful A/D attributions and group-independent
  repeated-draw provenance when testing attribution-dependent arms.
- Obtain authorized label/provenance/grouping evidence and sufficient genuinely
  independent groups. Freeze real thresholds and resources before looking at
  acceptance or subsequent results.
- Run independently reviewed, hash-bound experiments and measure upstream
  checkpoint/SHAP costs as well as the runner's own costs. Local synthetic
  execution does not discharge any DICC live-policy, review or confirmation gate.
- Test the hypotheses against simple controls, action-matching controls, the
  separate fixed-reference router control and relevant literature. No novelty
  or superiority over CLEX, Drift2Act or related expansion-routing work is claimed.

External evidence checks authenticate file content/binding, not a reviewer's
identity or truthfulness. They must be backed by genuine inspected reports, not
generated PASSED declarations. A trusted local filesystem and preservation of
the ledger remain required; this is not an adversarial security service.
