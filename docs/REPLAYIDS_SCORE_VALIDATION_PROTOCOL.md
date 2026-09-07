# ReplayIDS score validation: staged prospective follow-up

Execution update: [Stage A canonical CPU check](REPLAYIDS_CANONICAL_FIDELITY.md)
completed all 20 checkpoints and failed the cross-environment fidelity gate.
The matched historical GPU check and later candidate stages remain outstanding.
The [allocation-only GPU verifier and input binding](REPLAYIDS_GPU_FIDELITY_CANDIDATE.md)
are now prepared locally; their operational launcher and resource evidence are
not yet complete and no GPU result is implied.

Version 1, fixed 7 September 2026 before any new full-test candidate evaluation.
This is retrospective follow-up motivated by existing benchmark/probe diagnostics,
not a preregistration of the historical experiments. No candidate is promoted by
this document. The original last-epoch primary remains unchanged.

## Inputs now available

The [checkpoint coverage register](../results/replayids-checkpoint-coverage/CHECKPOINT_COVERAGE.json)
verifies all five seeds (1, 2, 3, 4, 42) and checkpoints 000-003: 20 snapshots and
60 state/manifest/probe-score files. State files total 349,397,744 bytes. Each
checkpoint's saved Joint formula and argmax were checked elementwise, including
sample coordinates, sample IDs and the seen-class axis. No new forward inference
or forgetting value was produced by this coverage audit.

| Checkpoint | Seen class IDs | Frozen probe rows per seed |
|---|---|---:|
| 000 | 0, 1 | 256 |
| 001 | 0, 1, 2, 3 | 512 |
| 002 | 0, 1, 2, 3, 4, 5 | 768 |
| 003 | 0, 1, 2, 3, 4, 5, 6, 7 | 898 |

All snapshots use the same registered source/data/model lineage. Repeated seeds
reuse fixed data and probes; they do not multiply the number of independent rows.
The earlier input-readiness JSON intentionally remains a historical final-only
snapshot, superseded for coverage by this new register.

## Stage A: legacy fidelity, no algorithm change

Objective: reproduce the registered scoring path, not maximize agreement by trying
different batches or tolerances. Recover configuration from the bound protocol.
The original `eval_batch_size` is **512**: `_embed` splits normalized rows at that
size, then the monitor scores Head and Router over the entire checkpoint probe
matrix. The prior batch32 and batch898 experiments are numerical stress tests,
not this canonical encoder batching path.

- Load the exact original runtime/checkpoint files, with their protected hashes.
- Use frozen Task-0 float64 normalization statistics and original float32 encoder,
  positive-probability Head and float32 Router squared-norm expansion.
- Compare all 20 checkpoints on their own seen-class probes and original class axis.
- Preserve score tolerance `atol=1e-4`, `rtol=1e-5`; also report bit-exactness.
  A tolerance pass requires every Head/Router/Joint cell within tolerance and zero
  argmax mismatch. Report failures without relaxing this gate.
- Record raw score hashes, prediction-vector hashes, mismatch identities and
  top-two margins. Equal mismatch counts alone do not prove identical row sets.
- First test the canonical batching path locally and label its environment
  difference. A matched historical GPU/software run remains a separate check.
- Match or explicitly flag Python, torch, CUDA, NumPy, upstream module/dependency
  versions, deterministic settings, device and numerical-kernel settings. Do not
  call an environment matched solely because the GPU model matches.
- No training, calibration, threshold selection or new performance promotion.

## Stage B: separately versioned arithmetic candidate

Candidate identifier: `joint_cap3000_direct64_v1`. This is NOT an alias for the
registered `joint_cap3000`. Keep all learned parameters, Task-0 statistics,
centroids, task order, seeds, checkpoint selection and full official test unchanged.

Fixed arithmetic contract:

1. Normalize raw features using the stored float64 mean/scale; cast to float32.
2. Encoder batch size is 512; use the original encoder and family heads in eval
   mode without fitting. Head score remains the two-logit positive softmax in
   float32. Use sorted seen-class IDs and never include a future class.
3. Promote embeddings and stored centroids to float64. Compute direct coordinate
   differences and sum their squared values in float64. Use the nearest centroid
   and negative square root. Do not silently use the old squared-norm expansion.
4. Across seen classes, subtract the per-row mean and divide by population standard
   deviation plus `1e-8`, all in float64. Cast the resulting Router score to float32.
5. Compute Joint as float32 `head + float32(0.5) * router_z`; argmax returns the
   first maximum in sorted class order. No new rejection or tie-abstention rule.
6. Bound pairwise scratch allocation with row blocks of 128 and centroid blocks
   of 16. Verify block independence against an independently implemented direct
   float64 reference before evaluation. Blocking is computational, not sampling.

Promotion is not automatic. First validate finite values, self/near-equal distances,
class order, ties, batch/centroid block sensitivity, score tolerances and per-row
prediction agreement. Record discrepant identities, not only aggregate counts.
The present direct64 diagnostic supports further investigation but does not prove
that this complete candidate meets these requirements.

## Stage C: full fixed-test comparison, only after executable gates

Evaluate all seen-class official-test rows at every checkpoint for all five seeds.
No balancing, subsampling or exclusion of difficult/rare test classes. Preserve the
legacy primary and report the new candidate separately. Reuse the same source
records to compare Head-only, Router-only and Joint when those arms are executed;
do not substitute unrelated checkpoints or use a selected guarded policy as primary.

Required outputs are per-checkpoint and final accuracy, average task accuracy,
Macro-F1, balanced accuracy, per-class precision/recall/F1/support and confusion
counts, signed forgetting under the existing definition, attack recall and Benign
FPR. Preserve signed improvements rather than silently clipping negative forgetting;
any clipped variant must be labelled secondary. Include row-level prediction hashes,
scoring-path/config/code/data/checkpoint hashes, elapsed time and peak resources.

Report all paired seed results, mean/standard deviation and uncertainty under the
registered analysis convention. Distinguish seeds 1-4 from historically exploratory
seed42 when relevant. Exact two-sided signed-rank inference at n=5 has limited
resolution; do not manufacture significance or select a favorable test afterward.
Predeclare comparison families and multiplicity handling if inferential claims are
planned. No superiority, Pareto improvement or best-model declaration follows from
improved numerical stability alone. Report trade-offs across all requested metrics.

## Stage D: training-only calibration and explanation/governance

This stage is not yet executable. Define its fitting objective and support policy
separately before examining candidate test metrics. At each increment use only
calibration rows from classes seen so far. Never fit on test rows or future-class
calibration rows. Heartbleed has one calibration row; repeated seeds do not cure
that support limitation. Keep an explicit uncalibrated comparison.

Any explanation/ETG follow-up must identify the actual versioned score being
explained and revalidate wrapper fidelity for that score. Numerical mismatch is
not SHAP drift or attribution-method disagreement. Do not reuse old ETG decisions
as if they certify a changed scoring rule. The outstanding attribution robustness,
ledger integrity and governance-effect questions remain within the research goal.

## Execution and evidence boundaries

Local synthetic tests and read-only input retrieval do not authorize HPC execution.
Any HPC candidate must bind executable files, environment, data, checkpoint and
protocol hashes; use measured resources; run only in an allocation; pass independent
review and obtain fresh exact-command user confirmation. Do not launch an automatic
multi-stage submission chain. Never burn repeated dummy work to inflate utilization.
W&B may log approved non-sensitive aggregates only; protected evidence must retain
the exact output hashes. No credential or raw traffic belongs in a public repository.
