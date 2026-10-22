# Section 81: ReplayIDS source and B1 adaptation

Recorded before B1 training. This is the single adaptation record requested in
work-order Section 81.4, not a new method-selection protocol or a result.

## Source

- Official repository: https://github.com/um-csnet/ReplayIDS
- Pinned commit: `b154c79aaf7c422cc5dc35c9763f2ab1d5775342`.
- Inspected implementation: `main.py`, `src/strategies/replay.py`,
  `src/data/preprocess.py`, `configs/config.yaml`, and
  `configs/data/features.yaml` at that commit.
- Upstream license: MIT, copyright 2026 afifhaziq. Any redistributed upstream
  implementation must retain its license notice.
- The upstream model is supplied by `tab-transformer-pytorch`; its exact
  dependency version must also be recorded by the executable adapter.

## B1 boundaries and necessary adaptations

B1 keeps our frozen data rows, task order, Task-0-fitted preprocessing,
permanent Normal calibration holdout, formal seeds 1/2/3/4/42 and calibration
targets 0.01/0.02. TEST must not fit vocabularies, scaling, stopping rules or
thresholds. ER-Bal-TabT receives the same 50,000-row-per-task benign stream as
AR, excluding the calibration holdout. This is a protocol adaptation, not a
reproduction of the upstream paper's absolute scores.

The following source differences must be explicit in the implementation:

1. Upstream `build_replay_buffer(..., balanced=True)` computes a percentage of
   all available samples, divides that total equally across seen classes, and
   samples without replacement. It revisits the supplied previous experiences;
   it is not an online bounded reservoir. B1 must obey Section 81's registered
   per-class memory settings and our old-class access restriction instead:
   admit a class while its training rows are available, retain the permitted
   sample, and never rebuild it by rereading discarded old-class pools.
   Preserve benign replay. Record actual per-class counts, retained validation
   rows and bytes rather than claiming identical upstream buffering.
   Thus B1 evaluates ER-Bal-TabT under the same no-revisit constraint as all
   baselines, not the upstream code's experience-rebuilding implementation.
   The paper describes reservoir sampling; that description is consistent
   with retaining samples without revisiting discarded experiences.
2. The fixed-budget setting is our existing 200-times-seen-class budget with
   existing AR accounting; the new setting is 1% per class's training flows.
   Keep AR's existing allocation semantics explicit, including its separate
   benign stream and calibration storage. Reuse existing AR-MLP cells only
   when the full memory/data/training configuration matches; do not silently
   relabel a different allocation as an existing result.
3. Upstream `main.py` sizes categorical vocabularies using train, validation
   and test arrays. B1 must not copy that TEST access. Use Task-0 training-only
   categorical vocabularies plus an explicit unknown token, fixed thereafter.
4. Upstream `configs/config.yaml` defaults to eight epochs. Section 81's
   explicit published ER schedule takes precedence: AdamW 5e-4, weight decay
   1e-5, batch 256, ReduceLROnPlateau factor 0.1/patience 5, validation-loss
   early stopping patience 10 and at most 100 epochs. No TEST-based stopping.
   For ER stopping, reserve a seed-fixed 10% of non-calibration benign TRAIN
   rows; intersect this fixed role assignment with each original 50,000-row
   AR benign stream. Its optimization/stopping union is the identical stream,
   but optimization counts are not identical to AR. Stable roles prevent an
   earlier optimized benign row from later becoming a stopping row. Attack
   validation uses the existing TRAIN split; old validation rows must fit in
   the bounded buffer. The permanent Normal calibration pool is never used
   for early stopping. This is a schedule adaptation, not compute matching.
5. Retain the registered TabTransformer dimensions (32, six blocks, ten
   heads, dropout 0.1, GELU head). AR-TabT uses that backbone in both G and F;
   it remains AR, with its registered loss and retention behavior, not ER.
   Implementation must disclose any head/interface adaptation before training.

## Categorical features

- UNSW-NB15: `proto`, `service`, `state`. Our cache contains their Task-0
  one-hot blocks. A deterministic reversal can recover the known category
  index; an all-zero block is the unknown token. Do not infer new categories
  from TEST or restore information absent from the frozen representation.
- CIC-IDS-2017: follow the pinned upstream feature-name list: Destination Port,
  Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, FIN Flag Count,
  SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count,
  CWE Flag Count, ECE Flag Count. All 13 names exist in our cache with the
  `numeric__` prefix. They are already standardized float32 values, not raw
  integer token IDs: never cast them directly to integers. Map distinct
  Task-0 training values to categorical IDs and map later unseen values to
  the unknown token. This preserves the available cache information, not an
  assertion that float32 preprocessing is invertible to the raw data.
- Remaining columns stay continuous. Do not substitute FT-Transformer or an
  all-continuous MLP while calling it the registered TabTransformer.

## Execution status and interpretation

As of 22 October 2026, both fixed200 TabTransformer methods have completed
all five formal seeds on UNSW-NB15 and CIC-IDS-2017 (20 cells). Their local
execution records and saved operating points are published under
`ncd/results/problem_scout/section81/publication_v1/`. The same-entry smoke
and adapters are implemented for fixed200; AR percent1 remains pending.

CIC-IDS-2017 fixed200 S3 uses final checkpoint alpha=0.01 operating points,
not the top-level AR metric summary (which uses alpha=0.02). Paired results:

| Seed | AR-TabT final BAcc (%) | ER-Bal-TabT final BAcc (%) | AR minus ER (pp) |
|---|---:|---:|---:|
| 1 | 89.9788 | 94.9962 | -5.0175 |
| 2 | 80.6051 | 93.1697 | -12.5646 |
| 3 | 89.7893 | 90.3642 | -0.5749 |
| 4 | 89.9304 | 94.8085 | -4.8780 |
| 42 | 91.5792 | 87.9333 | +3.6459 |
| Mean | 88.3766 | 92.2544 | -3.8778 |

The registered superiority test **fails**: paired one-sided t=-1.438814,
df=4, p=0.888200; exact sign-flip sensitivity p=0.9375 (32 assignments).
This is not an equivalence claim or a test of ER superiority. Each saved
BAcc was checked against its saved confusion matrix. The reproducible
consumer is `paper_aacil/compare_section81_cic_s3.py`; its full-precision
output and input hashes are in `paper_aacil/results/section81_cic_fixed200_s3.json`.
No training, inference, recalibration or cell re-export was performed.
S1/S2 remain the lead's Section 113 results and were not recomputed here.

Existing CSE45 results, historical FAIL decisions and AR selection remain
unchanged. B1 is a bounded RQ2 comparison and does not close RQ1, RQ3 or
the substantive-innovation requirement. Next is memory setting (ii), with
capacity A in parallel under Section 110 once its implementation, same-entry
smoke and measured resources are ready. No capacity HPC job has been submitted.
