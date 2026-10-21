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

The source inspection above is complete; the B1 adapter and same-entry smoke
are not yet implemented. No B1 training, result or comparative advantage is
claimed by this record. Existing CSE45 results, historical FAIL decisions and
AR selection remain unchanged. B1 is a bounded RQ2 comparison and does not
close RQ1, RQ3 or the substantive-innovation requirement. Continue in the
registered order B1, capacity A, B2, then Malaya; publish completed cells with
their actual configuration, source, seed, command and execution costs.
