# Recovered five-seed explanation-drift sensitivity

7 September 2026. Evidence recovery from completed Job 434747, not new training or SHAP execution.

The previously exported single-seed grid understated the available completed work. Each protected v9 seed package for seeds 1, 2, 3, 4 and 42 contains 64 Expected-Gradients settings. The [recovered summary](../results/malaya-sensitivity-recovered-20260907/summary.json) contains 320 verified seed-setting rows and 64 aggregate settings. Each analysis file matches its locally downloaded original checksum registry. All counts were recomputed from the saved transition values.

## Registered grid and primary setting

- Top-k: 5, 10, 15 or 20 features.
- Jaccard threshold: 0.50, 0.60, 0.70 or 0.80.
- Allowed recall drop: 0, 2, 5 or 10 percentage points.
- Eligibility uses recall change strictly greater than minus the allowed drop.
- A drift event uses Jaccard similarity strictly less than the threshold.
- Primary setting: k=15, threshold=0.70, allowed drop=0.05.

The primary gives 61 events among 82 eligible transitions. The mean of five seed-level rates is **73.98%**; pooling the counts gives **74.39%** (61/82). These estimands differ and must not be substituted for one another. The 100 total class-by-adjacent-checkpoint transitions also have finite cosine similarity and Kendall tau-b values.

## What this closes, and what it does not

At the fixed 5-pp recall-drop guard, the mean seed-level event percentages are:

| Top-k | Jaccard < 0.50 | < 0.60 | < 0.70 | < 0.80 |
|---|---:|---:|---:|---:|
| 5 | 58.09% | 58.09% | 96.40% | 96.40% |
| 10 | 17.00% | 57.75% | 92.16% | 92.16% |
| 15 | 3.69% | 48.20% | 73.98% | 96.40% |
| 20 | 1.18% | 13.17% | 60.42% | 81.50% |

This is a substantial operating-threshold dependence, not evidence that 0.70 identifies harmful changes. Event counts are expected to grow with a more permissive threshold, and equal-sized top-k overlaps take discrete values. The relevant next test is whether a frozen trigger improves later outcomes at a fixed repair budget, not whether a chosen setting produces a preferred event rate.

This closes the inventory gap for a completed five-seed EG drift grid and alternative-metric coverage. It does not establish robustness: a sensitivity grid must be interpreted, not merely counted.

It is not a new three-method governance-state grid. Changing top-k changes feature-deletion mass and can alter admission; changing drift thresholds can alter later recertification. Full sequential ledger replay for each policy, attribution method and seed remains separate work.

The source attribution scope is a deterministic CPU reconstruction evaluated on fixed true-class probe batches. It does not claim numerical equivalence to archived GPU score arrays or to other batch partitions. The present audit does not revalidate raw attribution arrays, rerun remote checksums, or close derivative-faithfulness questions.

The replication unit is the training seed on one fixed split. The 320 settings and 100 transitions are not 320 or 100 independent experiments. Do not select a favorable threshold on these test outcomes.

## Reproduction

Use Python 3 with the standard library:

```text
python tools/audit_existing_threshold_grid.py --root PATH_TO_DOWNLOADED_V9_SEED_PACKAGES --output recovered_summary.json
python -m unittest discover -s tools -p test_existing_threshold_grid.py
```

Each seed directory must contain analysis.json and its original SHA256SUMS registry. Input analysis and registry hashes are exported in input_bindings. Protected source paths and raw sample identifiers are intentionally not published.

See the [revised research plan](RESEARCH_PLAN_2026-09-07.md) and [validation priorities](VALIDATION_PRIORITIES_2026-09-07.md).
