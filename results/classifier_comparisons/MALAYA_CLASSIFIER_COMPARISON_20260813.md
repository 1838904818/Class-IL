# MalayaNetwork_GT Classifier Comparison — 2026-08-13

## Scope

This report records additive classifier diagnostics. It does not replace the
registered FT-Transformer OFRA method or claim that cumulative multiclass
models are matched OFRA competitors.

Dataset contract: 36,372 train rows, 10,370 official-test rows, 77 numerical
features, 10 application classes, and the fixed task order
`[0,1] -> [2,3] -> [4,5] -> [6,7] -> [8,9]`.

## Matched cumulative-multiclass diagnostics

Values are mean +/- sample standard deviation over seeds `1,2,3,4,42`.

| Official metric | CatBoost, 400 trees | TabM cumulative | CatBoost - TabM |
|---|---:|---:|---:|
| Final accuracy | 66.50 +/- 0.44% | 65.01 +/- 0.96% | +1.49 pp |
| Macro-F1 | 34.76 +/- 0.69% | 32.66 +/- 1.31% | +2.09 pp |
| Balanced accuracy | 39.10 +/- 0.73% | 34.41 +/- 1.51% | +4.69 pp |
| Average forgetting | 15.34 +/- 0.73 pp | 9.43 +/- 1.80 pp | +5.91 pp |

Both models retrain a fresh multiclass classifier at every checkpoint using all
training rows from classes seen so far. CatBoost improves final classification
metrics but has greater checkpoint-to-checkpoint forgetting under the same
metric definition.

## TabM mean-embedding OFRA arm

TabM was also integrated as a mean-embedding encoder inside the unchanged OFRA
prediction pipeline. This is distinct from the cumulative TabM row above. The
adapter averages 16 member embeddings to one 256-dimensional representation;
the existing family heads, replay budget, DP-Means routers, joint weight, and
cap-3,000 protocol remain unchanged.

Values are mean +/- sample standard deviation over seeds `1,2,3,4,42`.

| Official OFRA arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---:|---:|---:|---:|
| Head only | 56.17 +/- 1.79% | 11.11 +/- 1.99% | 14.43 +/- 2.35% | 0.61 +/- 0.46 pp |
| Router only, cap 3,000 | 40.15 +/- 3.49% | 17.44 +/- 1.44% | 19.29 +/- 1.11% | 12.75 +/- 1.89 pp |
| Joint, cap 3,000 | 57.99 +/- 3.93% | 20.46 +/- 1.19% | 21.60 +/- 1.70% | 4.39 +/- 2.56 pp |
| Joint, uncapped | 57.93 +/- 3.84% | 20.34 +/- 1.31% | 21.48 +/- 1.87% | 4.33 +/- 2.39 pp |

For the four seeds shared with FT-Transformer 512x12, TabM-OFRA joint-cap
accuracy is 59.59% rather than 54.37% (+5.23 percentage points). Macro-F1
changes by +0.15 points, balanced accuracy by -0.64 points, and forgetting by
-0.01 points. BitTorrent constitutes 53.39% of the test set and its recall
improves by 10.54 points, while recall declines for Steam, TeamViewer, Discord,
and Webex. The supported interpretation is therefore an aggregate-accuracy
gain rather than a uniform class-balanced improvement.

This TabM-OFRA campaign covers prediction only. It does not contain a TabM SHAP
or ETG analysis and must not be described as a completed TabM explanation-
governance chain.

## Role-separated OFRA context

These rows use the full OFRA family-head, router and memory pipeline and are
not directly matched to the cumulative models above.

| Model and protocol | Seeds | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---|---:|---:|---:|---:|
| FT-Transformer 64x4, OFRA joint cap-3000 | 1,2,3,4,42 | 57.99 +/- 3.58% | 18.82 +/- 2.85% | 20.36 +/- 3.33% | 3.07 +/- 1.52 pp |
| FT-Transformer 512x12, OFRA joint cap-3000 | 1,2,3,4 | 54.37 +/- 3.02% | 20.70 +/- 3.72% | 22.70 +/- 3.86% | 3.79 +/- 0.64 pp |

For the common seeds `1,2,3,4`, CatBoost cumulative accuracy is 66.38 +/-
0.40%, compared with 54.37 +/- 3.02% for FT-Transformer 512x12 OFRA. The
12.01 percentage-point difference cannot be attributed to the classifier
alone because CatBoost has access to all seen-class training rows and does not
use OFRA family heads, exemplar memory, DP-means routing, or cap-3000.

The supported conclusion is narrower: the fixed feature representation
contains additional classification headroom, while the OFRA constraints and
decision components account for part of the remaining performance gap.

## Interpretation limits

- CatBoost is a cumulative diagnostic, not an OFRA arm.
- Native CatBoost SHAP is not routed-margin SHAP and is not an ETG result.
- The 400-tree CatBoost setting was selected after inspecting a seed-1
  400-versus-500 sensitivity comparison. The five-seed run is exploratory and
  must not be described as an unbiased confirmatory superiority test.
- FT-Transformer 512x12 currently has four registered Malaya seeds, not five.
- TabM-OFRA has five Malaya prediction seeds but no TabM SHAP/ETG run.
- Runtime comparisons are not reported because CatBoost ran on a local CPU,
  while FT-Transformer OFRA ran on an A100 and includes additional pipeline
  work.

## Evidence bindings

- CatBoost implementation SHA-256:
  `C07F9F6AB8D456C294630007BAA759453DD4354D70EB36330E74606ECC3B46C5`
- Malaya feature schema SHA-256:
  `C99D5B7EE704D3BCACFD9F166615356F3154F4341CCFBAFB16E86CCA53729A12`
- Malaya cache manifest SHA-256:
  `AAEDDC7B73DD77B22A196524E34425CF8605B2DCCAB1BEE195E81097D27E0191`
- CatBoost result-file SHA-256 for seeds `1,2,3,4,42`:
  - `55E2FC2769D18C11FB91E59BE27EB9BCB5AB3706B4EEB7B678E703B6BAF29AA5`
  - `7E2F29E5BB9E15793F7CADA53DDA7583E20FBFD7BBB8EA1871248B46BCFAB440`
  - `3008F064FEDE0FD270E31AB842BDDC8597E11C45CF7042AA24A09DCC4414D8C8`
  - `53F1978D2468DB7AC43E300D407E93FBBE5AD52D66F55CC120350348AF657D7C`
  - `3C36E1F5671E65CC865568B84613B3972A8DE34FBB045EE4D35F7E1A0239C4E9`
- FT-Transformer 64x4 protocol SHA-256:
  `dfde6af447e43b669d59d37e3e002016657628c7f22b58472125da5f0450be4d`
- FT-Transformer 512x12 seed-1 protocol SHA-256:
  `b7e94a1d6cb78b9e003f0e984467d260e996c9060b996f8ff745fb454ca9b48a`
- TabM-OFRA campaign protocol SHA-256:
  `4a13a26d66b751c2f90e041298041c4f20468bd0d354b2a53934f73b93095ddb`
- TabM-OFRA `protocol.json` SHA-256:
  `6458bba71ff5e2ae0b2e3ee0d9d820bf3a60ce28640b68acd3a686f64717bd20`
- TabM-OFRA `summary.json` SHA-256:
  `1f56cc6751b6829a0268433ec9a07b4dff464c2d8550db50203ff9f07e772d10`
