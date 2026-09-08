# Paired frozen-encoder export: verified completion

Verified on 9 September 2026. Both local CUDA processes exited with code 0;
the independent read-only pair audit returned PAIR_AUDIT_PASS. This stage
performs inference only, not downstream classifier training or evaluation.

| Measurement | Prospective P | Offline O |
|---|---:|---:|
| Training rows exported | 149,476 | 268,697 |
| Official test rows exported | 227,723 | 227,723 |
| Total forwarded rows | 377,199 | 496,420 |
| Batches, maximum size 384 | 984 | 1,294 |
| Stage wall time, seconds | 55.999 | 70.844 |
| Encoder forward time, seconds | 51.458 | 67.017 |
| Peak CUDA allocated bytes | 921,456,640 | 921,456,640 |
| Peak CUDA reserved bytes | 1,231,028,224 | 1,231,028,224 |

Both use the same frozen Task-0 FT256x4 encoder and normalization. The encoder
state digest is unchanged before and after each export. Every test descriptor
(including embeddings, labels, row identity, group identity and task availability)
has the same digest across P and O. Actual data arrays are not published here.

Manifest digests:

- P: f2a1e3ff141bf44642207c92305c080ea453971510060ff4287f84fb476e641c
- O: 9db3e5c00cf87a3548e9ee54375cd78a21efd8627ec6ed36484592ab19540775

Profile digests:

- P: a156edc3391b27ffd24339dd1f84fe97af33e03e734704663e5dbfe1b7d51d3c
- O: c1663ef3fa0f4869c5e2f7423a348cf85ca5ab34a13c2cffc6908f5002f9ea9d

Shared encoder-state digest:
56643d56c68a00d6b7c1006b8a37e82855db0c2bc43dd47afd387928998cbb7b.

Limitations: no new accuracy, forgetting, attribution or governance result is
established. The later L1 runner is a limited head/loss control, not full OFRA
with DP-Means routing. Equal epoch counts across unequal P/O row counts would
not provide equal training exposure. Online experiment tracking still needs
its own implementation and review; these exports did not contact W&B.
