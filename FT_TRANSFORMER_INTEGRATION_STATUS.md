# OFRA FT-Transformer candidate status

## Outcome

The isolated candidate supports `encoder_type=mlp|ft_transformer` in the
shard-backed formal runner. The FT encoder consumes every post-cache feature as
a standardised continuous token (`categories=()`) and returns a `d_model`
embedding. FamilyHead/LoRA, routing, and exemplar logic are unchanged.

This is an implementation result, not an accuracy result. No real-data or GPU
training has been started from this directory.

## Candidate

| Field | Value |
|---|---:|
| Upstream | `lucidrains/tab-transformer-pytorch==0.6.1` |
| Git commit | `10c258aa7ecf8c7e948e38c104a87caed49a6a9a` |
| License | MIT |
| Embedding dimension | 64 |
| Depth | 4 |
| Heads / head dimension | 4 / 16 |
| Attention / FF dropout | 0.1 / 0.1 |
| Residual streams | 1 (standard residual; mHC disabled) |
| Train microbatch / accumulation | 64 / 4 |
| Nominal effective batch | 256 |
| Evaluation batch | 128 |

## Parameter audit

| Dataset | Features | MLP128x2 encoder | FT64x4 encoder | Encoder ratio |
|---|---:|---:|---:|---:|
| NSL-KDD | 122 | 32,256 | 277,632 | 8.61x |
| UNSW-NB15 | 194 | 41,472 | 282,240 | 6.81x |
| CIC-IDS-2017 | 78 | 26,624 | 274,816 | 10.32x |
| CIC-IDS-2018 | 78 | 26,624 | 274,816 | 10.32x |
| MalayaNetwork_GT | 77 | 26,496 | 274,752 | 10.37x |

Counts are generated from instantiated modules. The protocol also records the
per-class FamilyHead and final encoder-plus-head totals for each dataset.

## Verification

- Complete copied-project test suite: 39/39 passed.
- FT-specific tests: dependency pin, continuous-only input, output shape,
  FamilyHead compatibility, CPU one-step determinism, parameter records, and
  accumulation schedule all passed.
- Worst-width CPU check (UNSW, 194 features): output `[4, 64]`; two independent
  initialisations/forwards produced identical state and output hashes.
- End-to-end streaming synthetic run (Task-0 pretraining, accumulation, family
  heads, router, exemplars, checkpoints): two runs produced the same formal
  result hash.
- MLP regression against the untouched v3 copy: after excluding only the new
  accumulation audit keys, all pre-existing normalization, training, exposure,
  router, exemplar, and checkpoint fields are exactly identical.

## Required GPU probe

Use UNSW first because 194 feature tokens create the largest quadratic
attention matrix. Run one pretraining batch and one evaluation batch while
recording peak allocated and reserved CUDA memory. Start with train 64,
accumulation 4, evaluation 128. If that exceeds the 8 GB budget, create a new
protocol using train 32, accumulation 8, evaluation 64. Do not change a config
inside an existing output directory.

## Fair model comparison

Run the FT candidate and the MLP128x2 control with identical full-cache
manifests, evaluation views, five seeds, effective batch 256, epochs, learning
rate, losses, router settings, and exemplar settings. The paired replacement
comparison intentionally changes only the encoder architecture/capacity fields.
Keep historical MLP results as context; use the newly paired MLP control for
formal model-size inference because its microbatch/accumulation schedule exactly
matches FT.
