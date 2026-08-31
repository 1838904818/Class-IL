# ReplayIDS D2 optimizer-recipe diagnostic, seed 42

This directory records the public-safe comparison among the completed D2 Adam
control (DICC Job `414989`), the D2 AdamW recipe candidate (DICC Job `425182`),
and the matched lower-learning-rate Adam diagnostic (DICC Job `425382`). The
data protocol, replay capacity, FT-Transformer architecture, epoch budget,
focal loss, router settings, evaluation view and seed were held fixed.

The candidate changed the complete optimizer recipe from Adam with learning
rate `1e-3` and zero weight decay to AdamW with learning rate `5e-4` and weight
decay `1e-5`. The follow-up then kept Adam and zero weight decay and changed
only the learning rate from `1e-3` to `5e-4`. Together, the two runs separate
the learning-rate effect from the earlier bundled optimizer recipe.

## Registered result

The comparison uses `official/joint_cap3000`. Lower forgetting and Benign FPR
are better; the other metrics are better when higher.

| Recipe | Average task accuracy | Forgetting | Final accuracy | Macro-F1 | Balanced accuracy | Benign FPR | Attack recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adam, lr `1e-3`, wd `0` | 91.45% | 4.36% | 89.08% | 55.70% | 91.46% | 10.46% | 96.18% |
| AdamW, lr `5e-4`, wd `1e-5` | 81.17% | 2.15% | 92.10% | 59.52% | 71.45% | 6.23% | 87.08% |
| Adam, lr `5e-4`, wd `0` | 81.17% | 2.15% | 92.10% | 59.52% | 71.45% | 6.23% | 87.08% |

AdamW improved final overall accuracy, Macro-F1, forgetting and Benign FPR,
but materially reduced average task accuracy, balanced accuracy and attack
recall. Adam with learning rate `5e-4` reproduced the same seven registered
metric values as the AdamW recipe. The result hashes differ, so this is not a
claim that the trained models or predictions are bit-identical. It does show
that the measured metric-level trade-off does not require AdamW or non-zero
weight decay. The lower-learning-rate recipes are not selected as replacements
for the Adam `1e-3` control and are not expanded to five seeds.

This is a single-seed diagnostic. It does not establish statistical
superiority and is not a five-seed publication result.

## Execution and integrity

- Candidate job: `425182`, `COMPLETED`, exit code `0:0`
- Candidate runtime: 13 minutes 24 seconds
- Allocation: one A100 GPU, two CPUs and 12 GiB memory
- Candidate deterministic result SHA-256:
  `67250b60ca5b9d2fb1362c9db2ec881c6c18c188bf5c3b7d505f32b5932e03b3`
- Candidate protected checksum-registry SHA-256:
  `92d273d30cf483cd39f2424f1213da3e675eed72ae7fa57fdfbfd84be48993dc`
- Control deterministic result SHA-256:
  `314ff9f6b8742c6ccfca926b60af7582acfa32a6dfb34f244edd75fd4e5c38db`
- Control protected checksum-registry SHA-256:
  `6e66dfe888799baf227c4d96ad32993efa076798e4d96941cae41e3f995768ab`
- W&B run:
  `https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/26c7891b16704095228c3d2e`
- Learning-rate diagnostic job: `425382`, `COMPLETED`, exit code `0:0`
- Learning-rate diagnostic runtime: 13 minutes 56 seconds
- Learning-rate diagnostic allocation: one A100 GPU, two CPUs and 12 GiB memory
- Learning-rate diagnostic deterministic result SHA-256:
  `bb7e29c98dcdd3808cdf196d01235132e81158f21e3dac8a8fa23245ebaf1861`
- Learning-rate diagnostic protected checksum-registry SHA-256:
  `a11e8e55e32e3b11bb3b77bc23198e14af890c6870ef650a8f593f77627bf8c8`
- Learning-rate diagnostic W&B run:
  `https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/0b3a545aa832bdc4942a297e`

Both protected result packages pass their complete SHA-256 registries. The
learning-rate package contains 22 verified files and four checkpoint monitoring
records. W&B stored and closed both aggregate runs. The protected JSON and
checksum registries are authoritative.

SHAP and ETG were intentionally not computed in this optimizer screen.
