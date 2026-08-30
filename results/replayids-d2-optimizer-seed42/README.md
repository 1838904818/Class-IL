# ReplayIDS D2 optimizer-recipe diagnostic, seed 42

This directory records the public-safe comparison between the completed D2
Adam control (DICC Job `414989`) and the D2 AdamW recipe candidate (DICC Job
`425182`). The data protocol, replay capacity, FT-Transformer architecture,
epoch budget, focal loss, router settings, evaluation view and seed were held
fixed.

The candidate changed the complete optimizer recipe from Adam with learning
rate `1e-3` and zero weight decay to AdamW with learning rate `5e-4` and weight
decay `1e-5`. It is therefore an optimizer-recipe screen, not an isolated
estimate of an Adam-versus-AdamW main effect.

## Registered result

The comparison uses `official/joint_cap3000`. Lower forgetting and Benign FPR
are better; the other metrics are better when higher.

| Recipe | Average task accuracy | Forgetting | Final accuracy | Macro-F1 | Balanced accuracy | Benign FPR | Attack recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adam, lr `1e-3`, wd `0` | 91.45% | 4.36% | 89.08% | 55.70% | 91.46% | 10.46% | 96.18% |
| AdamW, lr `5e-4`, wd `1e-5` | 81.17% | 2.15% | 92.10% | 59.52% | 71.45% | 6.23% | 87.08% |

AdamW improved final overall accuracy, Macro-F1, forgetting and Benign FPR,
but materially reduced average task accuracy, balanced accuracy and attack
recall. The candidate is not selected as a replacement for the Adam control.
The next registered diagnostic is Adam with learning rate `5e-4` and zero
weight decay, which separates the learning-rate effect from the optimizer and
weight-decay changes.

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

The protected result package and all 23 listed files pass local and remote
SHA-256 verification. Four checkpoint monitoring records are present. W&B
created the run, stored the final summary and table references, and closed the
stream; its final internal log also records a context-cancelled run-files
metadata request. The protected JSON and checksum registry are authoritative.

SHAP and ETG were intentionally not computed in this optimizer screen.
