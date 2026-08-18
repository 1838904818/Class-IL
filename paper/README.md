# Manuscript status

The v2.2 supervisor-response manuscript is aligned with the validated evidence
available on 19 August 2026. Earlier v2.0 and v2.1 files are retained as
immutable revision snapshots.

The current paper reports:

- four-seed descriptive FT-Transformer 512x12 results for MalayaNetwork_GT and
  NSL-KDD;
- five-seed Malaya TabM mean-embedding results in the unchanged OFRA
  prediction pipeline;
- five-seed cumulative CatBoost and TabM classifier-capacity diagnostics,
  explicitly separated from OFRA and ETG claims;
- a separate five-seed CSE-CIC-IDS2018 FT-Transformer 256x4 closure under the
  one-Task-0-epoch plus one-later-task-epoch protocol;
- the MalayaNetwork_GT seed-1 explanation analysis of the registered
  `joint_cap3000` class margin;
- a same-probe three-method attribution robustness pilot using Expected
  Gradients, feature ablation, and Gradient x Input;
- offline ETG governance outcomes bound to completed DICC Job 389896; and
- a no-look-ahead preprocessing audit that passes Task-0-only numerical scaling
  but identifies bounded future-only one-hot schema columns for NSL-KDD and
  UNSW-NB15.

The present single-seed explanation result is `12/17` eligible
class-by-adjacent-checkpoint transitions and must not be interpreted as a
packet, flow, sample, or production-incident rate.

The attribution conclusion is method-sensitive. Across the three valid methods,
all-method agreement is 40.0% for admission, 43.3% for ETG state, and 20.0% for
the silent-drift conclusion. Integrated Gradients was attempted but excluded
from the primary comparison after failing the recorded completeness check.

The manuscript is a supervisor response, not a final submission-ready claim of
superiority. In particular, the TabM accuracy gain is not accompanied by a
matched balanced-accuracy improvement, and the cumulative CatBoost result is
not a continual-learning comparison. Strict Task-0-only categorical vocabulary
rebuilds and affected reruns remain open. The current validated result index is
under `results/`, and the itemized response is in
`SUPERVISOR_RESPONSE_2026-08-19.md`.
