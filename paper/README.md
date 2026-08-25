# Manuscript status

The v2.6 manuscript aligns the research questions, objectives, methods, and
results around prediction/retention (RQ1), explanation stability (RQ2), and
non-suppressing governance (RQ3). It retains the validated five-dataset evidence
available on 25 August 2026. Earlier v2.0 through v2.5 files are retained as
immutable revision snapshots.

The current paper reports:

- strict five-seed prediction evidence for NSL-KDD, UNSW-NB15,
  CIC-IDS-2017, CSE-CIC-IDS2018, and MalayaNetwork_GT, with capacity disclosed
  per dataset;
- five-seed Malaya TabM mean-embedding results in the unchanged OFRA
  prediction pipeline;
- five-seed cumulative CatBoost and TabM classifier-capacity diagnostics,
  explicitly separated from OFRA and ETG claims;
- the strict CSE-CIC-IDS2018 FT-Transformer 256x4, 8+10-epoch closure from
  DICC Job 402073, with the earlier 1+1-epoch result retained only as a
  protocol-separated historical result;
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

The threshold-sensitivity grid varies top-k over `10, 15, 20`, Jaccard over
`0.6, 0.7, 0.8`, and allowed class-recall drop over `2, 5, 10` percentage
points. This third dimension is class recall, not overall accuracy.

The attribution conclusion is method-sensitive. Across the three valid methods,
all-method agreement is 40.0% for admission, 43.3% for ETG state, and 20.0% for
the silent-drift conclusion. Integrated Gradients was attempted but excluded
from the primary comparison after failing the recorded completeness check.

The manuscript is a supervisor-facing evidence revision, not a final claim of
superiority. On Malaya, joint cap improves Macro-F1 and balanced accuracy but
does not establish an accuracy or forgetting gain. The TabM accuracy gain is
not accompanied by improved class-balanced or retention metrics, and the
cumulative CatBoost result is not a continual-learning comparison. Strict
Task-0-only categorical vocabulary rebuilds, multi-seed explanation/ETG, and
stronger matched baselines remain open. The
current validated result index is under `results/formal-five-seed-20260824/`.

The additional Malaya explanation/ETG seed attempt did not pass the complete
source-bound validation workflow, so it is not promoted into v2.6. The paper
continues to report only the validated seed-1 explanation and ETG evidence.
