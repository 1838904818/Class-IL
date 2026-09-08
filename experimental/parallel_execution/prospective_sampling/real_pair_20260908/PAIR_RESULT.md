# Real prospective and offline data derivation

All four increments for P and O completed locally and passed output validation. P uses a Task-0-only normal cap of 5559. O retains the historical offline normal cap of 124780 as a comparison policy, not a prospective rule.

| Data role | Rows |
|---|---:|
| P fitting across increments | 149476 |
| O fitting across increments | 268697 |
| Shared calibration | 68313 |
| Unchanged official test | 227723 |

The independent pair audit confirmed equal calibration identities and equal selected attack identities at each increment, P normal identities nested in O, and equal common transform cohort hashes. The difference is 119221 Benign fitting rows at Task 0; later-increment selected row counts are equal. This controls the intended sample-selection contrast but does not by itself equalize optimizer steps or total training exposure.

The official test was verified without sampling or modification. Heartbleed still has only two official-test rows; this preparation does not solve that statistical limitation. Capture independence remains unproven under row-identity proxy grouping.

No preprocessing transform, encoder, classifier or attribution method was trained or evaluated. The next stage requires one newly trained common-cohort Task-0 encoder and controlled downstream comparisons. Old checkpoints cannot silently substitute for that common-cohort training.

PAIR_AUDIT.json records per-increment manifest hashes. Published manifests and completion markers bind the local arrays, which are not included in this evidence-only directory. Original Task 0 evidence remains unchanged.
