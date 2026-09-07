# Source-bound confound audit

Local static source inspection, 7 September 2026; not new training or a new
independent audit of historical performance. Full file hashes and numbered source
anchors are in [source_evidence.json](source_evidence.json).

The inspected public source tree identifies commit
`4051ea2556c1215e839a2faea0cd2adb411c28da` of `1838904818/Class-IL`.
It contains OFRA source but no `continual_baselines/` directory at inspection.
Baseline evidence below is instead the archived `20260906-balanced-replay-seed1-v1`
payload. Its validation source hash exactly matches the source hash recorded in
the historical Malaya seed-1 protocol; the Malaya config hash matches too.
This binding is evidence about that historical implementation, not evidence that
the new ladder has run. A local candidate copy was byte-equal but is not used as
the canonical source or represented as published. No baseline source is copied
into this package. Historical protocol bytes are not republished here.

| Confound | Direct source evidence | Consequence |
| --- | --- | --- |
| Target and loss differ | OFRA `models.py:30-47,150-165`; `validation.py:1251-1266,1451-1475`. Baseline `validation.py:427-447`. | OFRA uses independent two-logit binary targets; focal is conditional on positive population below `minority_threshold`, otherwise binary CE. Balanced replay uses multiclass CE over seen classes. It is inaccurate to call every OFRA head focal. |
| Equal epochs do not equal exposure | OFRA `validation.py:1251-1269,1394-1455`; baseline `validation.py:367-420,454-473`. | OFRA uses all positives per head and capped without-replacement negatives from current classes plus old exemplars. At batch 384/ratio 4 its positive microbatch is 76. Balanced replay uses 192 new rows plus equal replacement replay draws. Different target/row counts and update counts follow, especially when negatives are scarce. Values here describe code, not measured run counters. |
| Storage count is not sampling weight | Baseline `validation.py:410-420,468-472,565-580`; OFRA `validation.py:1919-1934`. | Uniform raw exemplar selection versus farthest-first selection and repeated old-row draws differ. Identical per-class row capacity does not establish equal stored row identities or multiplicity. |
| Frozen versus updated representation | OFRA `validation.py:959-962,1221`; baseline `validation.py:184-191,1116-1119,1167-1193`. | OFRA freezes the Task-0 encoder and optimizes a new head; trainable replay updates its shared encoder/classifier. Equal dimensions and seeds are not a shared post-pretrain tensor checksum. |
| Task-0 work differs from increments | OFRA `validation.py:751-767` and subsequent `train_family`; baseline `validation.py:1048-1063,1135-1152`. | Shared pretraining must be counted separately from OFRA Task-0 family fitting. A baseline checkpoint immediately after common pretraining is not the same training schedule as a post-family OFRA checkpoint. |
| NME is genuinely non-gradient later | Baseline `validation.py:608-625,1117-1119,1169-1176`. | NME later optimization is N/A, while embedding/candidate selection/prototype construction have real costs. The retained multiclass classifier still exists in this implementation and must not disappear from memory accounting because prediction uses means. |
| Extra transient/resident models | Baseline `validation.py:275-279,1116`; OFRA `models.py:131-145`. | The generic baseline path allocates an old-model copy whenever old classes exist, although only iCaRL-style uses its predictions. The balanced path returns before that copy; do not charge that copy to balanced replay without observing it. The common model can remain resident. Head-only parameter counts exclude router and replay state. |
| Checkpoint rule can change | OFRA `validation.py:155,1222-1246`; guarded config `family_checkpoint_selection`. Baseline `validation.py:1188-1222`. | OFRA supports last and calibration-selected rules, while the inspected baseline evaluates after the fixed epoch loop. The inspected guarded config is a variant, not the registered ReplayIDS last-epoch primary. Bind the exact config/result before any comparison. |
| Router capacity is multidimensional | OFRA formal config and `routers.py:38-84`. | `router_cap_samples=3000` caps input fitting rows; `router_max_centroids=32` bounds retained centroids, not encoder width or total memory. Discovery/work arrays and centroid counts also cost bytes. |

There is no evidence here that loss, raw exposure, successful optimizer steps,
initial checkpoint tensors and full measured memory are simultaneously matched
across the historical OFRA and replay methods. That conclusion is narrower than
claiming their reported scores are invalid. Existing descriptive comparisons can
remain labelled heterogeneous method comparisons.

The new shared binary architecture control, exact common-initialization harness,
and TRIL3 adaptation are not implemented or reproduced. Literature citations
cannot fill these missing controls. The new validator audits supplied evidence;
it does not automatically instrument the existing trainers or repair their
historical records.
