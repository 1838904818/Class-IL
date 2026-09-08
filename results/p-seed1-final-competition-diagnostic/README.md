# Fixed final-checkpoint diagnostic: P seed 1

The approved local inference command exited 0. Both arms reproduced their
archived prediction SHA before writing aggregate diagnostics. No training,
parameter selection, score calibration or W&B upload occurred.

Original diagnostic SHA-256 (before publication provenance annotation):
`0d1cb9a23afa8fe952ef21f8085c799df7f2742f8eebf7b538b41b535d442f0e`

## Provenance

This is the seed-1 frozen-embedding L1 P-protocol diagnostic on the
ReplayIDS CIC-IDS-2017-derived dataset, not native routed OFRA.
Data reference: https://github.com/um-csnet/ReplayIDS .
The published JSON adds provenance only; its measured arm statistics match
the original diagnostic. Hashes identify retained source artifacts and do
not imply that checkpoints or row-level data are distributed here.

- Embedding manifest: `f2a1e3ff141bf44642207c92305c080ea453971510060ff4287f84fb476e641c`
- Execution binding: `8c4621a2d5291309f429ab361ea28a18bb4895c94a5a4ffb1cd2c999684bb6e0`
- Source result: `da9060cba18e988534da766704d754b3e99e4300ff623d2d6a657b0b12581211`
- Final checkpoint: `f839e14196c195ccffe4758522baa7de1e398722af671adbd49ad3cc1d75fae4`

The task class pairs are [0,1], [2,3], [4,5], [6,7]. Class IDs are Benign,
GoldenEye, Hulk, Slowhttptest, Slowloris, FTP-Patator, Heartbleed and
SSH-Patator in that order. The final test partition has 227,723 rows.
The P arm uses 149,476 fit rows with a Task-0-derived majority cap.
These data-processing choices are project protocol choices.

## Observed competition, without a demonstrated remedy

In the CE arm, 1,029 of 1,100 Slowhttptest examples and 1,576 of 1,588
FTP-Patator examples received their own binary-head positive probability
above 0.5, but another head won the final argmax. Their final multiclass
recall was zero. SSH-Patator similarly lost 1,175 above-threshold examples
out of 1,179 positives. These are directly observed score competition events,
not proof that the heads have useful discrimination.

At the same threshold, Slowhttptest's binary false-positive rate is 65.38%
and FTP-Patator's is 34.46%. Consequently, high binary recall alone cannot
justify deploying independent threshold decisions. This binary FPR uses
all other classes as negatives, and is not the overall Benign FPR.

Heartbleed has only two test positives, neither above threshold in either
arm. Conditional focal changes its negative-score behavior while leaving
final multiclass metrics unchanged. Identical aggregate metrics therefore
must not be interpreted as identical models or equivalent losses.

## Next falsifiable comparison

Evaluate ranking and calibration on an independently exported calibration
partition using the same frozen encoder. Compare a simple calibration-only
baseline with any proposed competition correction under matched resources.
Do not select thresholds from these inspected test results. Confirmation
requires evidence beyond this exploratory seed and inspected test set.

This analysis covers the final checkpoint only. It neither isolates the
time of failure nor tests DP-Means, the OFRA joint scorer, normalization
changes, a P/O difference, or an ETG intervention. AUROC/AP and a successful
algorithmic remedy remain unmeasured here.
