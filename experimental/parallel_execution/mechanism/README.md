# Expansion-normalization mechanism and executable control

7 September 2026. Prospective score-level pilot; not a historical scorer,
trained model, validated innovation, or production deployment.

## Why this control exists

For independent family probabilities p and raw router affinities r, the
current fusion is p_c + lambda (r_c - mean(r)) / (std(r) + epsilon), where
mean/std are taken over the currently seen classes for each input. Therefore
the difference between old classes i and j is

```text
S_i - S_j = p_i - p_j + lambda (r_i - r_j) / (sigma_seen(x) + epsilon).
```

The common mean cancels. Appending a class changes sigma_seen(x), hence the
effective router weight relative to the independent heads. Even a new class
that never wins can change which OLD class wins. This is not necessarily
weight forgetting, a head retraining effect, or a changed example.

The code contains a minimal counterexample: p_old=(0.2,0.8), r_old=(0,-1),
lambda=0.5. Adding p_new=0 and r_new=-100 can reverse the old-class winner
under all-seen standardization although that distant new class does not win.
This is an analytic/synthetic example, not a traffic-data result.

## Candidate: fixed-reference fusion

Freeze a Task-0, training-only reference class/router bank R. For every query,
use its affinities to that unchanged bank to define mu_R(x) and
d_R(x)=max(std_R(x), floor). Score every current class by

```text
S_ref,c(x) = p_c(x) + lambda (r_c(x) - mu_R(x)) / d_R(x).
```

No current ground-truth task/class identifier is passed to the prediction
rule. The same fixed reference is used for old and new examples; Task-0
membership is a design origin, not an oracle at inference. This is not a new
SHAP method and does not create a semantic class or suppress an alert.

Conditional proposition: if feature transformation, encoder, reference bank
and old per-class head/router scores are fixed, appending classes leaves
every old pairwise score difference unchanged. Proof: each term for an old
class depends only on the unchanged p_c, r_c and R, so subtracting any old
pair eliminates the common center without introducing any new-class term.
A stable class-ID tie rule also preserves the restricted old-class argmax.

This proposition does NOT preserve the full argmax: a genuinely competing
new class can still win. It does not guarantee old recall, false-positive
rate, useful new-class learning, calibration, safe deployment, or robustness
to changes of the frozen components. Finite precision is separately tested;
the mathematical statement is not a historical GPU bit-parity claim.

## Controls and failure criteria

The executable includes all-seen normalization with the SAME max-scale floor,
fixed reference, one globally fixed scalar, raw router and head-only controls.
It also includes `all_seen_legacy_formula` with population std plus epsilon;
that is a float64 formula reference, not legacy float32/GPU reconstruction.
Holding the floor formula constant isolates reference-domain change. Comparing
to the historical formula otherwise changes two factors. A global scalar or
raw affinity already has the same conditional pairwise invariance: invariance
alone cannot establish a distinct useful contribution for reference adaptation.

Reject the candidate as a main algorithmic contribution if it does not beat
the simple scaling controls on held-out utility/cost, if reference degeneracy
destabilizes scores, or if retention is purchased through unacceptable loss
of new-class learning or attack/FPR behavior. Report Task-0 sensitivity and
scale-floor sensitivity on development data; never tune them on official tests.

The minimal primary question is whether input-dependent, class-set-independent
router scaling offers useful accuracy/retention/cost behavior beyond a fixed
scalar. Only if that is supported is its added complexity justified.

## Execution

```text
python -B anchor_fusion.py SCORING_MANIFEST.json --output NEW_OUTPUT_DIRECTORY
python -B -m unittest discover -s . -p test_anchor_fusion.py -v
```

The manifest schema is `expansion-fusion-pilot-v1` (the test fixture shows a
complete synthetic example). It binds six `.npy` arrays: old/new independent
positive head probabilities, old/new raw router affinities, and aligned unique
old/new row IDs. It declares the class/reference axes, numerical policy,
positive bounded block size, and provenance fingerprints. Files are read via
read-only memmaps with `allow_pickle=False`, exact SHA checks, path checks and
a final input-identity recheck. Statistics are per query, not across batches.
Output creation is exclusive; a failure cannot receive COMPLETE.json.

Output counts restricted-old winner changes, full winner changes, new winners
and maximum old-pair margin change. No labels are consumed and no accuracy is
claimed. Supplied provenance identifiers do not authenticate the encoder,
reference-bank origin or original data: the real score-export stage must
independently verify those objects and numerical target semantics. The pilot
flags that limitation rather than treating a digest string as proof.

Raw affinities are necessary. Saved standardized z scores alone do not identify
the original mean/scale; they must not be inverted by an invented assumption.
The actual frozen checkpoint-to-score export is implemented in
`native_score_export.py`, using the hash-bound encoder adapter and the native
independent two-logit head/negative-nearest-centroid algebra. It emits both
native positive probabilities and positive-minus-negative logit margins, plus
raw affinities and aligned row/group identities. Checkpoint tensor dtype and
shape cannot be silently converted. It verifies a serialized Task-0 reference
bank against both checkpoints; this does not prove its training data lineage.
No real FT/GPU export has been executed. Do not replace missing raw scores with
old z arrays.

`prepare_fusion.py` validates the native export receipt, unchanged reference
state and a frozen development-only policy, then copies the exact six required
arrays into a new input bundle. It does not modify the export or tune settings.
`evaluate_fusion.py` requires a completed mechanism receipt before reading a
separately hash-bound label bundle. Labels carry the same ordered row IDs and
export-receipt fingerprint. It reports the six controls' confusion matrices,
accuracy, Macro-F1, balanced accuracy, per-class metrics and old-class adjacent
recall drops. Those adjacent drops are NOT full-stream maximum-past forgetting.
Synthetic provenance propagates through every stage and cannot be relabelled
as a prospective real-data evaluation.

```text
python -B native_score_export.py native-input.json --output native-scores --device cpu
python -B prepare_fusion.py --export native-scores --policy fusion-policy.json --output score-inputs
python -B anchor_fusion.py score-inputs/manifest.json --output mechanism-results
python -B evaluate_fusion.py score-inputs/manifest.json --pilot mechanism-results --labels label-bundle.json --labels-sha256 LABEL_BUNDLE_SHA256 --output confusion-results --evidence-kind prospective-score-evaluation
```

The symbolic commands above are separate stages, not an allocation or approval.
Use `synthetic` for synthetic bundles. `test_native_score_export.py` includes a
fully executed tiny MLP checkpoint-to-score-to-confusion example, plus failure,
label-order, state/dtype, manifest and evidence-kind negative controls. Raw
score arithmetic is float32 native-style, while fusion/repair controls explicitly
use float64. These are prospective numerical targets, not proof of historical
GPU parity. Real score/label provenance and independent capture groups still
need reviewed external evidence. The next native execution must occur in an
appropriately allocated compute session, never on an HPC login node.

## Prior-art boundary

[StaR-MoE](https://arxiv.org/html/2605.17571v1) already studies expansion-induced
routing drift with frozen old experts and learns sensitivity-aware historical
routing alignment plus capacity regularization. That observation is not ours.
[GCR](https://arxiv.org/html/2601.01856v1) motivates stable geometry for routing.
[BiC](https://arxiv.org/abs/1905.13260) and
[Weight Aligning](https://arxiv.org/abs/1911.07053) already address incremental
classifier bias. A changed normalization or calibration layer is not by itself
proof of novelty. The present narrow prospective mechanism/control needs a
matched adaptation comparison and independent utility results before any
priority or performance claim. No exhaustive literature search is implied.
