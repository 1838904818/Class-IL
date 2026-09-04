# Malaya attribution/ETG v9: isolated offline source snapshot

This directory supplies the offline scientific source for DICC Job 434747,
with its own dependency copies. It is **not a finished five-seed result, a
new predictive model, or a ready-to-submit HPC package**. It does not replace
the older analysis entry point at repository root.

## Why a separate snapshot?

The campaign-bound `streaming_full/data.py` and `validation.py` differ from
the current general-purpose repository versions. Passing unit tests with
those newer dependencies would not establish byte-identical reproduction.
The copies here retain the campaign source bytes. No analysis, classifier,
router, gradient, threshold, or data-selection code was changed for release.

`SOURCE_PROVENANCE.json` records each source hash, released hash, and origin.
The six offline test modules retain all 46 tests. Two fixture lookup paths
are adapted; their assertions are unchanged. `scheduler_contract.sbatch.txt`
is the submitted script **as a read-only test fixture**, with only its two
personal log paths replaced by relative paths. Never submit or execute it:
it references deployment inputs not included here.

Seven network-publication tests and the account-specific W&B publisher are
not included in this offline snapshot. The original campaign suite has 53
tests; **46 is not a claim that all 53 were rerun from this distribution**.
The existing scheduled job still uses its reviewed publisher. Publication
credentials, site-specific binding records, datasets, and checkpoints are
not distributed by this source snapshot.

## Verify and test without data or network access

From this directory, using an existing Python environment:

```text
python -B verify_snapshot.py
python -B verify_snapshot.py --self-test
```

The second command uses synthetic fixtures, hides CUDA, limits Torch to two
CPU threads, validates source hashes before and after tests, and checks that
all imported project modules came from this directory and its inventory.
It does not run SHAP on a dataset, contact W&B, or submit jobs. Run it locally
or within a suitable compute allocation, never as heavy login-node work.

Local release QA used Python 3.11, Torch 2.11.0+cu128, NumPy 2.4.6.
The scheduled experiment instead checks Python 3.11, Torch 2.6.0+cu118,
NumPy 2.2.6, SHAP 0.51.0, and W&B 0.23.0. Those are different environments;
unit-test success here is not numerical equivalence to the scheduled run.
The FT adapter separately checks `tab-transformer-pytorch` 0.6.1 and the
upstream implementation SHA-256 in `ofra_encoders/ft_transformer.py`.

The provenance inventory is a file-integrity check, not an independent
signature or proof of scientific validity. The Git commit identifies this
inventory. Changing a file and its inventory cannot prove authenticity.

## Scientific scope

The frozen score is `joint_c = p_c + 0.5*z_c`; the explanation target is
`joint_c - max(joint_other)` on fixed true-class probe batches reconstructed
on CPU. The three methods are Expected Gradients (SHAP approximation),
Feature Ablation, and Gradient x Input. ETG is a proposed offline,
non-suppressing explanation-governance ledger; it does not improve the
classifier by changing predictions or inventing new attack classes.

Same-batch forward agreement is checked at absolute tolerance `1e-6`.
The squared-distance floor `1e-12` stabilizes only the differentiable
surrogate at exact centroid matches. The exact forward value is restored
separately; this does not establish derivative equivalence. Archived GPU
scores and different batch partitions are **not** claimed equivalent.

Primary top-k 15, Jaccard threshold 0.70, and allowed recall drop 0.05 are
study-specific operational choices, not externally established standards.
The bundled `formal_v2_explanation_etg/DRIFT_ETG_METHOD_PROTOCOL.md` is the
byte-preserved method document; its legacy results are not v9 results.
Read its score descriptions with the narrower CPU/class-batch scope above.

## Input and reproduction boundary

The real analysis requires authorized, hash-bound source training results,
protocols, fixed probe/background manifests, data shards, checkpoint weights,
and router states. They are not replaced with synthetic data for reporting.
Use the modules' `--help` to inspect their input contracts; CLI help and unit
tests alone are not a completed full-data reproduction. The campaign-specific
`verify_multiseed_bindings` intentionally checks its registered dataset,
seeds, training job IDs, and storage layout. It is not a generic import tool.

`streaming_full/__init__.py` imports `validation.py`, which also imports
`recovery.py`. The original project manifest omitted `recovery.py`. Its
current remote hash was checked against the included local copy on
2026-09-04, and it is inventoried here as a supplementary import. **That
retrospective check cannot prove its pre-submission bytes.** The module
defines recovery helpers; this analysis does not invoke model-training
recovery. The remaining historical provenance gap must accompany the final
campaign audit rather than being silently treated as pre-bound evidence.

No five-seed aggregate, method robustness conclusion, or model superiority
claim is introduced by publishing this source snapshot.
