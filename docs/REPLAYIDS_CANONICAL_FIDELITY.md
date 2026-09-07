# ReplayIDS canonical-batching CPU fidelity

7 September 2026. This executes the local part of Stage A in the
[score-validation protocol](REPLAYIDS_SCORE_VALIDATION_PROTOCOL.md). All twenty
primary checkpoints were checked. **The cross-environment fidelity gate failed**;
no calibration was fitted and no protected historical result was replaced.

## Correct legacy batching and scope

The verifier obtains `eval_batch_size=512` from each hash-bound original protocol.
It reproduces `_embed` by chunking normalized features at 512, then computes Head
and Router scores on the whole checkpoint probe matrix. It does not call the
generic reload helper's `score()` method, whose internal batching differs from
the original monitor. The earlier batch32/898 checks remain labelled stress
diagnostics, not canonical monitor-path reproduction.

The same original source files, fixed data, frozen Task-0 normalization, stored
model weights and cap3000 centroids were used. All 60 checkpoint/state/score files,
five protocols, original runtime files, probe manifest, eight local official-test
shards and selected probe row hashes were verified. No training or test-label
selection was performed. The check covers seeds 1, 2, 3, 4 and 42, checkpoints
000-003 and only their seen-class fixed probes, not every official-test row.

Reconstruction ran locally on CPU, PyTorch 2.11.0+cu128 and NumPy 2.4.6; the
original run used A100 CUDA, PyTorch 2.6.0+cu118 and NumPy 2.2.6. The CUDA suffix
of the local PyTorch package does not mean this check used a GPU. The upstream
FT-Transformer version and implementation source were checked. The environments
are explicitly **not matched**.

## Results across five seeds

The fixed gate remains `atol=1e-4`, `rtol=1e-5` on every Head/Router/Joint score
cell plus zero argmax differences. Passing a tolerance is separate from bit-exact
equality. Maximum errors below are maxima over all five seeds at that checkpoint.

| Checkpoint | Probes per seed | Checkpoints passing / 5 | Head max absolute error | Router max absolute error | Joint max absolute error | Argmax changes across seeds |
|---|---:|---:|---:|---:|---:|---:|
| 000 | 256 | 5 | 0.00000703 | 0.00001729 | 0.00000870 | 0 |
| 001 | 512 | 0 | 0.00002033 | 0.00876093 | 0.00438046 | 1 |
| 002 | 768 | 0 | 0.00002044 | 0.01024866 | 0.00512433 | 0 |
| 003 | 898 | 0 | 0.00002044 | 0.02102029 | 0.01051009 | 1 |

Head scores passed in every checkpoint; Router/Joint failed in all fifteen
post-initial checkpoint instances. The five initial two-class checkpoints passed
the specified tolerance. Two-class population standardization tends to collapse
distinct distances toward opposite signed unit scores, so this initial-stage
pass does not establish that the underlying raw distances are exactly reproduced.

Both argmax changes concern the **same sample ID** in seed 3 at checkpoints 001
and 003. The saved prediction was class 2 (DoS Hulk), reconstructed as class 0
(Benign). Its saved top-two score margins were approximately 0.00026190 and
0.00026202, respectively. The report retains the complete sample ID, prediction
vector hashes, per-score hashes and both old/new margins.

These are two checkpoint evaluations of one row, not two independent new samples.
They are not a classification-error estimate, explanation drift rate, ETG
admission disagreement or revised forgetting result. In particular, do not mix
this canonical result with the earlier stress-check finding in seed 4.

## Consequence

Canonical batching is necessary but did not eliminate the observed cross-platform
score discrepancy. The previous [fixed-embedding arithmetic isolation](REPLAYIDS_ROUTER_ARITHMETIC.md)
remains relevant; it does not identify every CPU/GPU or dependency-version effect.
A historical-environment GPU verification is still outstanding and must bind its
actual environment and kernel settings. This local check cannot prove that the
original GPU run itself was non-reproducible in its original environment.

The versioned direct64 candidate remains separate and unpromoted. It must not be
used silently to replace old scores or declare the legacy fidelity gate passed.
Full-test performance, earlier-checkpoint forgetting and actual-score SHAP/ETG
remain separate evaluations after their stated numerical and protocol gates.

## Evidence and rerun

[CANONICAL_CPU_FIDELITY.json](../results/replayids-canonical-fidelity/CANONICAL_CPU_FIDELITY.json)
contains all twenty comparisons, original hashes and environment distinctions.
No raw traffic, weights or embeddings are included in the public evidence package.

```text
python tools/check_replayids_canonical_fidelity.py --runtime ORIGINAL_RUNTIME --inputs INPUT_DIR --data DATA_DIR --protocols PROTOCOL_DIR --registry PROTECTED_REGISTRY --output canonical.json --device cpu
python tests/test_replayids_canonical_fidelity.py
```

The unit test explicitly checks chunked encoder calls versus whole-probe Head and
Router calls. Other tests reject invalid score/class-axis inputs and check report
source binding. Passing these tests is not a model-fidelity pass. Source and
protected inputs must be supplied separately under the applicable governance.
