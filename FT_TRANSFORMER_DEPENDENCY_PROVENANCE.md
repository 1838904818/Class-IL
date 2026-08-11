# FT-Transformer dependency record

This candidate is isolated from the active OFRA v3 experiment directory. Its
local `.venv` was created with `--system-site-packages`; only the pinned extras
in `requirements-fttransformer.lock` were installed into that environment. The
system Python and the running v3 process were not modified.

## Upstream implementation

- Repository: `https://github.com/lucidrains/tab-transformer-pytorch`
- PyPI distribution/version: `tab-transformer-pytorch==0.6.1`
- Git tag: `0.6.1`
- Git commit: `10c258aa7ecf8c7e948e38c104a87caed49a6a9a`
- License: MIT
- Wheel SHA-256: `4f350e3e4c8f17869eb4825d5b0db8006078ce3ba645d80e768f6f9281b5d263`
- `ft_transformer.py` SHA-256: `db62c6e258467bb2d85b738fe1839f0b4279ec92f0bdbb83400ddd42fadd4d42`

The source hash from the 0.6.1 wheel exactly matches the file at the pinned Git
commit. Runtime construction is fail-closed: a different installed version or
implementation hash raises an error before training.

## OFRA adapter contract

All post-cache fields are finite numerical columns and are standardised with
frozen Task-0 statistics. The upstream model therefore receives
`categories=()` and `num_continuous=feature_dim`. Its CLS readout uses
`dim_out=d_model` and becomes the embedding consumed by the existing OFRA
FamilyHead/LoRA modules, router, and exemplar selector. Those downstream
components are unchanged.

The pinned upstream 0.6.1 `FTTransformer.forward` requests and stacks every
layer's post-softmax attention matrix even when its public `return_attn`
argument is false. OFRA never consumes that tensor. The adapter therefore
replays the same pinned embedding, CLS-token, transformer, and `to_logits`
operations but calls the transformer's existing `return_attn=False` path.
This changes neither parameters nor logits/gradients; it only avoids the final
`depth x batch x heads x tokens x tokens` attention-stack allocation. Exact
output and gradient equivalence is covered by the integration test.

The candidate uses `dim=64`, `depth=4`, `heads=4`, `dim_head=16`, attention and
feed-forward dropout `0.1`, and one residual stream. One stream deliberately
disables the upstream mHC extension and retains standard residual connections.

## 8 GB GPU execution policy

Attention memory grows quadratically with the number of feature tokens. UNSW
has 194 continuous feature tokens, so the MLP defaults (`batch_size=256`,
`eval_batch_size=4096`) are unsafe for this candidate. The FT config uses a
training microbatch of 64, four exact example-weighted gradient-accumulation
steps (nominal effective batch 256), and evaluation batch 128. All three values
are present in `RunConfig`, the exposure preflight, and the protocol hash.

Before any full run, perform a one-batch CUDA probe on UNSW (the longest input)
and record peak allocated/reserved memory. If 64/128 exceeds the 8 GB budget,
use 32/64 with eight accumulation steps and create a new hashed config. Do not
change batch values inside an existing result directory.
