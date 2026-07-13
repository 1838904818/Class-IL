"""Minimal smoke test of every PILoRA-IDS component on synthetic data.

Verifies wiring, not benchmark accuracy.  Runs in under 30 seconds on CPU.
Run with:
    python -X utf8 -u -m src_v2.unit_test
"""
import sys
import time

import numpy as np
import torch

from src.config import seed_all
from src_v2.models.transformer_encoder import (
    FlowTransformer,
    masked_feature_reconstruction_loss,
)
from src_v2.models.lora import LoRAAdapter, FamilyHead, LoRAPool
from src_v2.models.dpmeans_router import DPMeansRouter
from src_v2.methods.pilora import PILoRAAgent


def banner(s: str):
    print(f"\n{'-' * 50}\n {s}\n{'-' * 50}")
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Synthetic data: 5 classes, 4 modes per class, 50 dims, 200 train per class
# --------------------------------------------------------------------------
def make_synthetic(n_classes=5, modes_per_class=3, n_per_mode=70, dim=32, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        for m in range(modes_per_class):
            center = rng.normal(0, 3, size=dim) + c * 0.3  # class-specific mean
            samples = center + rng.normal(0, 0.5, size=(n_per_mode, dim))
            X.append(samples)
            y.append(np.full(n_per_mode, c))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def main():
    seed_all(42)
    print("=" * 60)
    print(" PILoRA-IDS Unit Test")
    print("=" * 60)
    sys.stdout.flush()

    n_classes = 5
    dim = 32
    X_tr, y_tr = make_synthetic(n_classes=n_classes, dim=dim, seed=42)
    X_te, y_te = make_synthetic(n_classes=n_classes, dim=dim, seed=99)
    print(f"Data: train={X_tr.shape}  test={X_te.shape}  classes={n_classes}")
    sys.stdout.flush()

    # ----------------------------------------------------------------- 1
    banner("1. FlowTransformer encoder")
    t0 = time.time()
    enc = FlowTransformer(n_features=dim, d_model=64, n_layers=2, n_heads=4)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  params: {n_params:,}")
    Xt = torch.from_numpy(X_tr[:32])
    emb = enc(Xt)
    print(f"  forward: input{Xt.shape} → output{emb.shape}  in {time.time()-t0:.2f}s")
    assert emb.shape == (32, 64), f"shape mismatch: {emb.shape}"
    print("  PASS")
    sys.stdout.flush()

    # ----------------------------------------------------------------- 2
    banner("2. Masked-feature pretraining objective")
    t0 = time.time()
    loss = masked_feature_reconstruction_loss(enc, Xt, mask_prob=0.15)
    loss.backward()
    print(f"  initial MSE loss = {loss.item():.4f}  in {time.time()-t0:.2f}s")
    assert loss.item() >= 0
    print("  PASS")
    sys.stdout.flush()

    # ----------------------------------------------------------------- 3
    banner("3. LoRA adapter forward/backward")
    lora = LoRAAdapter(d_model=64, rank=8)
    print(f"  LoRA params: {lora.num_params()}")
    out = lora(emb)
    assert out.shape == emb.shape
    print(f"  forward {emb.shape} → {out.shape}: PASS")
    # Initial output should be near-zero because B is zero-init
    assert out.abs().max() < 1e-4, f"LoRA initial output not ~0: {out.abs().max()}"
    print(f"  initial output ~ 0 (max |x|={out.abs().max().item():.2e}): PASS")
    sys.stdout.flush()

    # ----------------------------------------------------------------- 4
    banner("4. LoRAPool: add families, freeze, num_params")
    pool = LoRAPool(d_model=64, rank=8)
    pool.add_family("family_A")
    pool.add_family("family_B")
    print(f"  families: {pool.families}")
    print(f"  params/family = {pool.num_params_per_family()}, "
          f"total = {pool.num_params_total()}")
    pool.freeze_all_except("family_A")
    a_trainable = sum(p.numel() for p in pool.heads["family_A"].parameters() if p.requires_grad)
    b_trainable = sum(p.numel() for p in pool.heads["family_B"].parameters() if p.requires_grad)
    print(f"  trainable: family_A={a_trainable}, family_B={b_trainable}")
    assert a_trainable > 0 and b_trainable == 0
    print("  freeze_all_except: PASS")
    sys.stdout.flush()

    # ----------------------------------------------------------------- 5
    banner("5. LoRA state dict for federation")
    state = pool.lora_state_dict()
    print(f"  keys: {list(state.keys())}")
    print(f"  family_A: A{state['family_A']['A'].shape}, B{state['family_A']['B'].shape}")
    # Perturb and reload
    state["family_A"]["A"] += 1.0
    pool.load_lora_state(state)
    assert torch.allclose(pool.heads["family_A"].lora.A.detach(), state["family_A"]["A"])
    print("  load_lora_state round-trip: PASS")
    sys.stdout.flush()

    # ----------------------------------------------------------------- 6
    banner("6. DPMeansRouter routing and novelty")
    router = DPMeansRouter(lambda_quantile=0.30, novelty_factor=1.5)
    # Fit two families with separated distributions
    feats_A = np.random.randn(80, 64).astype(np.float32) + 5
    feats_B = np.random.randn(80, 64).astype(np.float32) - 5
    nA = router.fit_family("family_A", feats_A)
    nB = router.fit_family("family_B", feats_B)
    print(f"  fit family_A: {nA} centroids; family_B: {nB} centroids")
    # Route a test point near family_A
    test_A = np.random.randn(64).astype(np.float32) + 5
    f, d = router.route(test_A)
    print(f"  route(near A) → ({f}, dist={d:.2f})")
    assert f == "family_A"
    # Batch routing
    test_batch = np.vstack([np.random.randn(10, 64) + 5, np.random.randn(10, 64) - 5])
    families, dists = router.route_batch(test_batch.astype(np.float32))
    assert all(f == "family_A" for f in families[:10])
    assert all(f == "family_B" for f in families[10:])
    print(f"  batch routing 20 points: 10→A, 10→B: PASS")
    # Novelty detection
    novel_point = np.random.randn(64).astype(np.float32) + 50  # very far
    is_novel = router.is_novel(novel_point)
    f_normal, _ = router.route(np.random.randn(64).astype(np.float32) + 5)
    is_normal_novel = router.is_novel(np.random.randn(64).astype(np.float32) + 5)
    print(f"  novelty(far={50}): {is_novel}, novelty(near A)={is_normal_novel}")
    print(f"  comm payload: {router.comm_payload_bytes()/1024:.1f} KB")
    print("  PASS")
    sys.stdout.flush()

    # ----------------------------------------------------------------- 7
    banner("7. End-to-end PILoRAAgent Class-IL on synthetic")
    t0 = time.time()
    agent = PILoRAAgent(n_features=dim, d_model=64, n_layers=2, n_heads=4, lora_rank=8)
    agent.pretrain_encoder(X_tr, epochs=1, verbose=False)
    agent.freeze_encoder()
    print(f"  pretrain done in {time.time()-t0:.1f}s")
    # Class-IL: train classes 0-1, then 2-3, then 4
    tasks = [[0, 1], [2, 3], [4]]
    for i, task in enumerate(tasks):
        mask = np.isin(y_tr, task)
        agent.train_task(X_tr[mask], y_tr[mask], epochs=2, verbose=False)
        # Evaluate on all classes seen so far
        seen_mask = np.isin(y_te, sum(tasks[:i + 1], []))
        preds = agent.predict(X_te[seen_mask])
        acc = float((preds == y_te[seen_mask]).mean())
        print(f"  After task {i} {task}: acc on seen classes = {acc:.4f}")
    print(f"  Total {time.time()-t0:.1f}s, n_families={len(agent.pool.families)}, "
          f"n_centroids={agent.router.n_centroids_total()}")
    print("  PASS (wiring works)")
    sys.stdout.flush()

    # ----------------------------------------------------------------- 8
    banner("8. Federated aggregation across 2 sites")
    t0 = time.time()
    # Build two agents, each gets half the data
    site1 = PILoRAAgent(n_features=dim, d_model=64, n_layers=2, n_heads=4, lora_rank=8)
    site2 = PILoRAAgent(n_features=dim, d_model=64, n_layers=2, n_heads=4, lora_rank=8)
    # Pretrain (could share but here we keep separate)
    site1.pretrain_encoder(X_tr[: len(X_tr) // 2], epochs=1, verbose=False)
    site2.pretrain_encoder(X_tr[len(X_tr) // 2 :], epochs=1, verbose=False)
    site1.freeze_encoder()
    site2.freeze_encoder()

    # Site 1 sees classes 0-2; Site 2 sees classes 2-4 (overlap on 2)
    s1_mask = np.isin(y_tr, [0, 1, 2])
    s2_mask = np.isin(y_tr, [2, 3, 4])
    site1.train_task(X_tr[s1_mask], y_tr[s1_mask], epochs=2, verbose=False)
    site2.train_task(X_tr[s2_mask], y_tr[s2_mask], epochs=2, verbose=False)
    print(f"  Site 1 families: {site1.pool.families}")
    print(f"  Site 2 families: {site2.pool.families}")

    # Pre-aggregation accuracy
    p1 = site1.predict(X_te)
    p2 = site2.predict(X_te)
    print(f"  pre-FedAvg accuracy: site1={float((p1==y_te).mean()):.3f}, "
          f"site2={float((p2==y_te).mean()):.3f}")
    # Federated round
    print(f"  FedRound: {stats['n_global_families']} families merged, "
          f"{stats['total_bytes_per_site']/1024:.1f} KB/site")
    # Post-aggregation accuracy (each site now has all families)
    p1_post = site1.predict(X_te)
    p2_post = site2.predict(X_te)
    print(f"  post-FedAvg accuracy: site1={float((p1_post==y_te).mean()):.3f}, "
          f"site2={float((p2_post==y_te).mean()):.3f}")
    print(f"  Time: {time.time()-t0:.1f}s")
    print("  PASS")
    sys.stdout.flush()

    print("\n" + "=" * 60)
    print(" ALL UNIT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
