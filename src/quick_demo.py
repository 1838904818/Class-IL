"""
QUICK DEMO — runnable in ~90 seconds on CPU.

For live demonstration: runs ONLY NSL-KDD, ONLY Naive vs Replay,
generates a single side-by-side figure showing catastrophic forgetting
vs. mitigation.

Run from project root: `python -m src.quick_demo`
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

from src.config import (
    SEED, FIG_DIR,
    HIDDEN, EPOCHS_PER_TASK, REPLAY_BUFFER_PER_CLASS,
    seed_all,
)
from src.data.nslkdd import load_nslkdd
from src.methods.base import (
    build_model, train_one_task, subset_by_classes, make_task_split,
)
from src.metrics import evaluate_task


def run_naive(X_tr, y_tr, X_te, y_te, tasks, n_classes, seed):
    seed_all(seed)
    model = build_model(X_tr.shape[1], n_classes)
    task0_acc = []
    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)
        train_one_task(model, Xi, yi, epochs=EPOCHS_PER_TASK)
        task0_acc.append(evaluate_task(model, X_te, y_te, tasks[0]))
    return task0_acc


def run_replay(X_tr, y_tr, X_te, y_te, tasks, n_classes, seed):
    seed_all(seed)
    model = build_model(X_tr.shape[1], n_classes)
    task0_acc = []
    buf_X, buf_y = [], []
    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)
        Xc = np.vstack([Xi] + buf_X) if buf_X else Xi
        yc = np.concatenate([yi] + buf_y) if buf_y else yi
        train_one_task(model, Xc, yc, epochs=EPOCHS_PER_TASK)
        for c in task_i:
            m = (yi == c)
            if m.sum() == 0:
                continue
            idx = np.random.choice(
                np.where(m)[0],
                size=min(REPLAY_BUFFER_PER_CLASS, m.sum()),
                replace=False,
            )
            buf_X.append(Xi[idx])
            buf_y.append(yi[idx])
        task0_acc.append(evaluate_task(model, X_te, y_te, tasks[0]))
    return task0_acc


def main():
    print("=" * 68)
    print("GAP 3 — Quick Demo: Naive vs Replay on NSL-KDD (Class-IL)")
    print("=" * 68)

    t0 = time.time()
    print("\n[1/4] Loading NSL-KDD ...")
    X_tr, y_tr, X_te, y_te, class_order = load_nslkdd()
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    tasks = make_task_split(len(class_order), 1)
    print(f"      {X_tr.shape[0]} train / {X_te.shape[0]} test samples")
    print(f"      Tasks: {[[class_order[c] for c in t] for t in tasks]}")
    print(f"      ({time.time() - t0:.1f}s)")

    print("\n[2/4] Running Naive (no anti-forgetting) ...")
    t1 = time.time()
    naive = run_naive(X_tr, y_tr, X_te, y_te, tasks, len(class_order), SEED)
    print(f"      Task-0 accuracy over time: {[f'{a:.3f}' for a in naive]}  ({time.time() - t1:.1f}s)")

    print("\n[3/4] Running Replay (200 ex/cls) ...")
    t2 = time.time()
    replay = run_replay(X_tr, y_tr, X_te, y_te, tasks, len(class_order), SEED)
    print(f"      Task-0 accuracy over time: {[f'{a:.3f}' for a in replay]}  ({time.time() - t2:.1f}s)")

    print("\n[4/4] Plotting ...")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    xs = list(range(1, len(tasks) + 1))
    ax.plot(xs, naive, "o-", linewidth=2.5, markersize=10,
            color="#B85042", label="Naive (no anti-forgetting)")
    ax.plot(xs, replay, "s-", linewidth=2.5, markersize=10,
            color="#2C5F2D", label="Replay (200 ex/class)")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"T{i}\n+{'/'.join([class_order[c] for c in tasks[i - 1]])}" for i in xs],
        fontsize=10,
    )
    ax.set_xlabel("After training each Class-IL task", fontsize=12)
    ax.set_ylabel("Accuracy on Task-0 (Normal + DoS)", fontsize=12)
    ax.set_title(
        "Gap 3 — Catastrophic Forgetting Demonstrated on Network Traffic\n"
        "(NSL-KDD, Class-Incremental Learning, MLP backbone)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.95)
    ax.annotate(f"Naive forgets to {naive[-1]:.2%}", xy=(xs[-1], naive[-1]),
                xytext=(xs[-1] - 1.0, naive[-1] + 0.10),
                fontsize=10, color="#B85042",
                arrowprops=dict(arrowstyle="->", color="#B85042"))
    ax.annotate(f"Replay retains {replay[-1]:.2%}", xy=(xs[-1], replay[-1]),
                xytext=(xs[-1] - 1.4, replay[-1] - 0.18),
                fontsize=10, color="#2C5F2D",
                arrowprops=dict(arrowstyle="->", color="#2C5F2D"))
    fig.tight_layout()
    out_path = FIG_DIR / "00_quick_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {out_path}")

    print(f"\nTotal demo time: {time.time() - t0:.1f}s")
    print("\n>>> Key takeaway for audience:")
    print(f"    Naive sequential learning drops Task-0 accuracy from"
          f" {naive[0]:.2%} -> {naive[-1]:.2%}")
    print(f"    Replay with 200 ex/class holds Task-0 accuracy at"
          f" {replay[-1]:.2%}.")
    print(f"    Gap to retention: {(replay[-1] - naive[-1]) * 100:.1f} percentage points.")


if __name__ == "__main__":
    main()
