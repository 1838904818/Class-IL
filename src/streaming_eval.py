"""D13 — Streaming / prequential evaluation (Gap 4).

Real IDS deployments see traffic as a continuous stream, not discrete task batches.
This module evaluates each method under the *prequential* (test-then-train) protocol:

  For each mini-batch B_t:
    1. Evaluate current model on B_t  →  record accuracy acc_t
    2. Train model on B_t (possibly with exemplars / regularisation)

Metrics:
  AUT   = Area-Under-Time (trapezoidal integral of acc_t curve, normalised to [0,1])
  Final = acc_{T} at end of stream
  Drop  = peak(acc) − final(acc) (analogous to forgetting)

Simulates the same 4-task class arrival order as Class-IL, but data arrives in
`n_windows` equal-sized mini-batches per task, each evaluated before training.

Results saved to:   results/streaming_eval_results.json
Figure saved to:    results/figures/08_streaming_eval.png

Usage:
    python -m src.streaming_eval
    python -m src.streaming_eval --dataset CIC-IDS-2017 --n-windows 20
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

from src.config import (
    RESULTS_DIR, FIG_DIR, seed_all, REPLAY_BUFFER_PER_CLASS,
    EPOCHS_PER_TASK, HIDDEN,
)
from src.data import DATASET_LOADERS
from src.metrics.streaming import aut
from src.methods.base import (
    build_model, make_task_split, subset_by_classes, to_loader, train_one_task,
)
from src.methods.replay_dpmeans import _dpmeans_select
from src.methods.icarl import _herding_select, _evaluate_task_nme

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

STREAMING_METHODS = [
    "Naive",
    "Replay (random)",
    "Replay-DPMeans",
    "iCaRL",
]
METHOD_COLORS = {
    "Naive":            "#7F7F7F",
    "Replay (random)":  "#F96167",
    "Replay-DPMeans":   "#9B59B6",
    "iCaRL":            "#E91E63",
}


def _eval_all_seen(model, X_te, y_te, seen_classes):
    """Accuracy on all seen-so-far classes in test set."""
    mask = np.isin(y_te, seen_classes)
    if mask.sum() == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te[mask].astype(np.float32)))
        preds = logits.argmax(1).numpy()
    return float((preds == y_te[mask]).mean())


def _run_streaming(method_name, X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes,
                   n_windows: int = 10, buf_per_class: int = REPLAY_BUFFER_PER_CLASS):
    """Prequential streaming run for one method.

    Returns:
        acc_series: list of (window_idx, accuracy, n_seen_classes, task_idx)
        final_acc: float
        aut_score: float
    """
    model = build_model(in_dim, n_classes)
    buf_X: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []
    class_means: dict[int, np.ndarray] = {}   # for iCaRL NME
    seen_classes: list[int] = []
    use_nme = (method_name == "iCaRL")

    acc_series = []
    window_idx = 0

    import copy
    old_model = None

    for task_idx, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)
        n = len(Xi)
        window_size = max(1, n // n_windows)
        indices = np.random.permutation(n)

        for w in range(n_windows):
            start = w * window_size
            end = start + window_size if w < n_windows - 1 else n
            wX = Xi[indices[start:end]]
            wy = yi[indices[start:end]]

            # --- 1. Test BEFORE training (prequential) ---
            if seen_classes:
                if use_nme and class_means:
                    acc = float(np.mean([
                        _evaluate_task_nme(model, X_te, y_te, task_j, class_means)
                        for task_j in tasks[:task_idx + 1]
                        if any(c in seen_classes for c in task_j)
                    ]))
                else:
                    acc = _eval_all_seen(model, X_te, y_te, seen_classes)
            else:
                acc = float("nan")
            acc_series.append({
                "window": window_idx,
                "accuracy": acc,
                "task_idx": task_idx,
                "n_seen_classes": len(seen_classes),
            })
            window_idx += 1

            # --- 2. Assemble training batch ---
            if buf_X:
                Xc = np.vstack([wX] + buf_X)
                yc = np.concatenate([wy] + buf_y)
            else:
                Xc, yc = wX, wy

            # --- 3. Train (mini-batch, with optional KD for iCaRL) ---
            if method_name == "iCaRL" and old_model is not None:
                old_classes_copy = list(seen_classes)
                def kd_loss(m, bx, _by, _oc=old_classes_copy, _om=old_model):
                    if len(_oc) == 0:
                        return torch.tensor(0.0)
                    with torch.no_grad():
                        soft = F.softmax(_om(bx)[:, _oc] / 2.0, dim=1)
                    log_p = F.log_softmax(m(bx)[:, _oc] / 2.0, dim=1)
                    return F.kl_div(log_p, soft, reduction="batchmean") * 4.0
                train_one_task(model, Xc, yc, epochs=1, extra_loss_fn=kd_loss)
            else:
                train_one_task(model, Xc, yc, epochs=1)

        # --- 4. After finishing task: update buffer (end-of-task exemplar selection) ---
        # Get features of new task data for selection
        model.eval()
        with torch.no_grad():
            feats = model.extract_features(torch.from_numpy(Xi.astype(np.float32))).numpy()

        for c in task_i:
            seen_classes.append(c)
            mask = (yi == c)
            if mask.sum() == 0:
                continue
            feats_c = feats[mask]
            raw_idx = np.where(mask)[0]
            k = min(buf_per_class, len(feats_c))

            if method_name == "Naive":
                sel = np.random.choice(len(feats_c), k, replace=False)
            elif method_name == "Replay (random)":
                sel = np.random.choice(len(feats_c), k, replace=False)
            elif method_name == "Replay-DPMeans":
                sel = _dpmeans_select(feats_c, k)
            elif method_name == "iCaRL":
                sel = _herding_select(feats_c, k)
            else:
                sel = np.random.choice(len(feats_c), k, replace=False)

            if method_name == "Naive":
                # Naive: don't keep old exemplars, only track latest task
                # (still buffers for eval, but doesn't mix in training)
                pass
            else:
                buf_X.append(Xi[raw_idx[sel]])
                buf_y.append(yi[raw_idx[sel]])

        # Update iCaRL class means
        if method_name == "iCaRL" and buf_X:
            all_bX = np.vstack(buf_X)
            all_by = np.concatenate(buf_y)
            model.eval()
            with torch.no_grad():
                feats_b = model.extract_features(torch.from_numpy(all_bX.astype(np.float32))).numpy()
            for c in seen_classes:
                msk = (all_by == c)
                if msk.sum() > 0:
                    mu = feats_b[msk].mean(0)
                    class_means[c] = mu / (np.linalg.norm(mu) + 1e-8)
            old_model = copy.deepcopy(model)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad_(False)

    # Final accuracy on all seen classes
    final_acc = _eval_all_seen(model, X_te, y_te, list(range(n_classes)))
    accs = [e["accuracy"] for e in acc_series if not np.isnan(e.get("accuracy", np.nan))]
    aut_score = aut(accs)

    return acc_series, final_acc, aut_score


def run_streaming_eval(dataset_name: str = "CIC-IDS-2017", seed: int = 42,
                       n_windows: int = 10):
    loader_fn, classes_per_task = DATASET_LOADERS[dataset_name]
    print(f"\nStreaming evaluation on {dataset_name}  (seed={seed}, {n_windows} windows/task)")

    seed_all(seed)
    X_tr, y_tr, X_te, y_te, class_order = loader_fn()
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    in_dim, n_classes = X_tr_s.shape[1], len(class_order)
    tasks = make_task_split(n_classes, classes_per_task)
    print(f"  Train: {X_tr_s.shape}  Test: {X_te_s.shape}  Classes: {n_classes}")
    print(f"  Tasks: {[[class_order[c] for c in t] for t in tasks]}\n")

    results = {}
    for mname in STREAMING_METHODS:
        seed_all(seed)
        t0 = time.time()
        series, final_acc, aut_score = _run_streaming(
            mname, X_tr_s, y_tr, X_te_s, y_te, tasks, in_dim, n_classes, n_windows)
        elapsed = time.time() - t0
        print(f"  {mname:22s}  AUT={aut_score:.4f}  Final={final_acc:.4f}  t={elapsed:.1f}s")
        results[mname] = {
            "acc_series": series,
            "final_accuracy": final_acc,
            "aut": aut_score,
            "time_sec": elapsed,
        }

    out = {
        "dataset": dataset_name,
        "seed": seed,
        "n_windows": n_windows,
        "class_order": class_order,
        "tasks": [[class_order[c] for c in t] for t in tasks],
        "results": results,
    }
    out_path = RESULTS_DIR / "streaming_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    _plot_streaming(out)
    return out


def _plot_streaming(data: dict):
    dataset = data["dataset"]
    tasks = data["tasks"]
    n_windows = data["n_windows"]
    results = data["results"]
    methods = list(results.keys())
    total_windows = len(tasks) * n_windows

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left: accuracy-over-time curve
    ax = axes[0]
    for mname in methods:
        series = results[mname]["acc_series"]
        xs = [e["window"] for e in series]
        ys = [e["accuracy"] if not np.isnan(e.get("accuracy", np.nan)) else None for e in series]
        # Fill NaN at start with previous known value
        ys_clean = []
        last = None
        for v in ys:
            if v is None:
                ys_clean.append(last)
            else:
                ys_clean.append(v)
                last = v
        ax.plot(xs, ys_clean, linewidth=2, label=mname, color=METHOD_COLORS[mname])
    # Mark task boundaries
    for ti in range(1, len(tasks)):
        ax.axvline(ti * n_windows, color="grey", linestyle=":", alpha=0.5, linewidth=1)
        ax.text(ti * n_windows + 0.3, 0.02, f"T{ti}", fontsize=8, color="grey")
    ax.set_xlabel("Window index (prequential steps)")
    ax.set_ylabel("Accuracy on all seen classes (test-before-train)")
    ax.set_xlim(0, total_windows)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset} — Streaming accuracy over time")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)

    # Right: AUT bar chart
    ax = axes[1]
    auts = [results[m]["aut"] for m in methods]
    finals = [results[m]["final_accuracy"] for m in methods]
    x = np.arange(len(methods))
    w = 0.36
    bars_a = ax.bar(x - w/2, auts, w, label="AUT ↑",
                    color=[METHOD_COLORS[m] for m in methods], edgecolor="black", linewidth=0.6)
    bars_f = ax.bar(x + w/2, finals, w, label="Final Acc ↑",
                    color=[METHOD_COLORS[m] for m in methods], alpha=0.45,
                    edgecolor="black", linewidth=0.6, hatch="///")
    for b, v in zip(bars_a, auts):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    for b, v in zip(bars_f, finals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(" ")[0] for m in methods], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset} — AUT & Final accuracy")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(f"D13 — Streaming (prequential) evaluation — {dataset}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    ds_slug = dataset.replace("-", "_").lower()
    out_fig = FIG_DIR / f"08_streaming_eval_{ds_slug}.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CIC-IDS-2017",
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-windows", type=int, default=10,
                        help="Number of mini-batches per task (evaluation granularity)")
    args = parser.parse_args()
    run_streaming_eval(args.dataset, args.seed, args.n_windows)
