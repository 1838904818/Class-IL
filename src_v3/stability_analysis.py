"""A6 — Decomposition analysis of explanation drift vs forgetting.

Aggregates all shap_stability_{dataset}_seed{N}.json runs and answers:

  1. Multi-seed stability summary: per dataset x method, mean +/- std of
     cosine / Kendall tau / top-15 Jaccard (per-seed means -> std across seeds).
  2. Drift-forgetting decomposition: Spearman correlation between per-class
     instability (1 - Jaccard) and accuracy change at each transition.
  3. SILENT DRIFT RATE — the headline stat: fraction of (class, transition)
     points where accuracy is PRESERVED (acc_change > -ACC_DROP_THR) yet the
     top-15 feature set churned (Jaccard < JACCARD_THR). Accuracy-only metrics
     are blind to exactly these points.

Usage:
    python -X utf8 -u -m src_v3.stability_analysis
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats as sstats

RESULTS = Path("results")
TOP_K = 15
ACC_DROP_THR = -0.05   # acc_change above this = "accuracy preserved"
JACCARD_THR = 0.70     # top-15 Jaccard below this = "explanation churned"


def load_runs():
    runs = []
    for p in sorted(RESULTS.glob("shap_stability_*_seed*.json")):
        d = json.load(open(p))
        runs.append(d)
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("No shap_stability_*_seed*.json files found.")
        return
    print(f"Loaded {len(runs)} runs: "
          f"{sorted(set(r['config']['dataset'] for r in runs))} x "
          f"seeds {sorted(set(r['config']['seed'] for r in runs))}")

    jac_key = f"top{TOP_K}_jaccard"

    # ---- collect per-point records ----------------------------------------
    # point = (dataset, seed, method, class, transition)
    points = []
    for r in runs:
        ds, seed = r["config"]["dataset"], r["config"]["seed"]
        for method, m in r["methods"].items():
            for tr in m["transitions"]:
                for c, met in tr["per_class"].items():
                    points.append({
                        "dataset": ds, "seed": seed, "method": method,
                        "cls": int(c), "t": tr["to"],
                        "cosine": met["cosine"], "tau": met["kendall_tau"],
                        "jaccard": met[jac_key],
                        "acc_change": met.get("acc_change"),
                    })

    methods = sorted(set(p["method"] for p in points),
                     key=lambda m: ["Naive", "Replay", "Replay-DPMeans", "PILoRA"].index(m)
                     if m in ["Naive", "Replay", "Replay-DPMeans", "PILoRA"] else 99)
    datasets = sorted(set(p["dataset"] for p in points))

    # ---- 1. multi-seed summary --------------------------------------------
    summary = {}
    print("\n" + "=" * 86)
    print(" 1. Multi-seed stability (per-seed means -> mean +/- std across seeds)")
    print("=" * 86)
    for ds in datasets:
        print(f"\n--- {ds} ---")
        print(f"{'Method':<16}{'cosine':<18}{'Kendall tau':<18}{'top15-Jaccard':<18}")
        for method in methods:
            per_seed = {}
            for p in points:
                if p["dataset"] == ds and p["method"] == method:
                    per_seed.setdefault(p["seed"], []).append(p)
            cos = [float(np.mean([q["cosine"] for q in v])) for v in per_seed.values()]
            tau = [float(np.mean([q["tau"] for q in v])) for v in per_seed.values()]
            jac = [float(np.mean([q["jaccard"] for q in v])) for v in per_seed.values()]
            summary.setdefault(ds, {})[method] = {
                "n_seeds": len(per_seed),
                "cosine_mean": float(np.mean(cos)), "cosine_std": float(np.std(cos)),
                "tau_mean": float(np.mean(tau)), "tau_std": float(np.std(tau)),
                "jaccard_mean": float(np.mean(jac)), "jaccard_std": float(np.std(jac)),
            }
            s = summary[ds][method]
            print(f"{method:<16}"
                  f"{s['cosine_mean']:.4f}±{s['cosine_std']:.4f}    "
                  f"{s['tau_mean']:.4f}±{s['tau_std']:.4f}    "
                  f"{s['jaccard_mean']:.4f}±{s['jaccard_std']:.4f}")

    # ---- 2. drift vs forgetting correlation -------------------------------
    print("\n" + "=" * 86)
    print(" 2. Drift-vs-forgetting: Spearman corr( 1 - Jaccard , -acc_change )")
    print("    (positive = explanation churn co-moves with forgetting)")
    print("=" * 86)
    corr_out = {}
    for method in methods:
        pts = [p for p in points if p["method"] == method
               and p["acc_change"] is not None]
        if len(pts) < 5:
            continue
        instab = [1 - p["jaccard"] for p in pts]
        forget = [-p["acc_change"] for p in pts]
        rho, pval = sstats.spearmanr(instab, forget)
        corr_out[method] = {"spearman_rho": float(rho), "p": float(pval),
                            "n_points": len(pts)}
        print(f"{method:<16} rho={rho:+.3f}  p={pval:.4f}  (n={len(pts)})")

    # ---- 3. silent drift rate (the headline) -------------------------------
    print("\n" + "=" * 86)
    print(f" 3. SILENT DRIFT RATE: accuracy preserved (acc_change > {ACC_DROP_THR})")
    print(f"    yet top-{TOP_K} feature set churned (Jaccard < {JACCARD_THR})")
    print("=" * 86)
    silent_out = {}
    print(f"{'Method':<16}{'acc-preserved pts':<20}{'of which churned':<20}{'SILENT DRIFT %':<16}")
    for method in methods:
        pts = [p for p in points if p["method"] == method
               and p["acc_change"] is not None]
        preserved = [p for p in pts if p["acc_change"] > ACC_DROP_THR]
        churned = [p for p in preserved if p["jaccard"] < JACCARD_THR]
        rate = 100 * len(churned) / len(preserved) if preserved else float("nan")
        silent_out[method] = {
            "n_points": len(pts), "n_acc_preserved": len(preserved),
            "n_silent_drift": len(churned), "silent_drift_pct": rate,
        }
        print(f"{method:<16}{len(preserved):<20}{len(churned):<20}{rate:<16.1f}")

    out = {
        "config": {"top_k": TOP_K, "acc_drop_thr": ACC_DROP_THR,
                   "jaccard_thr": JACCARD_THR, "n_runs": len(runs)},
        "multi_seed_summary": summary,
        "drift_forgetting_correlation": corr_out,
        "silent_drift": silent_out,
    }
    with open(RESULTS / "shap_stability_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS / 'shap_stability_summary.json'}")


if __name__ == "__main__":
    main()
