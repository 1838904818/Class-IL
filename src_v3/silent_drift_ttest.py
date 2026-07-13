"""Significance test for the silent-drift headline (uses existing 5-seed data).

Following the paired forgetting analysis, this script computes each
method's pooled silent-drift rate (over the 5 datasets), giving 5 paired values
per method; then test PILoRA vs each shared-parameter baseline with a paired
t-test, Cohen's d, and a Wilcoxon signed-rank, with Holm-Bonferroni over the
m=3 comparisons. PILoRA's rate is 0 in every seed (structural), so the test
asks whether the baselines' silent drift is reliably > 0.

With five non-zero paired differences of the same sign, the minimum two-sided
exact Wilcoxon signed-rank p-value is 0.0625. Wilcoxon is therefore reported as
a distribution-free companion to the paired t-test rather than as a test that
can cross the 0.05 threshold in that configuration.

No new runs — reads results/shap_stability_{dataset}_seed{N}.json.

    python -X utf8 -u -m src_v3.silent_drift_ttest
"""
import glob
import json
from collections import defaultdict

import numpy as np
from scipy import stats as sstats

from src.config import RESULTS_DIR

ACC_THR, JAC_THR, TOP_K = -0.05, 0.70, 15
METHODS = ["Naive", "Replay", "Replay-DPMeans", "PILoRA"]
BASELINES = ["Naive", "Replay", "Replay-DPMeans"]


def per_seed_rates():
    """rate[method][seed] = pooled silent-drift rate over all datasets for that seed."""
    jac_key = f"top{TOP_K}_jaccard"
    pres = defaultdict(lambda: defaultdict(int))   # [method][seed] preserved count
    chrn = defaultdict(lambda: defaultdict(int))   # [method][seed] churned count
    seeds = set()
    for p in glob.glob(str(RESULTS_DIR / "shap_stability_*_seed*.json")):
        d = json.load(open(p))
        seed = d["config"]["seed"]
        seeds.add(seed)
        for method, m in d["methods"].items():
            for tr in m["transitions"]:
                for c, met in tr["per_class"].items():
                    ac = met.get("acc_change")
                    if ac is None or ac <= ACC_THR:
                        continue
                    pres[method][seed] += 1
                    if met[jac_key] < JAC_THR:
                        chrn[method][seed] += 1
    seeds = sorted(seeds)
    rates = {}
    for method in METHODS:
        rates[method] = [100 * chrn[method][s] / pres[method][s]
                         if pres[method][s] else 0.0 for s in seeds]
    return rates, seeds


def main():
    rates, seeds = per_seed_rates()
    print(f"Per-seed pooled silent-drift rate (%), seeds={seeds}")
    for m in METHODS:
        print(f"  {m:16}{[round(x, 1) for x in rates[m]]}  mean={np.mean(rates[m]):.2f}")

    pil = np.array(rates["PILoRA"])
    raw_p, results = [], {}
    for b in BASELINES:
        bv = np.array(rates[b])
        diff = bv - pil
        t, p = sstats.ttest_rel(bv, pil)
        d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
        try:
            w_stat, w_p = sstats.wilcoxon(bv, pil)
        except ValueError:
            w_p = float("nan")
        raw_p.append(p)
        results[b] = {"mean_baseline": float(np.mean(bv)), "mean_pilora": float(np.mean(pil)),
                      "t": float(t), "p_raw": float(p), "cohens_d": d,
                      "wilcoxon_p": float(w_p)}

    # Holm-Bonferroni over the 3 comparisons
    order = np.argsort(raw_p)
    m = len(raw_p)
    holm = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, raw_p[idx] * (m - rank))
        adj = max(adj, prev)
        holm[idx] = adj
        prev = adj
    for i, b in enumerate(BASELINES):
        results[b]["p_holm"] = float(holm[i])
        results[b]["sig_holm"] = bool(holm[i] < 0.05)

    print("\n" + "=" * 76)
    print(" SILENT-DRIFT SIGNIFICANCE — PILoRA (0%) vs baselines, 5-seed paired")
    print("=" * 76)
    print(f"{'comparison':<26}{'baseline%':<11}{'t':<9}{'p_holm':<11}{'d':<9}{'Wilcoxon':<10}")
    for b in BASELINES:
        r = results[b]
        sig = "***" if r["p_holm"] < 0.001 else "**" if r["p_holm"] < 0.01 else "*" if r["p_holm"] < 0.05 else "ns"
        print(f"{'PILoRA<'+b:<26}{r['mean_baseline']:<11.1f}{r['t']:<9.2f}"
              f"{r['p_holm']:<11.4g}{r['cohens_d']:<9.2f}{r['wilcoxon_p']:<10.4g}{sig}")

    out = RESULTS_DIR / "silent_drift_ttest.json"
    json.dump({"per_seed_rates": rates, "seeds": seeds,
               "test": "paired t-test PILoRA vs baseline, Holm m=3, pooled over 5 datasets",
               "results": results}, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")
    print("Note: n=5 → Wilcoxon floors at 0.0625; read significance from t-test/Holm.")


if __name__ == "__main__":
    main()
