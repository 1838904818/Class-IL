"""Paired t-tests (forgetting + accuracy) with Holm-Bonferroni correction.

Updated analysis:
  - Forgetting: PILoRA vs Phase-I best (lowest mean forgetting) per dataset,
    paired by seed, Holm-Bonferroni corrected across the dataset family.
  - Accuracy:   PILoRA vs Phase-I best (highest mean accuracy) per dataset,
    same protocol — this quantifies the accuracy-cost side of the
    Pareto trade-off with corrected significance.
  - Robust PILoRA loader: picks the candidate file with the MOST seeds for a
    dataset (fixes glob-order shadowing between old single- and new 5-seed
    files).
  - Summary block: per-dataset forgetting-reduction multiples + geometric
    mean, accuracy deltas in pp, correction details.

Outputs:
  results/pilora_ttest.json     — list of forgetting rows (back-compatible
                                  schema + new fields p_holm / significance_holm
                                  / reduction_multiple)
  results/pilora_ttest_v2.json  — full structure incl. accuracy family + summary

Usage:
    python -X utf8 -u -m src_v2.paired_ttest
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path("results")

PHASE1_BASELINES = ["Replay", "Replay-Herding", "Replay-DPMeans", "iCaRL"]
DATASETS = ["NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CIC-IDS-2018"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_pilora_multi(ds):
    """Return dict with per-seed forget/acc lists for `ds`.

    Scans all pilora_v05_*.json files and picks the entry with the MOST
    seeds (ties broken by later filename), so a new 5-seed file can never be
    shadowed by an older shorter run that happens to sort first.
    """
    best = None
    for p in sorted(RESULTS.glob("pilora_v05_*.json")):
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if ds not in d:
            continue
        entry = d[ds]
        per_f = entry.get("per_seed_forget")
        if not per_f:
            continue
        if best is None or len(per_f) >= len(best["per_seed_forget"]):
            best = {
                "per_seed_forget": per_f,
                "per_seed_acc": entry.get("per_seed_acc"),
                "seeds": entry.get("seeds"),
                "source": p.name,
            }
    return best


def load_phase1_multi(ds):
    """Return dict method -> {forget: [...], acc: [...]} from {ds}_results_5seed.json."""
    p = RESULTS / f"{ds}_results_5seed.json"
    if not p.exists():
        return None
    d = json.load(open(p))["results"]
    out = {}
    for m, v in d.items():
        per = v.get("per_seed", {})
        f = per.get("avg_forgetting")
        a = per.get("avg_accuracy")
        if f is not None:
            out[m] = {"forget": f, "acc": a}
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def paired_test(pilora_arr, base_arr):
    """Paired t-test + Cohen's d (paired) for pilora - base."""
    diff = pilora_arr - base_arr
    t, p = stats.ttest_rel(pilora_arr, base_arr)
    sd = float(np.std(diff, ddof=1))
    d = float(np.mean(diff) / sd) if sd > 0 else 0.0
    return float(t), float(p), d


def holm_bonferroni(pvals):
    """Step-down Holm-Bonferroni. Returns adjusted p-values (same order as input).

    adj_p_(i) = max_{j<=i} ( (m - j) * p_(j) ), clipped at 1, where p_(1)<=...<=p_(m)
    (0-based j). Monotonicity enforced.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 88)
    print(" Paired t-tests: PILoRA vs Phase-I best  (forgetting + accuracy, Holm-Bonferroni)")
    print("=" * 88)

    forget_rows, acc_rows = [], []

    for ds in DATASETS:
        pilora = load_pilora_multi(ds)
        if pilora is None:
            print(f"{ds:<14} NO PILoRA data")
            continue
        pf = np.array(pilora["per_seed_forget"], dtype=float)
        pa = (np.array(pilora["per_seed_acc"], dtype=float)
              if pilora.get("per_seed_acc") else None)

        p1 = load_phase1_multi(ds)
        if p1 is None:
            # Single-seed fallback (should disappear once *_results_5seed.json exists)
            single = json.load(open(RESULTS / f"{ds}_results.json"))["results"]
            method, best_f = min(
                ((m, single.get(m, {}).get("avg_forgetting", 1.0)) for m in PHASE1_BASELINES),
                key=lambda x: x[1],
            )
            forget_rows.append({
                "dataset": ds, "test": "single-seed",
                "pilora_mean": float(pf.mean()),
                "phase1_method": method, "phase1_mean": best_f,
                "delta": float(pf.mean()) - best_f,
                "reduction_multiple": best_f / float(pf.mean()) if pf.mean() > 0 else None,
                "pilora_source": pilora["source"],
            })
            print(f"{ds:<14} Phase-I 5-seed missing -> single-seed fallback ({method})")
            continue

        # ---------- forgetting family: lowest-mean-forget baseline ----------
        cands = {m: np.array(v["forget"], dtype=float) for m, v in p1.items()
                 if m in PHASE1_BASELINES and v["forget"] and len(v["forget"]) == len(pf)}
        if not cands:
            print(f"{ds:<14} can't pair forgetting (no matching baseline arrays)")
            continue
        bm = min(cands, key=lambda m: cands[m].mean())
        bf = cands[bm]
        t, p, d = paired_test(pf, bf)
        forget_rows.append({
            "dataset": ds, "test": "paired_t",
            "pilora_per_seed": pf.tolist(), "phase1_per_seed": bf.tolist(),
            "pilora_mean": float(pf.mean()), "phase1_method": bm,
            "phase1_mean": float(bf.mean()),
            "delta": float(pf.mean() - bf.mean()),
            "reduction_multiple": float(bf.mean() / pf.mean()) if pf.mean() > 0 else None,
            "t": t, "p_value": p, "cohen_d": d,
            "pilora_source": pilora["source"],
        })

        # ---------- accuracy family: highest-mean-acc baseline ----------
        if pa is not None:
            acands = {m: np.array(v["acc"], dtype=float) for m, v in p1.items()
                      if m in PHASE1_BASELINES and v.get("acc") and len(v["acc"]) == len(pa)}
            if acands:
                am = max(acands, key=lambda m: acands[m].mean())
                ba = acands[am]
                t2, p2, d2 = paired_test(pa, ba)
                acc_rows.append({
                    "dataset": ds, "test": "paired_t",
                    "pilora_per_seed": pa.tolist(), "phase1_per_seed": ba.tolist(),
                    "pilora_mean": float(pa.mean()), "phase1_method": am,
                    "phase1_mean": float(ba.mean()),
                    "delta_pp": float((pa.mean() - ba.mean()) * 100),
                    "t": t2, "p_value": p2, "cohen_d": d2,
                })

    # ---------------- Holm-Bonferroni per metric family ----------------
    for rows in (forget_rows, acc_rows):
        paired = [r for r in rows if r["test"] == "paired_t"]
        if paired:
            adj = holm_bonferroni([r["p_value"] for r in paired])
            for r, ph in zip(paired, adj):
                r["p_holm"] = ph
                r["significance"] = sig_stars(r["p_value"])
                r["significance_holm"] = sig_stars(ph)

    # ---------------- print tables ----------------
    print(f"\n--- FORGETTING (lower is better) ---")
    print(f"{'Dataset':<14}{'PILoRA':<10}{'Best Phase-I':<26}{'mult':<7}{'t':<9}{'p_raw':<11}{'p_holm':<11}{'d':<8}")
    for r in forget_rows:
        if r["test"] == "paired_t":
            print(f"{r['dataset']:<14}{r['pilora_mean']:<10.4f}"
                  f"{r['phase1_method']+' '+format(r['phase1_mean'],'.4f'):<26}"
                  f"{r['reduction_multiple']:<7.2f}{r['t']:<+9.2f}"
                  f"{r['p_value']:<11.2e}{r['p_holm']:<11.2e}{r['cohen_d']:<+8.2f} {r['significance_holm']}")
        else:
            print(f"{r['dataset']:<14}{r['pilora_mean']:<10.4f}"
                  f"{r['phase1_method']+' '+format(r['phase1_mean'],'.4f'):<26}"
                  f"{r['reduction_multiple']:<7.2f}{'1-seed fallback'}")

    print(f"\n--- ACCURACY (PILoRA cost, pp) ---")
    for r in acc_rows:
        print(f"{r['dataset']:<14}{r['pilora_mean']:<10.4f}"
              f"{r['phase1_method']+' '+format(r['phase1_mean'],'.4f'):<26}"
              f"{r['delta_pp']:<+8.1f}pp  t={r['t']:<+8.2f}p_holm={r['p_holm']:.2e} {r['significance_holm']}")

    mults = [r["reduction_multiple"] for r in forget_rows if r.get("reduction_multiple")]
    paired_mults = [r["reduction_multiple"] for r in forget_rows
                    if r["test"] == "paired_t" and r.get("reduction_multiple")]
    summary = {
        "correction": "Holm-Bonferroni, applied separately within each metric family "
                      f"(forgetting m={len([r for r in forget_rows if r['test']=='paired_t'])}, "
                      f"accuracy m={len(acc_rows)})",
        "geomean_reduction_all": geomean(mults),
        "geomean_reduction_paired_only": geomean(paired_mults),
        "n_datasets_paired": len(paired_mults),
        "accuracy_delta_pp": {r["dataset"]: round(r["delta_pp"], 1) for r in acc_rows},
    }
    print(f"\nGeomean forgetting reduction: paired-only={summary['geomean_reduction_paired_only']:.2f}x"
          f"  all={summary['geomean_reduction_all']:.2f}x  (paired n={summary['n_datasets_paired']})")

    with open(RESULTS / "pilora_ttest.json", "w") as f:
        json.dump(forget_rows, f, indent=2)
    with open(RESULTS / "pilora_ttest_v2.json", "w") as f:
        json.dump({"forgetting": forget_rows, "accuracy": acc_rows, "summary": summary}, f, indent=2)
    print(f"\nSaved: {RESULTS/'pilora_ttest.json'} (back-compat) + {RESULTS/'pilora_ttest_v2.json'} (full)")

    # ---------------- LaTeX ----------------
    print("\n\\begin{table}[h]\\caption{Paired $t$-tests, Holm-Bonferroni corrected "
          "($m=" + str(len([r for r in forget_rows if r['test']=='paired_t'])) + "$ per family).}")
    print("\\label{tab:ttest}\\centering\\small\\begin{tabular}{lccccc}\\toprule")
    print("Dataset & PILoRA $\\bar F$ & Baseline & Reduction & $p_{\\mathrm{Holm}}$ & $d$ \\\\\\midrule")
    for r in forget_rows:
        if r["test"] == "paired_t":
            print(f"{r['dataset']} & {r['pilora_mean']:.3f} & {r['phase1_method']} ({r['phase1_mean']:.3f}) & "
                  f"{r['reduction_multiple']:.1f}$\\times$ & {r['p_holm']:.4f} {r['significance_holm']} & {r['cohen_d']:+.2f} \\\\")
    print("\\bottomrule\\end{tabular}\\end{table}")


if __name__ == "__main__":
    main()
