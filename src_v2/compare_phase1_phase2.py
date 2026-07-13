"""Compare PILoRA-IDS (Phase II) results to Phase I baselines.

Reads:
  results/NSL-KDD_results.json              — 8 Phase I methods
  results/fedmac_single_site_nsl_kdd.json   — PILoRA v0.1

Prints a side-by-side table.
"""
import json
from pathlib import Path

RESULTS = Path("results")


def main():
    print("=" * 70)
    print(" Phase I vs Phase II Comparison (NSL-KDD)")
    print("=" * 70)

    # ---- Phase I
    phase1 = json.load(open(RESULTS / "NSL-KDD_results.json"))
    p1 = phase1.get("results", {})

    # ---- Phase II (single-site)
    fm_path = RESULTS / "fedmac_single_site_nsl_kdd.json"
    if not fm_path.exists():
        print(f"\nNo Phase II result yet at {fm_path}")
        return
    p2 = json.load(open(fm_path))

    # ---- side by side
    rows = []
    for method, m in p1.items():
        rows.append({
            "method": method,
            "phase": "I",
            "acc": m.get("avg_accuracy"),
            "forget": m.get("avg_forgetting"),
            "time_s": m.get("time_sec"),
            "params": "MLP (~17K)",
        })
    rows.append({
        "method": "PILoRA-IDS ★",
        "phase": "II",
        "acc": p2.get("avg_accuracy"),
        "forget": p2.get("avg_forgetting"),
        "time_s": p2.get("wall_time_sec"),
        "params": f"Transformer ({p2.get('lora_params_per_family', 0)}/family + 102K frozen)",
    })

    print(f"\n{'Method':<25}{'Phase':<7}{'AvgAcc':<10}{'AvgForget':<12}{'Time(s)':<10}{'Params':<30}")
    print("-" * 100)
    for r in rows:
        acc_s = f"{r['acc']:.4f}" if r['acc'] is not None else "—"
        fgt_s = f"{r['forget']:.4f}" if r['forget'] is not None else "—"
        t_s = f"{r['time_s']:.1f}" if r['time_s'] is not None else "—"
        print(f"{r['method']:<25}{r['phase']:<7}{acc_s:<10}{fgt_s:<12}{t_s:<10}{r['params']:<30}")

    # ---- Phase II diagnostics
    print(f"\nPhase II details:")
    print(f"  Families spawned:  {p2.get('n_families')}")
    print(f"  Centroids total:   {p2.get('n_centroids_total')}")
    print(f"  Comm payload:      {p2.get('centroid_bytes', 0) / 1024:.1f} KB")
    print(f"  LoRA params/family:{p2.get('lora_params_per_family')}")
    print(f"  LoRA total:        {p2.get('lora_params_total')}")


if __name__ == "__main__":
    main()
