"""Section29 paired comparisons from saved stagewise metrics only; no model calls."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_aacil.generate_forgetting import DATA, METHODS, escape, table

SEEDS = [1, 2, 3, 4, 42]
EXTERNAL = ["replay", "derpp", "wacb", "freq", "replay_la"]
CANDIDATES = ["aacil_kd", "aacil_iso"]
COMPARED = CANDIDATES + ["aacil_nogkd"]
AIA = "average_incremental_accuracy"
FORGET = "forgetting_all_old_classes"
ADR = "final_attack_detection"
METRICS = {
    AIA: ("Mean incremental BAcc", "greater"),
    FORGET: ("All-old forgetting", "less"),
    ADR: ("ADR at alpha=2%", "greater"),
    "final_balanced_accuracy": ("Final BAcc", "greater"),
    "final_macro_f1": ("Final MF1", "greater"),
    "forgetting_old_attack_classes": ("Attack-only forgetting", "less"),
}


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def safe(value):
    return value if value is not None and math.isfinite(value) else None


def aggregate(values):
    return {"mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def load_snapshot(folder, datasets):
    summary = json.loads((folder/"result.json").read_text(encoding="utf-8"))
    groups, rows = {}, {}
    for group in summary["groups"]:
        dataset, method = group["dataset"], group["method"]
        if dataset not in datasets or method not in EXTERNAL + COMPARED + ["aacil"]:
            continue
        groups[dataset, method] = group
        for seed in group["seeds"]:
            if seed not in SEEDS:
                raise ValueError("Unexpected seed in the analysis snapshot")
            path = folder/dataset.lower().replace("-", "_")/f"{method}_seed{seed}"/"result.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            if (not row.get("completed") or (row["dataset"], row["method"], row["seed"]) !=
                    (dataset, method, seed) or row["protocol"] != summary["protocol"]):
                raise ValueError("Saved cell does not belong to the selected completed snapshot")
            rows[dataset, method, seed] = row | {"comparison_input": relative(path)}
    return summary, groups, rows


def complete(groups, dataset, method):
    group = groups.get((dataset, method), {})
    return bool(group.get("complete_five_seed") and sorted(group.get("seeds", [])) == SEEDS)


def paired(rows, groups, dataset, candidate, reference, metric):
    seeds = [s for s in SEEDS if (dataset, candidate, s) in rows and (dataset, reference, s) in rows]
    pairs = [{"seed": s, "candidate": rows[dataset, candidate, s]["metrics"].get(metric),
              "reference": rows[dataset, reference, s]["metrics"].get(metric)} for s in seeds]
    for pair in pairs:
        a, b = pair["candidate"], pair["reference"]
        pair["candidate_minus_reference"] = a-b if a is not None and b is not None else None
    result = {"dataset": dataset, "candidate": candidate, "reference": reference, "metric": metric,
              "paired_values": pairs, "n": len(pairs), "required_seeds": SEEDS,
              "difference_direction": "candidate minus reference", "p_greater": None, "p_less": None,
              "p_two_sided": None, "t_statistic": None, "delta_pp": {"mean": None, "sample_sd": None}}
    if not (complete(groups, dataset, candidate) and complete(groups, dataset, reference) and seeds == SEEDS):
        return result | {"status": "pending", "reason": "Both matched groups must contain all five seeds in the saved summary"}
    if any(p["candidate_minus_reference"] is None or not math.isfinite(p["candidate_minus_reference"]) for p in pairs):
        return result | {"status": "unavailable", "reason": "Undefined/nonfinite paired metric; no seed is discarded"}
    delta = [p["candidate_minus_reference"] for p in pairs]
    result.update(candidate= candidate, candidate_summary=aggregate([p["candidate"] for p in pairs]),
                  reference_summary=aggregate([p["reference"] for p in pairs]),
                  delta_pp=aggregate([100*x for x in delta]))
    if statistics.stdev(delta) == 0:
        return result | {"status": "unavailable", "reason": "Zero paired variance; no finite t test and no significance claim"}
    tests = {side: stats.ttest_1samp(delta, 0., alternative=side) for side in ("greater", "less", "two-sided")}
    result.update(status="complete", t_statistic=safe(float(tests["greater"].statistic)),
                  p_greater=safe(float(tests["greater"].pvalue)), p_less=safe(float(tests["less"].pvalue)),
                  p_two_sided=safe(float(tests["two-sided"].pvalue)))
    if any(result[k] is None for k in ("p_greater", "p_less", "p_two_sided")):
        result.update(status="unavailable", reason="Nonfinite paired-test output; no significance claim")
    return result


def strongest(groups, dataset, metric):
    means = {m: groups[dataset, m]["metrics"][metric]["mean"]
             for m in EXTERNAL if complete(groups, dataset, m)}
    result = {"external_set": EXTERNAL, "five_seed_group_means": means,
              "direction": METRICS[metric][1], "tie_rule": "First in the explicitly listed external set"}
    if len(means) != len(EXTERNAL) or any(x is None or not math.isfinite(x) for x in means.values()):
        return result | {"status": "pending", "selected": None,
                         "reason": "Strongest of the exact external five is not available until every group has five defined seeds"}
    winner = min(EXTERNAL, key=means.get) if METRICS[metric][1] == "less" else max(EXTERNAL, key=means.get)
    return result | {"status": "complete", "selected": winner, "selected_mean": means[winner]}


def test_status(comparison, direction, avoid_significant_harm=False):
    if comparison.get("status") != "complete":
        return "pending"
    significant = comparison["p_"+direction] < .05
    return ("fail" if significant else "pass") if avoid_significant_harm else ("pass" if significant else "fail")


def all_required(statuses):
    if "fail" in statuses:
        return "fail"
    return "pass" if statuses and all(s == "pass" for s in statuses) else "pending"


def at_least_two(statuses):
    passed = statuses.count("pass")
    if passed >= 2:
        return "pass"
    return "fail" if passed + statuses.count("pending") < 2 else "pending"


def build(summary, groups, rows, datasets, ablation_scope):
    comparisons, selections, lookup = [], {}, {}
    for dataset in datasets:
        for metric in METRICS:
            selection = strongest(groups, dataset, metric)
            selections[dataset, metric] = selection
            for candidate in COMPARED:
                ref = selection["selected"]
                comparison = (paired(rows, groups, dataset, candidate, ref, metric) if ref else
                    {"dataset": dataset, "candidate": candidate, "reference": None, "metric": metric,
                     "status": "pending", "reason": selection["reason"], "n": 0, "paired_values": []})
                comparison.update(role="external_strongest", selection=selection,
                                  registered_endpoint=metric in (AIA, FORGET, ADR) and candidate in CANDIDATES)
                comparisons.append(comparison); lookup[dataset, candidate, "external", metric] = comparison
        for candidate in COMPARED:
            for metric in (AIA, FORGET, ADR):
                comparison = paired(rows, groups, dataset, candidate, "aacil", metric) | {"role": "original_AA_control"}
                comparisons.append(comparison); lookup[dataset, candidate, "aacil", metric] = comparison
        for candidate, reference, role, metrics in (
                ("aacil_iso", "aacil_kd", "Section29.6_descriptive_ISO_minus_KD", (AIA, FORGET)),
                ("aacil_kd", "aacil_nogkd", "noGKD_ablation_descriptive", (AIA, FORGET, ADR))):
            for metric in metrics:
                comparisons.append(paired(rows, groups, dataset, candidate, reference, metric) | {"role": role})
    verdicts = []
    for candidate in CANDIDATES:
        per_dataset = []
        for dataset in datasets:
            ext = lambda metric: lookup[dataset, candidate, "external", metric]
            aa = lambda metric: lookup[dataset, candidate, "aacil", metric]
            f_less = test_status(aa(FORGET), "less")
            adr_noharm = test_status(aa(ADR), "less", True)
            per_dataset.append({"dataset": dataset,
                "candidate_five_seed_complete": complete(groups, dataset, candidate),
                "incremental_bacc_significantly_greater": test_status(ext(AIA), "greater"),
                "forgetting_not_significantly_greater": test_status(ext(FORGET), "greater", True),
                "adr_significantly_greater": test_status(ext(ADR), "greater"),
                "ablation_forgetting_significantly_less_than_AA": f_less,
                "ablation_adr_not_significantly_less_than_AA": adr_noharm,
                "ablation_combined": all_required([f_less, adr_noharm])})
        incremental = at_least_two([d["incremental_bacc_significantly_greater"] for d in per_dataset])
        forgetting_guard = all_required([d["forgetting_not_significantly_greater"] for d in per_dataset])
        endpoint1 = all_required([incremental, forgetting_guard])
        endpoint2 = at_least_two([d["adr_significantly_greater"] for d in per_dataset])
        ablation_values = [d["ablation_combined"] for d in per_dataset]
        completed_ablation_values = [d["ablation_combined"] for d in per_dataset
                                    if d["candidate_five_seed_complete"] and complete(groups, d["dataset"], "aacil")]
        endpoint3 = all_required(ablation_values) if ablation_scope == "all" else at_least_two(ablation_values)
        complete_all = all(complete(groups, d, m) for d in datasets for m in EXTERNAL+[candidate, "aacil"])
        overall = all_required([endpoint1, endpoint2, endpoint3])
        if overall == "pass" and not complete_all:
            overall = "pending"
        verdicts.append({"candidate": candidate, "dataset_results": per_dataset,
            "criterion_1_class_il": {"status": endpoint1, "at_least_two_incremental_bacc": incremental,
                                     "any_dataset_forgetting_harm_guard": forgetting_guard},
            "criterion_2_adr": {"status": endpoint2},
            "criterion_3_ablation": {"status": endpoint3, "scope": ablation_scope,
                "scope_authority": "direct_user_confirmation_all_completed_formal_datasets" if ablation_scope == "all" else "explicit_CLI_override",
                "completed_dataset_scope_status": all_required(completed_ablation_values),
                "uncompleted_datasets_are_not_counted_as_pass": True},
            "overall": overall, "complete_all_selected_datasets": complete_all,
            "positioning": ("Only endpoint 2 passes: lower-false-alarm/detection positioning without a forgetting advantage"
                            if endpoint2 == "pass" and endpoint1 == endpoint3 == "fail" else
                            "All three criteria pass within the explicitly selected complete dataset scope"
                            if overall == "pass" else
                            "Known necessary condition fails; missing evidence cannot repair that condition"
                            if overall == "fail" else "Incomplete evidence or undefined test; no overall success claim")})
    return {"completed_at": datetime.now().astimezone().isoformat(), "analysis_role": "Section29.4_and_29.6_paired_comparisons",
        "datasets": datasets, "required_seeds": SEEDS, "external_methods": EXTERNAL,
        "formal_candidates": CANDIDATES, "noGKD_role": "ablation_only_not_a_separate_formal_candidate",
        "metric_units": "Source values are fractions; delta_pp is candidate minus reference in percentage points",
        "primary_forgetting_metric": FORGET, "alpha": .02,
        "test": "Two-sided and both one-sided paired t tests on the same five seed differences; registered directions are specified in verdict fields; p<0.05",
        "source_summary_completed_at": summary["completed_at"], "stagewise_protocol": summary["protocol"],
        "comparisons": comparisons, "verdicts": verdicts,
        "inputs": [{"dataset": d, "method": m, "seed": s, "stagewise_result": row["comparison_input"],
                    "training_git_commit": row["training_git_commit"]} for (d,m,s),row in sorted(rows.items())],
        "limitations": ["No model inference, training, checkpoint audit, or recomputation of confusion-matrix metrics is performed here.",
            "No significant degradation is a failure to detect harm, not noninferiority or equivalence.",
            "Tests are unadjusted across metrics/datasets. The strongest external comparator is selected by group mean, not per seed; selection and multiplicity limit inference.",
            "Known failed necessary conditions take precedence over pending evidence. Per-dataset tests require five complete matched seeds; incomplete groups are never dropped to create significance.",
            "Joint, FARQ, Replay-BS, and AA variants are not members of the registered external-five comparison set.",
            "Section27.8's original decision is unchanged; descriptive ISO-KD/noGKD comparisons do not add success requirements."]}


def number(value, digits=4):
    return "--" if value is None else f"{value:.{digits}g}"


def tex(result):
    fragments = []
    for dataset in result["datasets"]:
        rows = []
        for c in result["comparisons"]:
            if c["dataset"] != dataset or c["role"] != "external_strongest" or c["metric"] not in (AIA, FORGET, ADR):
                continue
            delta = c.get("delta_pp", {})
            alternative = "greater"  # forgetting uses the registered higher-is-harm guard
            effect = "--" if delta.get("mean") is None else f"{delta['mean']:.2f}"+rf"$\pm${delta['sample_sd']:.2f}"
            rows.append([escape(METHODS[c["candidate"]]), escape(METRICS[c["metric"]][0]),
                escape(METHODS.get(c.get("reference"), "pending")), str(c["n"]), effect,
                number(c.get("p_"+alternative)), escape(c["status"])])
        fragments.append(table(escape(DATA[dataset])+": Section 29 external-five paired comparisons.",
            "section29-external-"+dataset.lower().replace("-", "_"), "lllrrrr",
            ["Candidate", "Metric", "Strongest reference", "$n$", r"$\Delta$ (pp)", "$p_{>}$", "Evidence"], rows,
            r"Differences are candidate minus reference; mean$\pm$sample SD. For incremental BAcc and ADR, $p_{>}<0.05$ supports improvement. "
            r"For forgetting, $p_{>}<0.05$ indicates harmful higher forgetting and fails its guard; $p\geq0.05$ is not equivalence. "
            r"External selection uses complete five-seed group means and the exact Replay/DER++/WACB/Freq/Replay+LA set. "
            r"AA-CIL-noGKD is an ablation only. Incomplete or undefined tests remain pending/unavailable."))
        control_rows = []
        for c in result["comparisons"]:
            if c["dataset"] != dataset or c["role"] == "external_strongest":
                continue
            delta = c.get("delta_pp", {})
            effect = "--" if delta.get("mean") is None else f"{delta['mean']:.2f}"+rf"$\pm${delta['sample_sd']:.2f}"
            control_rows.append([escape(METHODS[c["candidate"]]+" minus "+METHODS[c["reference"]]),
                escape(METRICS[c["metric"]][0]), str(c["n"]), effect,
                number(c.get("p_less")), number(c.get("p_greater")), number(c.get("p_two_sided"))])
        fragments.append(table(escape(DATA[dataset])+": original-AA controls and descriptive ablations.",
            "section29-controls-"+dataset.lower().replace("-", "_"), "llrrrrr",
            ["Pair", "Metric", "$n$", r"$\Delta$ (pp)", "$p_{<}$", "$p_{>}$", "$p_{2s}$"], control_rows,
            r"Section 29.4(3) requires lower forgetting ($p_{<}<0.05$) and no detected lower ADR ($p_{<}\geq0.05$) versus original AA-CIL. "
            r"The user-confirmed scope is every completed formal dataset; uncompleted datasets do not pass by default. "
            r"ISO--KD and KD--noGKD comparisons are descriptive; both directions and two-sided values are retained without extra success criteria."))
    rows = [[escape(METHODS[v["candidate"]]), v["criterion_1_class_il"]["status"],
             v["criterion_2_adr"]["status"], v["criterion_3_ablation"]["status"], v["overall"],
             "yes" if v["complete_all_selected_datasets"] else "no"] for v in result["verdicts"]]
    fragments.append(table("Section 29.4 decision and completeness are reported separately.", "section29-verdict", "llllll",
        ["Candidate", "Class-IL", "ADR", "Ablation", "Overall", "Complete scope"], rows,
        r"Class-IL requires significantly greater incremental BAcc on at least two datasets and no detected significantly higher forgetting on any selected dataset. "
        r"ADR requires significant improvement on at least two datasets. All three criteria are necessary for the main-method positioning. "
        r"An observed necessary-condition failure is not hidden by unfinished cells. Pending is not pass; nonsignificance is not equivalence. "
        r"The selected dataset scope and each seed's paired values are retained in the generated JSON."))
    return "% Generated from saved stagewise cells and summary only.\n"+"\n".join(fragments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"ncd/results/problem_scout/section28/forgetting_v1")
    parser.add_argument("--datasets", nargs="+", choices=list(DATA), default=["UNSW-NB15", "NSL-KDD", "CIC-IDS-2017"])
    parser.add_argument("--ablation-scope", choices=["all", "at-least-two"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT/"ncd/results/problem_scout/section29/comparisons_v1/result.json")
    parser.add_argument("--tex-output", type=Path, default=ROOT/"paper_aacil/generated/section29_comparisons.tex")
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate datasets are not permitted")
    summary, groups, rows = load_snapshot(args.input.resolve(), args.datasets)
    result = build(summary, groups, rows, args.datasets, args.ablation_scope)
    result["source_summary"] = relative(args.input.resolve()/"result.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    args.tex_output.write_text(tex(result), encoding="utf-8")
    print(json.dumps({"status": "complete", "comparisons": len(result["comparisons"]),
                      "verdicts": {v["candidate"]: v["overall"] for v in result["verdicts"]}}))


if __name__ == "__main__":
    main()
