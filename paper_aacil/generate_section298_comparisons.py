"""Section29.8 frozen-selection assessment from saved stagewise metric cells only."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_aacil.generate_forgetting import DATA, METHODS, escape, table, value
from paper_aacil.generate_section29_comparisons import (
    SEEDS, EXTERNAL, AIA, FORGET, ADR, relative, paired, complete,
    strongest, test_status, all_required, number,
)

VERSIONS = ["aacil_kd", "aacil_iso", "aacil_wa", "aacil_cbf", "aacil_ncm"]
REQUIRED_DATASETS = ["UNSW-NB15", "NSL-KDD"]
NAMES = METHODS | {"aacil_kd": "AA-CIL-KD", "aacil_iso": "AA-CIL-ISO",
                  "aacil_wa": "AA-CIL-WA", "aacil_cbf": "AA-CIL-CBF", "aacil_ncm": "AA-CIL-NCM"}
FINAL_BACC = "final_balanced_accuracy"
DESCRIPTIVE = [AIA, FINAL_BACC, FORGET, "forgetting_old_attack_classes", ADR,
               "final_macro_f1", "final_normal_fpr"]
CONFIG_KEYS = {"lambda_g": "lambda_g", "der_alpha": "derpp_alpha", "der_beta": "derpp_beta",
               "cbf_epochs": "cbf_epochs", "cbf_lr_multiplier": "cbf_lr_multiplier"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_selection(path):
    selection = read(path)
    if (not selection.get("completed") or not selection.get("selection_complete") or
            selection.get("section") != "29.8" or selection.get("selection_seeds") != [5, 6] or
            selection.get("selection_dataset") != "UNSW-NB15" or
            selection.get("test_metrics_used_for_selection") is not False):
        raise ValueError("A complete immutable Section29.8 validation-only selection is required")
    versions = {v["method"]: v for v in selection["per_version"]}
    if set(versions) != set(VERSIONS):
        raise ValueError("Selection must retain all five formal-report configurations")
    main = selection["selected_main"]
    if main is not None and (main["method"] not in versions or not main.get("eligible") or
            main["selected_candidate_id"] != versions[main["method"]]["selected_candidate_id"]):
        raise ValueError("Selected main does not match the frozen eligible per-version configuration")
    return selection, versions


def formal_configuration(row, selection, chosen):
    """Necessary selection binding from small JSON, not a checkpoint/source audit."""
    configuration = row.get("training_configuration", {})
    if not configuration.get("section298") or any(k not in configuration for k in CONFIG_KEYS.values()):
        source = read(ROOT/row["source_result"])
        configuration = source["identity"]["effective_config"]
    if not configuration.get("section298") or configuration.get("selection_only") or configuration.get("smoke"):
        raise ValueError("A historical/development/smoke cell cannot stand in for a selected Section29.8 formal version")
    if row["training_git_commit"] != selection["source_commit"]:
        raise ValueError("Formal source differs from the source fixed in selection.json")
    for selected_key, config_key in CONFIG_KEYS.items():
        if configuration.get(config_key) != chosen["configuration"][selected_key]:
            raise ValueError("Formal hyperparameters differ from the immutable per-version selection")
    return {k: configuration[k] for k in CONFIG_KEYS.values()}


def load_snapshot(folder, selection, versions, include_cic):
    summary = read(folder/"result.json")
    groups, rows, bindings = {}, {}, []
    methods = set(EXTERNAL+VERSIONS+["aacil", "replay_bs"])
    datasets = REQUIRED_DATASETS+(["CIC-IDS-2017"] if include_cic else [])
    for group in summary["groups"]:
        dataset, method = group["dataset"], group["method"]
        if dataset not in datasets or method not in methods:
            continue
        groups[dataset, method] = group
        for seed in group["seeds"]:
            if seed not in SEEDS:
                raise ValueError("Development seeds cannot enter the formal report")
            path = folder/dataset.lower().replace("-", "_")/f"{method}_seed{seed}"/"result.json"
            row = read(path)
            if (not row.get("completed") or (row["dataset"], row["method"], row["seed"]) !=
                    (dataset, method, seed) or row["protocol"] != summary["protocol"]):
                raise ValueError("Stagewise cell does not belong to the saved complete summary")
            if method in VERSIONS:
                config = formal_configuration(row, selection, versions[method])
                bindings.append({"dataset": dataset, "method": method, "seed": seed,
                    "selected_candidate_id": versions[method]["selected_candidate_id"],
                    "configuration": config, "stagewise_result": relative(path),
                    "training_git_commit": row["training_git_commit"]})
            rows[dataset, method, seed] = row | {"comparison_input": relative(path)}
    return summary, groups, rows, bindings, datasets


def assess(summary, groups, rows, selection, versions, bindings, datasets):
    selected = selection["selected_main"]
    method = selected["method"] if selected else None
    dataset_results, descriptive, iso_kd = [], [], []
    for dataset in datasets:
        for version in VERSIONS:
            group = groups.get((dataset, version), {})
            descriptive.append({"dataset": dataset, "method": version, "n": group.get("n", 0),
                "seeds": group.get("seeds", []), "complete_five_seed": complete(groups, dataset, version),
                "selected_main": version == method, "development_eligible": versions[version]["eligible"],
                "selection_reason": versions[version]["selection_reason"],
                "selected_candidate_id": versions[version]["selected_candidate_id"],
                "metrics": {k: group.get("metrics", {}).get(k) for k in DESCRIPTIVE}})
        for metric in (FORGET, AIA):
            iso_kd.append(paired(rows, groups, dataset, "aacil_iso", "aacil_kd", metric) |
                {"role": "descriptive_ISO_minus_KD_not_a_selection_rule_or_success_requirement"})
        if method is None:
            dataset_results.append({"dataset": dataset, "required": dataset in REQUIRED_DATASETS,
                "status": "not_applicable_no_eligible_main", "complete": False, "tests": []})
            continue
        lowest = strongest(groups, dataset, FORGET)
        forgetting = (paired(rows, groups, dataset, method, lowest["selected"], FORGET)
            if lowest["selected"] else {"dataset": dataset, "candidate": method, "reference": None,
                "metric": FORGET, "status": "pending", "n": 0, "paired_values": [], "reason": lowest["reason"]})
        forgetting.update(criterion=1, comparator_selection=lowest, p_direction="greater",
            rule="Fail if candidate forgetting is significantly higher: one-sided p_greater < .05",
            decision=test_status(forgetting, "greater", True))
        final_bacc = paired(rows, groups, dataset, method, "replay_bs", FINAL_BACC)
        final_bacc.update(criterion=2, p_direction="greater",
            rule="Pass only if final BAcc is significantly higher than Replay-BS: one-sided p_greater < .05",
            decision=test_status(final_bacc, "greater"))
        detection = paired(rows, groups, dataset, method, "aacil", ADR)
        detection.update(criterion=3, p_direction="less",
            rule="Fail if alpha=.02 ADR is significantly lower than original AA: one-sided p_less < .05",
            decision=test_status(detection, "less", True))
        tests = [forgetting, final_bacc, detection]
        dataset_results.append({"dataset": dataset, "required": dataset in REQUIRED_DATASETS,
            "status": all_required([t["decision"] for t in tests]), "tests": tests,
            "complete": all(complete(groups, dataset, m) for m in EXTERNAL+[method,"replay_bs","aacil"])})
    required = [d for d in dataset_results if d["required"]]
    overall = "fail" if method is None else all_required([d["status"] for d in required])
    all_versions_complete = all(complete(groups, d, v) for d in REQUIRED_DATASETS for v in VERSIONS)
    return {"completed_at": datetime.now().astimezone().isoformat(), "section": "29.8",
        "assessment_role": "Frozen validation-selected main, not a new test-based version choice",
        "required_datasets": REQUIRED_DATASETS, "optional_CIC17_role": "Reported separately when requested; does not replace either required dataset",
        "formal_seeds": SEEDS, "primary_forgetting": FORGET, "external_five": EXTERNAL,
        "selected_main": selected, "main_selection_status": selection["main_selection_status"],
        "selection_completed_at": selection["completed_at"], "selection_source_commit": selection["source_commit"],
        "selection_test_metrics_used": False, "source_summary_completed_at": summary["completed_at"],
        "stagewise_protocol": summary["protocol"], "dataset_results": dataset_results,
        "overall_required_dataset_verdict": overall,
        "required_selected_comparisons_complete": method is not None and all(d["complete"] for d in required),
        "all_five_versions_required_dataset_report_complete": all_versions_complete,
        "all_versions_descriptive": descriptive, "ISO_minus_KD_descriptive": iso_kd,
        "formal_selection_bindings": bindings,
        "positioning": ("Lower false alarms and forgetting remain unresolved; AA-CIL is not claimed as the method contribution"
                        if overall == "fail" else "All Section29.8 necessary tests pass on both required datasets"
                        if overall == "pass" else "Required formal evidence remains incomplete; no success claim"),
        "test_definition": "Unadjusted paired t tests on the five same-seed differences; candidate minus reference; p<.05. Lowest-forgetting external comparator selected by whole five-seed mean, with listed external order breaking ties",
        "units": "Metric values are fractions; delta_pp values are percentage points",
        "limitations": ["Only saved stagewise metrics are consumed. No inference, training, checkpoint audit, or CM-metric recomputation is performed.",
            "A failure to detect significant forgetting/ADR harm is not noninferiority or equivalence.",
            "Known necessary-condition failure takes precedence over missing evidence. Incomplete five-seed groups never yield a test or lose a seed to obtain significance.",
            "All five configurations, including ineligible fallbacks, remain in the descriptive report; formal test results do not change the frozen main selection.",
            "External comparator selection and multiple unadjusted tests limit inferential claims.",
            "The original Section27.8 and Section29.4 decisions are unchanged; this assessment is separately scoped to Section29.8."]}


def effect(c):
    delta = c.get("delta_pp", {})
    if delta.get("mean") is None:
        return "--"
    return f"{delta['mean']:.2f}"+rf"$\pm${delta['sample_sd']:.2f}"


def render(result):
    pieces = []
    datasets = list(dict.fromkeys(r["dataset"] for r in result["all_versions_descriptive"]))
    headers = ["Version", "$n$", "Mean inc. BAcc", "Final BAcc", "Forget (all)", "Forget (att.)", "ADR", "MF1", "FPR"]
    for dataset in datasets:
        rows = []
        for group in result["all_versions_descriptive"]:
            if group["dataset"] != dataset:
                continue
            marker = r"$^\star$" if group["selected_main"] else ""
            marker += r"$^\dagger$" if not group["development_eligible"] else ""
            rows.append([escape(NAMES[group["method"]])+marker, str(group["n"])] +
                [value(group["metrics"][k]) if group["metrics"][k] else "--" for k in DESCRIPTIVE])
        slug = dataset.lower().replace("-", "_")
        pieces.append(table(escape(DATA[dataset])+r": all five Section 29.8 versions at $\alpha=2\%$.",
            "section298-all-"+slug, "lrrrrrrrr", headers, rows,
            r"Mean$\pm$sample SD; accuracy/rates are percentages, signed forgetting is in percentage points. "
            r"$\star$ marks the main version selected only on UNSW training validation seeds 5/6; "
            r"$\dagger$ marks an ineligible development fallback retained for mandatory formal reporting, not an eligible success. "
            r"Incomplete groups ($n<5$) are descriptive only. Low forgetting is interpreted together with incremental/final BAcc."))
    decision_rows = []
    for dataset in result["dataset_results"]:
        for test in dataset["tests"]:
            decision_rows.append([escape(DATA[dataset["dataset"]]), str(test["criterion"]),
                escape(NAMES.get(test.get("reference"), "pending")), str(test["n"]), effect(test),
                number(test.get("p_"+test["p_direction"])), test["decision"]])
    if decision_rows:
        pieces.append(table("Section 29.8 necessary tests for the frozen selected main version.", "section298-required", "llrrrrl",
            ["Data", "Criterion", "Reference", "$n$", r"$\Delta$ (pp)", "One-sided $p$", "Decision"], decision_rows,
            r"1: all-old forgetting must not be significantly higher than the lowest-forgetting member of the exact external five ($p_{>}<0.05$ means failure). "
            r"2: final BAcc must exceed Replay-BS ($p_{>}<0.05$ required). "
            r"3: ADR must not be significantly lower than original AA-CIL ($p_{<}<0.05$ means failure). "
            r"UNSW and NSL must both satisfy all three; optional CIC17 is separate. No detected harm is not equivalence. "
            "Overall required-dataset verdict: "+escape(result["overall_required_dataset_verdict"])+". "))
    else:
        pieces.append(r"\par\noindent\textit{Section 29.8: no eligible version was selected on development validation; "
                      r"formal fallback results do not convert this into success.}\par"+"\n")
    rows = [[escape(DATA[c["dataset"]]), "Forget (all)" if c["metric"] == FORGET else "Mean inc. BAcc",
             str(c["n"]), effect(c), number(c.get("p_less")), number(c.get("p_greater")),
             number(c.get("p_two_sided"))] for c in result["ISO_minus_KD_descriptive"]]
    pieces.append(table("AA-CIL-ISO minus AA-CIL-KD: descriptive paired comparison.", "section298-iso-kd", "llrrrrr",
        ["Data", "Metric", "$n$", r"$\Delta$ (pp)", "$p_{<}$", "$p_{>}$", "$p_{2s}$"], rows,
        r"These formal-seed comparisons do not select the main method or add success criteria. "
        r"Five complete matched seeds are required; missing or undefined tests remain unavailable. "
        r"All p values are unadjusted. Prior Section 29.4 and Section 27.8 decisions are not amended."))
    return "% Generated from immutable Section29.8 selection and saved stagewise metrics only.\n"+"\n".join(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT/"ncd/results/problem_scout/section29/retention_queue_v1/selection.json")
    parser.add_argument("--input", type=Path, default=ROOT/"ncd/results/problem_scout/section28/forgetting_v1")
    parser.add_argument("--include-cic", action="store_true", help="Add optional CIC17 comparisons without replacing UNSW or NSL")
    parser.add_argument("--output", type=Path, default=ROOT/"ncd/results/problem_scout/section29/retention_comparisons_v1/result.json")
    parser.add_argument("--tex-output", type=Path, default=ROOT/"paper_aacil/generated/section298_comparisons.tex")
    args = parser.parse_args()
    selection, versions = load_selection(args.selection.resolve())
    summary, groups, rows, bindings, datasets = load_snapshot(args.input.resolve(), selection, versions, args.include_cic)
    result = assess(summary, groups, rows, selection, versions, bindings, datasets)
    result.update(selection_result=relative(args.selection), source_summary=relative(args.input/"result.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    args.tex_output.write_text(render(result), encoding="utf-8")
    print(json.dumps({"status": "complete", "selected_main": result["selected_main"],
        "verdict": result["overall_required_dataset_verdict"],
        "all_five_versions_report_complete": result["all_five_versions_required_dataset_report_complete"]}))


if __name__ == "__main__":
    main()
