"""Compact-family reporting from saved complete stagewise cells only.

No model, inference, training, or selection search is invoked. The immutable
UNSW/NSL validation selection fixes every candidate configuration and the main
method. Development-dataset formal rows remain descriptive. CSE18 is opt-in:
all 35 physical reference completion files and their aggregate release must
exist before any CSE18 result or reference-summary JSON is read.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_aacil.generate_forgetting import escape, table, value
from paper_aacil.generate_section29_comparisons import (
    SEEDS, EXTERNAL, AIA, FORGET, ADR, paired, complete, strongest,
    test_status, all_required, number,
)

VERSIONS = ["aacil_kd_weak", "aacil_kd_wa", "aacil_kd_cbf", "aacil_ema_kd", "aacil_dual"]
DEVELOPMENT_DATASETS = ["UNSW-NB15", "NSL-KDD"]
CIC = "CIC-IDS-2017"
CSE = "CSE-CIC-IDS2018"
PHYSICAL_REFERENCES = ["replay", "derpp", "wacb", "freq", "replay_bs", "aacil", "aacil_kd"]
NAMES = {"aacil_kd_weak": "KD-weak", "aacil_kd_wa": "KD+WA", "aacil_kd_cbf": "KD+CBF",
         "aacil_ema_kd": "EMA-KD", "aacil_dual": "DUAL", "aacil_kd": "AA-CIL-KD",
         "aacil": "AA-CIL", "replay_bs": "Replay-BS", "replay": "Replay", "derpp": "DER++",
         "wacb": "WACB", "freq": "Freq", "replay_la": "Replay+LA"}
DATA = {"UNSW-NB15": "UNSW", "NSL-KDD": "NSL", CIC: "CIC17", CSE: "CSE18"}
BACC = "final_balanced_accuracy"
METRICS = [AIA, BACC, FORGET, ADR, "final_macro_f1", "final_normal_fpr"]
CONFIG_KEYS = {"lambda_g": "lambda_g", "der_alpha": "derpp_alpha", "der_beta": "derpp_beta",
               "cbf_epochs": "cbf_epochs", "cbf_lr_multiplier": "cbf_lr_multiplier",
               "ema_decay": "ema_decay", "ema_consistency": "ema_consistency"}
CAPACITY_KEYS = ["training_parameters", "optimizer_parameters", "inference_parameters",
                 "reference_full_attack_head_parameters", "extra_components"]
STORAGE_KEYS = ["replay_memory_payload_bytes", "replay_sample_payload_bytes", "replay_logit_target_bytes",
                "replay_memory_storage", "normal_gate_training_storage"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    # Published evidence links never contain a private absolute path.
    return Path(path).resolve().relative_to(ROOT).as_posix()


def source_path(label):
    path = (ROOT / label).resolve()
    path.relative_to(ROOT)
    return path


def stamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Source completion requires a timezone-aware timestamp")
    return parsed


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values):
    if not values or any(not numeric(v) for v in values):
        return {"n": len(values), "mean": None, "sample_sd": None}
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def load_selection(path):
    selection = read(path)
    if (selection.get("completed") is not True or selection.get("selection_complete") is not True
            or selection.get("selection_datasets") != DEVELOPMENT_DATASETS
            or selection.get("selection_seeds") != [5, 6]
            or selection.get("formal_seeds") != SEEDS
            or selection.get("test_metrics_used_for_selection") is not False):
        raise ValueError("Complete immutable UNSW/NSL seeds5/6 validation-only selection is required")
    versions = {v["method"]: v for v in selection["per_version"]}
    selected = selection["selected_main"]
    if set(versions) != set(VERSIONS) or selected.get("method") not in versions:
        raise ValueError("Selection must retain all five compact candidates and a fixed main")
    chosen = versions[selected["method"]]
    if any(selected.get(k) != chosen.get(k) for k in ("selected_candidate_id", "configuration", "eligible")):
        raise ValueError("Main method differs from its immutable per-version choice")
    if any(set(v["configuration"]) != set(CONFIG_KEYS) for v in versions.values()):
        raise ValueError("Selection configuration keys differ from the registered bounded grid")
    stamp(selection["completed_at"])
    return selection, versions


def capacity_and_resources(source):
    capacity = source.get("architecture_capacity")
    if not isinstance(capacity, dict) or any(k not in capacity for k in CAPACITY_KEYS):
        raise ValueError("Compact source lacks mandatory model/component accounting")
    stages = source.get("checkpoints", [])
    if not stages or len(stages) != len(source["tasks"]):
        raise ValueError("Completed compact source has an incomplete trajectory")
    metadata = [stage.get("family_metadata", {}) for stage in stages]
    if any(any(row.get(k) != capacity[k] for k in CAPACITY_KEYS) for row in metadata):
        raise ValueError("Compact F parameter or component count changed across stages")
    if (capacity["extra_components"] not in (0, 1)
            or capacity["training_parameters"] > 2 * capacity["reference_full_attack_head_parameters"]
            or capacity["inference_parameters"] > 2 * capacity["reference_full_attack_head_parameters"]):
        raise ValueError("Compact F exceeds its fixed parameter/component constraint")
    return {
        "architecture_capacity": {k: capacity[k] for k in CAPACITY_KEYS},
        "constant_across_tasks": True,
        "cost_seconds": {k: v for k, v in source.items() if k.endswith("_seconds_total") and numeric(v)},
        "cost_scope": source.get("cost_scope"),
        "stage_storage": [{"task_index": stage["task_index"],
            "memory": {k: stage.get("allocation", {}).get(k) for k in STORAGE_KEYS},
            "permanent_normal_calibration": stage.get("calibration_pool_storage"),
            "selection_attack_holdout": stage.get("selection_attack_holdout_storage"),
            "ephemeral_normal_training_feature_and_id_bytes": stage.get("ephemeral_normal_training_feature_and_id_bytes")}
            for stage in stages],
    }


def load_compact_group(folder, dataset, method, selection, version):
    paths = [folder / dataset.lower().replace("-", "_") / f"{method}_seed{seed}" / "result.json" for seed in SEEDS]
    present = [seed for seed, path in zip(SEEDS, paths) if path.is_file()]
    if present != SEEDS:
        # Do not read partial cell metrics, including a partial CSE main group.
        return None, {}, {"dataset": dataset, "method": method, "available_seed_files": present,
                          "missing_seeds": [s for s in SEEDS if s not in present], "metrics_read": False}, []
    rows, bindings, resources = {}, [], []
    protocol = None
    for seed, path in zip(SEEDS, paths):
        row = read(path)
        if (row.get("completed") is not True
                or (row.get("dataset"), row.get("method"), row.get("seed")) != (dataset, method, seed)
                or row.get("protocol", {}).get("alpha") != .02):
            raise ValueError("Invalid or incomplete stagewise compact cell")
        if protocol is not None and row["protocol"] != protocol:
            raise ValueError("Stagewise definitions differ within a candidate group")
        protocol = row["protocol"]
        raw_path = source_path(row["source_result"])
        source = read(raw_path)
        identity, config = source["identity"], source["identity"]["effective_config"]
        if (source.get("completed") is not True or source.get("test_evaluation_performed") is not True
                or source.get("selection_only") or config.get("smoke") or config.get("selection_only")
                or config.get("section31") is not True
                or tuple(identity.get(k) for k in ("dataset", "method", "seed")) != (dataset, method, seed)):
            raise ValueError("Development, smoke, or incompatible source cannot supply compact formal metrics")
        if identity.get("git_commit") != selection["source_commit"] or row["training_git_commit"] != selection["source_commit"]:
            raise ValueError("Compact formal training differs from the immutable selection source")
        for key, config_key in CONFIG_KEYS.items():
            if config.get(config_key) != version["configuration"][key]:
                raise ValueError("Compact formal configuration differs from validation selection")
        if stamp(source["completed_at"]) < stamp(selection["completed_at"]):
            raise ValueError("A formal compact result predates the frozen complete validation selection")
        if row["class_order"] != source["class_order"] or row["tasks"] != source["tasks"]:
            raise ValueError("Stagewise class/task definition differs from its trained source")
        record = capacity_and_resources(source)
        resources.append({"seed": seed, **record})
        binding = {"dataset": dataset, "method": method, "seed": seed,
            "stagewise_result": relative(path), "stagewise_sha256": digest(path),
            "source_result": relative(raw_path), "source_sha256": digest(raw_path),
            "source_completed_at": source["completed_at"], "training_git_commit": identity["git_commit"],
            "selected_candidate_id": version["selected_candidate_id"], "configuration": version["configuration"]}
        bindings.append(binding)
        rows[dataset, method, seed] = row | {"comparison_input": relative(path)}
    group = {"dataset": dataset, "method": method, "n": 5, "seeds": SEEDS, "complete_five_seed": True,
        "metrics": {k: stats([rows[dataset, method, s]["metrics"].get(k) for s in SEEDS]) for k in METRICS},
        "resources_by_seed": resources, "stagewise_protocol": protocol,
        "role": "development_dataset_descriptive" if dataset in DEVELOPMENT_DATASETS else "fixed_main_external_dataset"}
    return group, rows, None, bindings


def cse_release_guard(root):
    """Existence/completion release gate before reading any CSE metric JSON."""
    paths = [root / "cells" / method / f"seed_{seed}" / "section31_result.json"
             for method in PHYSICAL_REFERENCES for seed in SEEDS]
    if not all(path.is_file() for path in paths):
        raise ValueError("CSE18 remains sealed: all35 physical reference completion files are required before metrics are read")
    summary_path = root / "reference_summary.json"
    if not summary_path.is_file():
        raise ValueError("CSE18 remains sealed: the guarded all35 aggregate release does not exist")
    # reference_summary is emitted only by the all35 release routine. No earlier
    # call in this script reads CSE model data, stagewise data, or this summary.
    release = read(summary_path)
    if (release.get("completed") is not True or release.get("smoke") is not False
            or release.get("physical_cells") != 35 or release.get("test_selected") is not False
            or release.get("primary_selected_candidate_included") is not False):
        raise ValueError("CSE reference release is incomplete, smoke, or test-selected")
    expected_bindings = [{"path": path.relative_to(root).as_posix(), "sha256": digest(path)} for path in paths]
    if release.get("source_results") != expected_bindings:
        raise ValueError("CSE all35 completion files differ from the guarded aggregate release")
    return paths, release, {"all35_reference_release_satisfied": True, "physical_cells": 35,
        "reference_summary": relative(summary_path), "reference_summary_sha256": digest(summary_path),
        "released_at": release["released_at"], "source_results": expected_bindings}


def load_cse_references(root, paths, release):
    groups, rows, bindings = {}, {}, []
    expected_groups = set(PHYSICAL_REFERENCES + ["replay_la"])
    released = {group["method"]: group for group in release["groups"]}
    if set(released) != expected_groups or any(group.get("seeds") != SEEDS for group in released.values()):
        raise ValueError("CSE release must include all five seeds of each registered logical reference")
    for path in paths:
        source = read(path)
        method, seed = source.get("method"), source.get("seed")
        if (source.get("completed") is not True or source.get("smoke") is not False
                or source.get("dataset") != CSE or method not in PHYSICAL_REFERENCES or seed not in SEEDS
                or path != root / "cells" / method / f"seed_{seed}" / "section31_result.json"):
            raise ValueError("CSE released cell is not an assigned completed formal reference")
        rows[CSE, method, seed] = {"metrics": source["metrics"], "comparison_input": relative(path),
            "class_order": source["identity"]["class_order"], "tasks": source["identity"]["tasks"],
            "training_git_commit": source["identity"]["git_commit"],
            "cost_seconds": source.get("cost_seconds"), "adapter_cost_seconds": source.get("adapter_cost_seconds")}
        bindings.append({"dataset": CSE, "method": method, "seed": seed, "source_result": relative(path),
                         "source_sha256": digest(path), "source_completed_at": source["completed_at"]})
    for seed in SEEDS:
        rows[CSE, "replay_la", seed] = rows[CSE, "replay", seed] | {
            "derived_from": "replay", "training_cost_double_charged": False}
    for method, released_group in released.items():
        calculated = {k: stats([rows[CSE, method, s]["metrics"].get(k) for s in SEEDS]) for k in METRICS}
        for metric in METRICS:
            for field in ("mean", "sample_sd"):
                expected, observed = calculated[metric][field], released_group["metrics"][metric].get(field)
                if expected != observed and not (numeric(expected) and numeric(observed) and math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-14)):
                    raise ValueError("Released group does not match its saved per-seed metric values")
        groups[CSE, method] = {"dataset": CSE, "method": method, "n": 5, "seeds": SEEDS,
            "complete_five_seed": True, "metrics": calculated, "role": "external_reference",
            "resources_by_seed": [{"seed": s, "cost_seconds": rows[CSE, method, s].get("cost_seconds"),
                "adapter_cost_seconds": rows[CSE, method, s].get("adapter_cost_seconds")}
                for s in SEEDS] if method != "replay_la" else [],
            "training_cost_double_charged": False if method == "replay_la" else None}
    return groups, rows, bindings


def cse_tests(groups, rows, selected_method):
    lowest = strongest(groups, CSE, FORGET)
    first = paired(rows, groups, CSE, selected_method, lowest["selected"], FORGET)
    first.update(criterion=1, comparator_selection=lowest, p_direction="greater",
        rule="Fail only if forgetting is significantly higher than the lowest-mean external-five member: p_greater < .05",
        decision=test_status(first, "greater", True))
    second = paired(rows, groups, CSE, selected_method, "replay_bs", BACC)
    second.update(criterion=2, p_direction="greater",
        rule="Pass only if final BAcc is significantly higher than Replay-BS: p_greater < .05",
        decision=test_status(second, "greater"))
    third = paired(rows, groups, CSE, selected_method, "aacil", BACC)
    mean = third["delta_pp"]["mean"]
    full = complete(groups, CSE, selected_method) and complete(groups, CSE, "aacil") and third["n"] == 5
    third.update(criterion=3, p_direction=None, margin_pp=-2.,
        rule="Pass if mean final-BAcc difference from original AA-CIL is at least -2pp; no significance/noninferiority test is required",
        decision=("pass" if mean >= -2. else "fail") if full and numeric(mean) else "pending")
    fourth = paired(rows, groups, CSE, selected_method, "aacil", ADR)
    fourth.update(criterion=4, p_direction="less",
        rule="Fail if alpha=.02 ADR is significantly lower than original AA-CIL: p_less < .05",
        decision=test_status(fourth, "less", True))
    tests = [first, second, third, fourth]
    for test in tests:
        differences = [p["candidate_minus_reference"] for p in test["paired_values"]]
        if len(differences) == 5 and all(numeric(v) for v in differences):
            observed = statistics.mean(differences)
            null = [statistics.mean(s * d for s, d in zip(signs, differences))
                    for signs in itertools.product((-1, 1), repeat=5)]
            tolerance = 1e-15
            test["exact_sign_flip"] = {
                "statistic": "mean paired difference; all 32 sign assignments, inclusive tails",
                "assignments": 32, "observed_mean": observed,
                "p_greater": sum(v >= observed - tolerance for v in null) / 32,
                "p_less": sum(v <= observed + tolerance for v in null) / 32,
                "p_two_sided": sum(abs(v) >= abs(observed) - tolerance for v in null) / 32,
                "role": "Sensitivity check; does not replace the registered t-test or criterion3 mean rule",
            }
    return tests


def summarize(selection, versions, groups, rows, missing, bindings, selection_path, cse_guard):
    selected = selection["selected_main"]
    descriptive = []
    for dataset in DEVELOPMENT_DATASETS + ([CSE] if cse_guard else []):
        methods = VERSIONS if dataset in DEVELOPMENT_DATASETS else [selected["method"]] + PHYSICAL_REFERENCES + ["replay_la"]
        for method in methods:
            if (dataset, method) not in groups:
                continue
            group = groups[dataset, method]
            descriptive.append(group | {"selected_main": method == selected["method"],
                "development_eligible": versions[method]["eligible"] if method in versions else None,
                "selected_candidate_id": versions[method]["selected_candidate_id"] if method in versions else None})
    tests = cse_tests(groups, rows, selected["method"]) if cse_guard else []
    if cse_guard and complete(groups, CSE, selected["method"]):
        candidate = rows[CSE, selected["method"], SEEDS[0]]
        for method in PHYSICAL_REFERENCES:
            reference = rows[CSE, method, SEEDS[0]]
            if candidate["class_order"] != reference["class_order"] or candidate["tasks"] != reference["tasks"]:
                raise ValueError("CSE main and reference class/task definitions differ")
    status = all_required([t["decision"] for t in tests])
    return {"schema_version": 1, "completed_at": datetime.now().astimezone().isoformat(), "section": "31",
        "selection_source": relative(selection_path), "selection_sha256": digest(selection_path),
        "selection_completed_at": selection["completed_at"], "selection_source_commit": selection["source_commit"],
        "selection_datasets": DEVELOPMENT_DATASETS, "selection_seeds": [5, 6], "selection_test_metrics_used": False,
        "selected_main": {k: selected[k] for k in ("method", "selected_candidate_id", "configuration", "eligible", "selection_reason")},
        "main_selection_status": selection["main_selection_status"],
        "selection_summary": [{k: version[k] for k in ("method", "selected_candidate_id", "configuration", "eligible",
            "average_validation_final_bacc", "average_validation_forgetting", "forgetting_constraint_by_dataset", "selection_reason")}
            for version in selection["per_version"]],
        "formal_seeds": SEEDS, "alpha": .02, "external_five": EXTERNAL,
        "descriptive_groups": descriptive, "incomplete_groups": missing,
        "completed_group_keys": sorted([dataset + "/" + method for dataset, method in groups]),
        "cse_release_guard": cse_guard, "CSE18_necessary_tests": tests, "CSE18_necessary_verdict": status,
        "CSE18_all_required_evidence_complete": bool(cse_guard and all(complete(groups, CSE, method)
            for method in EXTERNAL + [selected["method"], "aacil", "aacil_kd", "replay_bs"])),
        "all_five_versions_development_dataset_report_complete": all(complete(groups, dataset, method)
            for dataset in DEVELOPMENT_DATASETS for method in VERSIONS),
        "source_bindings": bindings,
        "positioning": "All four fixed external-dataset numerical requirements are met within this protocol" if status == "pass" else
            "A necessary external-dataset condition fails; the false-alarm/forgetting/accuracy trade-off remains unresolved" if status == "fail" else
            "External-dataset evidence remains incomplete; no resolved-trade-off claim",
        "units": "Saved metric values are fractions; delta_pp is candidate minus reference in percentage points",
        "limitations": ["UNSW and NSL are development datasets, including their five-seed formal reports; neither is blind confirmation.",
            "The main method and each configuration remain fixed by validation-only selection; no formal metric can reselect a candidate.",
            "If the validation forgetting constraint was unsatisfied, the designated fallback remains explicitly ineligible even when externally assessed.",
            "Only complete five-seed groups are summarized; no partial-group significance, seed dropping or imputation is allowed.",
            "CSE metrics are consumed only after all35 physical reference completion files and their bound release exist; Replay+LA is a calibrated Replay alias without extra training cost.",
            "Paired one-sided tests are unadjusted. Not detecting harm is not equivalence or statistical noninferiority.",
            "The -2pp final-BAcc rule is a fixed mean-difference rule, not a significance or noninferiority test.",
            "Signed all-old-class forgetting includes Normal and is not clipped; low forgetting alone is not successful learning.",
            "Model and sample storage are distinct. Nested gate/family/CBF training cost fields must not be added twice.",
            "This script reads saved scalar metrics only, without model calls or confusion-matrix recomputation. Historical failed decisions and unresolved RQ1/RQ3/innovation remain unchanged."]}


def render(result):
    fragments = []
    for dataset in DEVELOPMENT_DATASETS + [CIC, CSE]:
        groups = [g for g in result["descriptive_groups"] if g["dataset"] == dataset]
        if not groups:
            continue
        rows = [[escape(NAMES[g["method"]]) + (r"$^\star$" if g["selected_main"] else ""), "5"] +
                [value(g["metrics"][metric]) for metric in METRICS] for g in groups]
        note = (r"Mean$\pm$sample SD over all five fixed formal seeds; signed forgetting is in percentage points. "
                r"$\star$ identifies the validation-selected main, never a test-selected winner. ")
        note += ("CIC17 was previously exposed during development; this fixed selected candidate is descriptive, not blind confirmation. "
                 "CIC17 was not used in the frozen UNSW/NSL validation selection. No CSE criterion is tested here."
                 if dataset == CIC else
                 "These are exposed development-dataset results, not blind confirmation. All five candidates are retained as they complete."
                 if dataset in DEVELOPMENT_DATASETS else
                 "External data are released only after all 35 reference cells finish. Replay+LA aliases calibrated Replay and adds no training.")
        fragments.append(table(escape(DATA[dataset]) + r": compact retention at stagewise $\alpha=2\%$.",
            "compact-results-" + dataset.lower().replace("-", "_"), "lrrrrrrr",
            ["Method", "$n$", "Mean inc. BAcc", "Final BAcc", "Forget (pp)", "ADR", "MF1", "FPR"], rows, note))
    capacity_rows = []
    for group in result["descriptive_groups"]:
        if group["method"] not in VERSIONS:
            continue
        capacities = [r["architecture_capacity"] for r in group["resources_by_seed"]]
        capacity = capacities[0]
        if any(row != capacity for row in capacities):
            raise ValueError("Model architecture differs across formal seeds")
        capacity_rows.append([escape(DATA[group["dataset"]]), escape(NAMES[group["method"]]),
            str(capacity["training_parameters"]), str(capacity["optimizer_parameters"]),
            str(capacity["inference_parameters"]), str(capacity["extra_components"])])
    if capacity_rows:
        fragments.append(table("Compact family parameter and component counts.", "compact-capacity", "llrrrr",
            ["Data", "Method", "Training-resident", "Optimizer", "Inference", "Extra components"], capacity_rows,
            "F only; the common G is separate. Counts are fixed across stages. EMA training-resident count includes its frozen teacher; inference uses only EMA. "
            "DUAL is one F with two fixed encoders and a joint head. Sample, calibration and transient storage and separate gate/family costs remain in the generated JSON."))
    tests = result["CSE18_necessary_tests"]
    if any(test.get("n") == 5 for test in tests):
        rows = []
        for test in tests:
            delta = test.get("delta_pp", {})
            effect = "--" if delta.get("mean") is None else f"{delta['mean']:.2f}" + rf"$\pm${delta['sample_sd']:.2f}"
            p = number(test.get("p_" + test["p_direction"])) if test["p_direction"] else "not used"
            sign_p = number(test.get("exact_sign_flip", {}).get("p_" + test["p_direction"])) if test["p_direction"] else "not used"
            rows.append([str(test["criterion"]), escape(NAMES.get(test.get("reference"), "pending")), str(test["n"]), effect, p, sign_p, test["decision"]])
        fragments.append(table("External-dataset necessary tests for the fixed validation-selected main.", "compact-cse-tests", "llrrrrl",
            ["Criterion", "Reference", "$n$", r"$\Delta$ (pp)", "$t$ $p$", "Sign-flip $p$", "Decision"], rows,
            r"1: forgetting harm is $p_{>}<.05$ against the lowest-mean external-five member. "
            r"2: final BAcc superiority over Replay-BS requires $p_{>}<.05$. "
            r"3: mean final-BAcc difference from original AA must be at least $-2$pp; no $p$-test is used. "
            r"4: ADR harm is $p_{<}<.05$ against original AA. No detected harm is not equivalence. "
            "Exact sign-flip p-values enumerate all 32 assignments as a sensitivity check. "
            "Overall numerical decision: " + escape(result["CSE18_necessary_verdict"]) + "."))
    return "% Generated only from complete saved metric groups; no inference or test selection.\n" + "\n".join(fragments)


def verify_saved_bindings(bindings):
    """Check provenance without recomputing any previously published group."""
    for binding in bindings:
        for label, sha in (("stagewise_result", "stagewise_sha256"), ("source_result", "source_sha256"),
                           ("adapter_result", "adapter_sha256")):
            if label not in binding:
                if label == "source_result":
                    raise ValueError("Missing mandatory saved source binding")
                continue
            if digest(source_path(binding[label])) != binding[sha]:
                raise ValueError("Previously consumed source changed; no silent regeneration")


def complete_cse_release(args, selection, versions):
    """Consume the closed native releases once; never recalculate old groups."""
    previous = read(args.output)
    expected = {d + "/" + m for d in DEVELOPMENT_DATASETS for m in VERSIONS} | {CIC + "/aacil_kd_wa"}
    if (previous.get("selection_sha256") != digest(args.selection)
            or set(previous["completed_group_keys"]) != expected
            or previous.get("cse_release_guard") is not None
            or previous.get("CSE18_necessary_tests") != []
            or "cse_release_append" in previous
            or len(previous["descriptive_groups"]) != 11
            or len(previous["source_bindings"]) != 55):
        raise ValueError("Requires intact eleven-group pre-release comparison; cannot repeat CSE tests")
    old_group_keys = {(g["dataset"], g["method"]) for g in previous["descriptive_groups"]}
    if (len(old_group_keys) != 11 or
            {(b["dataset"], b["method"], b["seed"]) for b in previous["source_bindings"]}
            != {(d, m, s) for d, m in old_group_keys for s in SEEDS}):
        raise ValueError("Prior group/source inventory is incomplete")
    verify_saved_bindings(previous["source_bindings"])
    root = args.cse_reference_root.resolve()
    selected_root = args.cse_selected_root.resolve()
    method = selection["selected_main"]["method"]
    version = versions[method]
    selected_paths = [selected_root / "cells" / method / f"seed_{s}" / "section32_result.json" for s in SEEDS]
    summary_path = selected_root / "selected_main_summary.json"
    if not all(p.is_file() for p in selected_paths) or not summary_path.is_file():
        raise ValueError("All five selected completion files and native release required")
    paths, release, guard = cse_release_guard(root)
    selected_release = read(summary_path)
    if (selected_release.get("completed") is not True or selected_release.get("smoke") is not False
            or selected_release.get("test_selected") is not False or selected_release.get("physical_cells") != 5
            or selected_release.get("seeds") != SEEDS or selected_release.get("method") != method
            or selected_release.get("source_results") != [
                {"path": p.relative_to(selected_root).as_posix(), "sha256": digest(p)} for p in selected_paths]):
        raise ValueError("Invalid or stale selected release")
    groups, rows, bindings = load_cse_references(root, paths, release)
    resources = []
    for seed, path in zip(SEEDS, selected_paths):
        wrapper = read(path)
        raw_path = (path.parent / wrapper["raw_result"]).resolve()
        raw_path.relative_to(path.parent.resolve())
        raw = read(raw_path)
        identity, config = raw["identity"], raw["identity"]["effective_config"]
        frozen = identity.get("frozen_selection", {})
        if (wrapper.get("completed") is not True or wrapper.get("smoke") is not False
                or (wrapper.get("dataset"), wrapper.get("method"), wrapper.get("seed")) != (CSE, method, seed)
                or wrapper.get("raw_result_sha256") != digest(raw_path)
                or wrapper["identity"] != identity or raw.get("completed") is not True
                or raw.get("test_evaluation_performed") is not True or raw.get("selection_only")
                or config.get("smoke") or config.get("selection_only") or config.get("section31") is not True
                or identity["git_commit"] != selection["source_commit"]
                or frozen.get("selection_sha256") != previous["selection_sha256"]
                or frozen.get("candidate_id") != version["selected_candidate_id"]
                or frozen.get("eligible") != version["eligible"]
                or any(config.get(v) != version["configuration"][k] for k, v in CONFIG_KEYS.items())
                or stamp(raw["completed_at"]) < stamp(selection["completed_at"])):
            raise ValueError("Selected CSE execution differs from frozen validation choice")
        if wrapper.get("cse_execution") != raw.get("cse_execution") or raw["cse_execution"]["cell"]["exit_code"] != 0:
            raise ValueError("CSE execution provenance is incomplete")
        rows[CSE, method, seed] = {
            "metrics": wrapper["metrics"], "comparison_input": relative(path),
            "class_order": identity["class_order"], "tasks": identity["tasks"],
            "training_git_commit": identity["git_commit"],
        }
        resources.append({"seed": seed, **capacity_and_resources(raw),
                          "cse_execution": raw["cse_execution"],
                          "adapter_cost_seconds": wrapper["adapter_cost_seconds"]})
        bindings.append({"dataset": CSE, "method": method, "seed": seed,
                         "source_result": relative(raw_path), "source_sha256": digest(raw_path),
                         "adapter_result": relative(path), "adapter_sha256": digest(path),
                         "source_completed_at": raw["completed_at"],
                         "training_git_commit": identity["git_commit"]})
    metrics = {k: stats([rows[CSE, method, s]["metrics"][k] for s in SEEDS]) for k in METRICS}
    for metric in METRICS:
        for field in ("mean", "sample_sd"):
            if not math.isclose(metrics[metric][field], selected_release["metrics"][metric][field],
                                rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("Selected release differs from saved per-seed metrics")
    groups[CSE, method] = {"dataset": CSE, "method": method, "n": 5, "seeds": SEEDS,
                          "complete_five_seed": True, "metrics": metrics, "resources_by_seed": resources,
                          "role": "fixed_main_external_dataset"}
    for seed in SEEDS:
        for reference in PHYSICAL_REFERENCES:
            if any(rows[CSE, method, seed][k] != rows[CSE, reference, seed][k] for k in ("class_order", "tasks")):
                raise ValueError("Candidate/reference class-task assignment mismatch")
    guard.update(selected_five_release_satisfied=True, selected_summary=relative(summary_path),
                 selected_summary_sha256=digest(summary_path), selected_released_at=selected_release["released_at"])
    # Only NEW CSE groups are supplied to summarize; the old eleven are opaque.
    new = summarize(selection, versions, groups, rows, [], bindings, args.selection.resolve(), guard)
    if any(new[k] != previous[k] for k in ("selected_main", "selection_summary", "main_selection_status")):
        raise ValueError("Frozen selection or prior disclosure changed")
    result = previous | {k: new[k] for k in (
        "completed_at", "cse_release_guard", "CSE18_necessary_tests", "CSE18_necessary_verdict",
        "CSE18_all_required_evidence_complete", "positioning")}
    result.update(
        descriptive_groups=previous["descriptive_groups"] + new["descriptive_groups"],
        source_bindings=previous["source_bindings"] + bindings,
        completed_group_keys=sorted(expected | set(new["completed_group_keys"])),
        cse_release_append={"previous_comparison_sha256": digest(args.output),
            "previous_completed_at": previous["completed_at"], "added_physical_cells": 40,
            "previous_groups_recomputed": False, "selection_changed": False,
            "training_inference_or_confusion_recomputed": False,
            "new_registered_comparisons": 4, "exact_sign_flip_assignments_each": 32},
        limitations=previous["limitations"] + [
            "CSE reference seeds2/42 and selected seed42 ran locally; remaining seeds ran on HPC A100. "
            "Training identities and data/task protocol are bound; wall times are not hardware-matched.",
            "Exact sign-flip p-values are sensitivity checks; the registered paired-t and mean-difference decision rules are unchanged."])
    text = render(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    args.tex_output.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"written": True, "completed_groups": len(result["descriptive_groups"]),
                      "previous_groups_recomputed": False, "new_registered_comparisons": 4,
                      "CSE18_necessary_verdict": result["CSE18_necessary_verdict"]}))


def append_section39(args, selection, versions):
    """Append only the authorized CIC17 candidate; preserve all prior statistics."""
    previous = read(args.output)
    expected = {d + "/" + m for d in DEVELOPMENT_DATASETS for m in VERSIONS}
    key = CIC + "/aacil_kd_wa"
    if (previous.get("selection_sha256") != digest(args.selection)
            or previous.get("selected_main") != {k: selection["selected_main"][k] for k in
                ("method", "selected_candidate_id", "configuration", "eligible", "selection_reason")}
            or previous.get("selection_datasets") != DEVELOPMENT_DATASETS
            or previous.get("selection_test_metrics_used") is not False
            or previous.get("cse_release_guard") is not None
            or previous.get("CSE18_necessary_tests") != []
            or previous.get("CSE18_necessary_verdict") != "pending"
            or previous.get("CSE18_all_required_evidence_complete") is not False
            or set(previous.get("completed_group_keys", [])) not in (expected, expected | {key})):
        raise ValueError("Section39 append requires the intact ten-group sealed-CSE comparison")
    bindings = previous["source_bindings"]
    group_keys = [(g["dataset"], g["method"]) for g in previous["descriptive_groups"]]
    source_keys = [(b["dataset"], b["method"], b["seed"]) for b in bindings]
    if (len(group_keys) != len(set(group_keys))
            or {d + "/" + m for d, m in group_keys} != set(previous["completed_group_keys"])
            or len(source_keys) != len(set(source_keys))
            or set(source_keys) != {(d, m, s) for d, m in group_keys for s in SEEDS}):
        raise ValueError("Previously published group/source inventory is incomplete")
    verify_saved_bindings(bindings)
    if key in previous["completed_group_keys"]:
        print(json.dumps({"written": False, "reason": "Section39 already appended; no statistics recomputed"}))
        return
    group, cells, missing, sources = load_compact_group(
        args.input.resolve(), CIC, "aacil_kd_wa", selection, versions["aacil_kd_wa"])
    if missing:
        raise ValueError("Section39 requires all five completed CIC17 candidate cells")
    version = versions["aacil_kd_wa"]
    if selection["selected_main"]["method"] != "aacil_kd_wa":
        raise ValueError("Section39 cannot change the frozen selected candidate")
    for resource, seed in zip(group["resources_by_seed"], SEEDS):
        row = cells[CIC, "aacil_kd_wa", seed]
        source = read(source_path(row["source_result"]))
        execution = row.get("section39_execution", {})
        invocation = execution.get("invocation", {})
        frozen = execution.get("wrapper_identity", {}).get("frozen_selection", {})
        if (execution != source.get("section39_execution") or not execution
                or invocation.get("actual_entrypoint") != "ncd/run_section39_local.py"
                or (invocation.get("dataset"), invocation.get("method"), invocation.get("seed")) != (CIC, "aacil_kd_wa", seed)
                or invocation.get("smoke") is not False
                or invocation.get("selection_changed") is not False
                or invocation.get("descriptive_development_dataset") is not True
                or invocation.get("CSE18_model_metrics_accessed") is not False
                or execution.get("cell_attempt", {}).get("exit_code") != 0
                or frozen.get("selection_sha256") != previous["selection_sha256"]
                or frozen.get("candidate_id") != version["selected_candidate_id"]
                or frozen.get("eligible") != version["eligible"]):
            raise ValueError("Section39 provenance or frozen-selection identity differs")
        resource["section39_execution"] = execution
    group.update(role="development_dataset_descriptive", selected_main=True,
                 development_eligible=version["eligible"], selected_candidate_id=version["selected_candidate_id"],
                 used_for_frozen_selection=False, blind_confirmation=False)
    result = previous | {
        "completed_at": datetime.now().astimezone().isoformat(),
        "descriptive_groups": previous["descriptive_groups"] + [group],
        "source_bindings": bindings + sources,
        "completed_group_keys": sorted(expected | {key}),
        "section39_append": {"previous_comparison_sha256": digest(args.output),
            "previous_completed_at": previous["completed_at"], "added_group": key,
            "previous_groups_recomputed": False, "CSE18_metrics_read": False,
            "new_significance_tests": False, "selection_changed": False},
        "limitations": previous["limitations"] + [
            "CIC17 selected-candidate results are descriptive development evidence, not blind confirmation. "
            "CIC17 was not part of the frozen UNSW/NSL seeds5/6 selection. No CSE18 criterion is applied to CIC17."]}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    args.tex_output.write_text(render(result), encoding="utf-8", newline="\n")
    print(json.dumps({"written": True, "added_group": key, "completed_groups": 11,
                      "previous_groups_recomputed": False, "new_significance_tests": False,
                      "CSE18_necessary_verdict": result["CSE18_necessary_verdict"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "ncd/results/problem_scout/section31/development_v1/selection.json")
    parser.add_argument("--input", type=Path, default=ROOT / "ncd/results/problem_scout/section28/forgetting_v1")
    parser.add_argument("--include-cse", action="store_true", help="Requires all35 reference completion files and guarded release before any CSE metric read")
    parser.add_argument("--append-section39", action="store_true", help="Append only completed CIC17 KD+WA; retain the original ten descriptive groups and all CSE decisions")
    parser.add_argument("--complete-cse-release", action="store_true", help="Append closed native CSE releases once without recomputing development groups")
    parser.add_argument("--cse-selected-root", type=Path)
    parser.add_argument("--cse-reference-root", type=Path, help="Local retrieved protected reference tree containing cells/ and reference_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "ncd/results/problem_scout/section31/comparisons_v1/result.json")
    parser.add_argument("--tex-output", type=Path, default=ROOT / "paper_aacil/generated/section31_comparisons.tex")
    args = parser.parse_args()
    if bool(args.cse_reference_root) != args.include_cse:
        parser.error("--include-cse and --cse-reference-root must be supplied together")
    if args.append_section39 and args.include_cse:
        parser.error("Section39 descriptive append cannot consume CSE metrics")
    if bool(args.cse_selected_root) != args.complete_cse_release or (args.complete_cse_release and not args.include_cse):
        parser.error("--complete-cse-release requires --include-cse and both CSE release roots")
    selection, versions = load_selection(args.selection.resolve())
    if args.append_section39:
        append_section39(args, selection, versions)
        return
    if args.complete_cse_release:
        complete_cse_release(args, selection, versions)
        return
    groups, rows, missing, bindings = {}, {}, [], []
    guard = None
    if args.include_cse:
        # Fail closed even before reading local development metrics if the CSE
        # opt-in was premature; never expose a partially completed CSE summary.
        paths, release, guard = cse_release_guard(args.cse_reference_root.resolve())
    for dataset in DEVELOPMENT_DATASETS:
        for method in VERSIONS:
            group, cells, absent, sources = load_compact_group(args.input.resolve(), dataset, method, selection, versions[method])
            if group:
                groups[dataset, method] = group
                rows.update(cells)
                bindings.extend(sources)
            else:
                missing.append(absent)
    if args.include_cse:
        references, cells, sources = load_cse_references(args.cse_reference_root.resolve(), paths, release)
        groups.update(references)
        rows.update(cells)
        bindings.extend(sources)
        method = selection["selected_main"]["method"]
        group, cells, absent, sources = load_compact_group(args.input.resolve(), CSE, method, selection, versions[method])
        if group:
            groups[CSE, method] = group
            rows.update(cells)
            bindings.extend(sources)
        else:
            missing.append(absent)
    if not groups:
        print(json.dumps({"written": False, "reason": "No new complete formal group; no all-pending output generated"}))
        return
    result = summarize(selection, versions, groups, rows, missing, bindings, args.selection.resolve(), guard)
    if args.output.is_file():
        previous = read(args.output)
        # The normal CSE-release path must retain the independently appended
        # CIC17 descriptive evidence, without treating it as selection data.
        if "section39_append" in previous:
            cic_groups = [g for g in previous["descriptive_groups"] if g["dataset"] == CIC]
            cic_sources = [b for b in previous["source_bindings"] if b["dataset"] == CIC]
            verify_saved_bindings(cic_sources)
            if len(cic_groups) != 1 or len(cic_sources) != 5:
                raise ValueError("Incomplete previously appended Section39 evidence")
            result["descriptive_groups"] += cic_groups
            result["source_bindings"] += cic_sources
            result["completed_group_keys"].append(CIC + "/aacil_kd_wa")
            result["completed_group_keys"].sort()
            result["section39_append"] = previous["section39_append"]
            result["limitations"] += [v for v in previous["limitations"] if v.startswith("CIC17 selected-candidate")]
        if previous.get("selection_sha256") != result["selection_sha256"]:
            raise ValueError("Existing comparison belongs to another immutable selection; use a new output namespace")
        old_keys, new_keys = set(previous.get("completed_group_keys", [])), set(result["completed_group_keys"])
        if not old_keys.issubset(new_keys):
            raise ValueError("A previously published complete group disappeared; no overwrite")
        old_bindings = {(r["dataset"], r["method"], r["seed"]): r for r in previous.get("source_bindings", [])}
        new_bindings = {(r["dataset"], r["method"], r["seed"]): r for r in result["source_bindings"]}
        if any(new_bindings.get(key) != row for key, row in old_bindings.items()):
            raise ValueError("Previously consumed source changed; do not regenerate silently")
        if old_keys == new_keys:
            print(json.dumps({"written": False, "reason": "No newly completed group; existing analysis and tables retained"}))
            return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    args.tex_output.write_text(render(result), encoding="utf-8")
    print(json.dumps({"written": True, "completed_groups": len(groups), "selected_main": result["selected_main"]["method"],
                      "CSE18_necessary_verdict": result["CSE18_necessary_verdict"]}))


if __name__ == "__main__":
    main()
