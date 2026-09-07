"""Read-only, fail-closed evidence accounting. No training or submission code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


class Blocked(ValueError):
    pass


LOCKS = {
    "split_registry_sha256", "task_order_sha256", "available_labels_sha256",
    "preprocessing_sha256", "backbone_spec_sha256", "initialization_policy_sha256",
    "checkpoint_policy_sha256", "evaluation_support_sha256", "code_manifest_sha256",
    "environment_sha256", "profile_evidence_sha256", "method_specs_sha256",
}
COUNTERS = {
    "optimizer_steps", "raw_row_presentations", "unique_raw_rows",
    "gradient_raw_row_presentations", "fit_raw_row_presentations",
    "old_raw_row_presentations", "new_raw_row_presentations",
    "binary_targets", "multiclass_targets",
}
CATEGORIES = {
    "encoder_parameters", "head_parameters", "centroid_state", "replay_state",
    "optimizer_state", "other_persistent_state",
}
MEMORY = {
    "parameter_numel", "parameter_bytes", "centroid_bytes", "replay_bytes",
    "optimizer_bytes", "optimizer_peak_bytes", "other_bytes",
    "checkpoint_retained_bytes", "host_peak_rss_bytes",
    "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
}
METRICS = COUNTERS | MEMORY


def need(condition, message):
    if not condition:
        raise Blocked(message)


def keys(value, required, context):
    need(isinstance(value, dict), f"{context}: expected object")
    need(set(value) == set(required), f"{context}: missing or unknown keys")


def integer(value, context, minimum=0):
    need(type(value) is int and value >= minimum, f"{context}: invalid integer")
    return value


def digest(value):
    need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def nonempty(value, context):
    need(isinstance(value, str) and bool(value.strip()), f"{context}: empty string")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def decode(data):
    def bad_constant(value):
        raise Blocked(f"nonfinite JSON constant: {value}")
    def finite_float(value):
        parsed = float(value)
        need(math.isfinite(parsed), "nonfinite JSON number")
        return parsed
    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=_pairs,
                      parse_constant=bad_constant, parse_float=finite_float)


def artifact(root, ref):
    keys(ref, {"path", "sha256"}, "artifact reference")
    digest(ref["sha256"])
    raw = ref["path"]
    need(isinstance(raw, str) and raw and not re.search(r"[:\\]", raw), "unsafe artifact path")
    relative = Path(raw)
    need(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path")
    base = root.resolve()
    path = (base / relative).resolve()
    need(path.is_relative_to(base) and path != base, "artifact escapes root")
    data = path.read_bytes()
    need(hashlib.sha256(data).hexdigest() == ref["sha256"], "artifact hash mismatch")
    return decode(data)


def protocol_check(p):
    need(isinstance(p, dict) and p.get("status") == "BOUND", "protocol is preparation only or unbound")
    keys(p, {"schema_version", "status", "evidence_kind", "dataset", "locks", "seeds",
             "tasks", "available_labels", "shared_initial_encoder_sha256",
             "shared_task0_encoder_sha256", "checkpoint_policy", "arms", "contrasts",
             "real_measurements", "submission_authorized"}, "protocol")
    need(type(p["schema_version"]) is int and p["schema_version"] == 1, "schema version")
    need(p["evidence_kind"] in {"real", "synthetic"}, "evidence kind")
    need(p["submission_authorized"] is False, "accounting cannot authorize submission")
    need(p["real_measurements"] == ("SYNTHETIC_ONLY" if p["evidence_kind"] == "synthetic"
                                    else "SUPPLIED_IN_ACCOUNTING"), "measurement status")
    nonempty(p["dataset"], "dataset")
    keys(p["locks"], LOCKS, "locks")
    for value in p["locks"].values():
        digest(value)
    seeds = p["seeds"]
    need(isinstance(seeds, list) and seeds, "seeds missing")
    for seed in seeds:
        integer(seed, "seed")
    need(len(set(seeds)) == len(seeds), "duplicate seeds")
    for name in ("shared_initial_encoder_sha256", "shared_task0_encoder_sha256"):
        keys(p[name], {str(seed) for seed in seeds}, name)
        for value in p[name].values():
            digest(value)
    tasks = p["tasks"]
    need(isinstance(tasks, list) and tasks, "tasks missing")
    seen = set()
    available = []
    for task in tasks:
        need(isinstance(task, list) and task, "empty task")
        for c in task:
            integer(c, "class id")
            need(c not in seen, "repeated task class")
            seen.add(c)
        available.append(sorted(seen))
    need(p["available_labels"] == available, "available labels must be cumulative task labels")
    need(p["checkpoint_policy"] == "fixed_last_registered_update_no_test_selection",
         "v1 supports fixed last update only; calibration selection needs a new schema")
    arms = p["arms"]
    need(isinstance(arms, list) and arms, "arms missing")
    by_arm = {}
    for arm in arms:
        keys(arm, {"id", "status", "phase_plan"}, "arm")
        nonempty(arm["id"], "arm id")
        need(arm["id"] not in by_arm and arm["status"] == "bound", "unbound/duplicate arm")
        phases = arm["phase_plan"]
        need(isinstance(phases, list) and phases, "phase plan missing")
        phase_ids = set()
        covered = set()
        previous_task = -1
        for phase in phases:
            keys(phase, {"id", "task", "gradient", "target_kind", "device", "na_reason", "limits"}, "phase")
            nonempty(phase["id"], "phase id")
            need(phase["id"] not in phase_ids, "duplicate phase")
            phase_ids.add(phase["id"])
            t = integer(phase["task"], "task")
            need(previous_task <= t < len(tasks), "phase task order")
            previous_task = t
            if phase["id"] != "shared_pretrain":
                covered.add(t)
            need(type(phase["gradient"]) is bool, "gradient flag must be bool")
            need(phase["target_kind"] in {"binary", "multiclass", "none"}, "target kind")
            need(phase["device"] in {"cpu", "cuda"}, "device")
            if phase["gradient"]:
                need(phase["target_kind"] != "none" and phase["na_reason"] is None, "gradient target contract")
            else:
                need(phase["target_kind"] == "none", "N/A phase has targets")
                nonempty(phase["na_reason"], "no-gradient N/A reason")
            keys(phase["limits"], METRICS, "phase limits")
            for name, value in phase["limits"].items():
                if name.startswith("cuda_") and phase["device"] == "cpu":
                    need(value is None, "CPU device limits must be N/A")
                else:
                    integer(value, "measured registered limit")
        first = phases[0]
        need(first["id"] == "shared_pretrain" and first["task"] == 0 and first["gradient"]
             and first["target_kind"] == "multiclass", "missing shared pretraining cost phase")
        need(covered == set(range(len(tasks))), "incomplete task phase coverage")
        by_arm[arm["id"]] = arm
    need(isinstance(p["contrasts"], list) and p["contrasts"], "register at least one contrast")
    for contrast in p["contrasts"]:
        keys(contrast, {"arms", "estimand", "residual_confounds", "match_metrics"}, "contrast")
        pair = contrast["arms"]
        need(isinstance(pair, list) and len(pair) == 2 and len(set(pair)) == 2
             and all(x in by_arm for x in pair), "contrast arms")
        nonempty(contrast["estimand"], "estimand")
        need(isinstance(contrast["residual_confounds"], list), "residual confounds")
        for residual in contrast["residual_confounds"]:
            nonempty(residual, "residual confound")
        matched = contrast["match_metrics"]
        need(isinstance(matched, list) and len(matched) == len(set(matched))
             and set(matched) <= METRICS, "match metrics")
        nonpretrain_grad = [any(x["gradient"] for x in by_arm[a]["phase_plan"]
                               if x["id"] != "shared_pretrain") for a in pair]
        if len(set(nonpretrain_grad)) > 1:
            need(not (set(matched) & {"optimizer_steps", "gradient_raw_row_presentations",
                                      "binary_targets", "multiclass_targets"}), "NME optimization matching is N/A")
    return by_arm


def registry_check(value, p):
    keys(value, {"rows"}, "split registry")
    need(isinstance(value["rows"], list) and value["rows"], "empty registry")
    rows = {}
    for row in value["rows"]:
        keys(row, {"id", "class_id", "partition", "available_from_task"}, "registry row")
        digest(row["id"])
        need(row["id"] not in rows, "duplicate original raw row")
        c = integer(row["class_id"], "registry class")
        t = integer(row["available_from_task"], "row availability")
        need(t < len(p["tasks"]) and c in p["available_labels"][t], "row label unavailable")
        need(row["partition"] in {"train", "calibration", "test"}, "partition")
        rows[row["id"]] = row
    return rows


def trace_check(value, registry, p, phase):
    keys(value, {"events"}, "trace")
    need(isinstance(value["events"], list) and value["events"], "empty measured trace")
    totals = dict.fromkeys(COUNTERS, 0)
    seen = set()
    pending = 0
    for event in value["events"]:
        need(isinstance(event, dict), "invalid event")
        kind = event.get("kind")
        if kind == "step":
            keys(event, {"kind"}, "step")
            need(phase["gradient"] and pending > 0, "orphan or prohibited optimizer step")
            totals["optimizer_steps"] += 1
            pending = 0
            continue
        keys(event, {"kind", "rows", "binary_targets", "multiclass_targets"}, "consumption event")
        need(kind == ("microbatch" if phase["gradient"] else "fit_read"), "event/phase mismatch")
        ids = event["rows"]
        need(isinstance(ids, list) and ids, "empty consumed batch")
        for rid in ids:
            digest(rid)
            need(rid in registry, "unknown original row")
            row = registry[rid]
            need(row["partition"] == "train", "non-training row in fitting trace")
            need(row["available_from_task"] <= phase["task"]
                 and row["class_id"] in p["available_labels"][phase["task"]], "future row/label exposure")
            old = row["class_id"] not in p["tasks"][phase["task"]]
            totals["old_raw_row_presentations" if old else "new_raw_row_presentations"] += 1
            seen.add(rid)
        totals["raw_row_presentations"] += len(ids)
        totals["gradient_raw_row_presentations" if phase["gradient"] else "fit_raw_row_presentations"] += len(ids)
        covered = set()
        target_pairs = set()
        for field, count in (("binary_targets", 3), ("multiclass_targets", 2)):
            targets = event[field]
            need(isinstance(targets, list), "target list")
            expected = phase["target_kind"] + "_targets"
            need(field == expected or not targets, "wrong loss/target representation")
            for target in targets:
                need(isinstance(target, list) and len(target) == count, "target tuple shape")
                index = integer(target[0], "target row")
                c = integer(target[1], "target class")
                need(index < len(ids) and c in p["available_labels"][phase["task"]], "target unavailable")
                need((index, c) not in target_pairs, "duplicate target decision")
                target_pairs.add((index, c))
                truth = registry[ids[index]]["class_id"]
                if count == 3:
                    integer(target[2], "binary target")
                    need(target[2] in (0, 1) and target[2] == int(c == truth), "wrong binary label")
                else:
                    need(c == truth and index not in covered, "wrong/repeated multiclass label")
                covered.add(index)
            totals[field] += len(targets)
        if phase["gradient"]:
            need(covered == set(range(len(ids))), "target coverage mismatch")
            pending += 1
        else:
            need(not covered, "fit phase has gradient targets")
    need(pending == 0, "uncommitted gradient microbatch")
    need(not phase["gradient"] or totals["optimizer_steps"] > 0, "no measured training steps")
    totals["unique_raw_rows"] = len(seen)
    return totals, seen


def memory_check(value, phase):
    keys(value, {"scope", "snapshot_id", "inventory_method", "peak_method", "components",
                 "optimizer_peak_bytes", "host_peak_rss_bytes", "cuda_peak_allocated_bytes",
                 "cuda_peak_reserved_bytes", "device_na_reason"}, "memory")
    need(value["scope"] == "phase_boundary_live_storage_and_process_tree_phase_peaks", "memory scope")
    for key in ("snapshot_id", "inventory_method", "peak_method"):
        nonempty(value[key], key)
    components = value["components"]
    need(isinstance(components, list) and components, "empty memory inventory")
    totals = dict.fromkeys(CATEGORIES, 0)
    categories = set()
    stores = set()
    parameters = 0
    for item in components:
        keys(item, {"storage_id", "category", "numel", "itemsize", "bytes"}, "storage")
        nonempty(item["storage_id"], "storage id")
        need(item["storage_id"] not in stores, "storage counted twice")
        stores.add(item["storage_id"])
        cat = item["category"]
        need(cat in CATEGORIES, "unknown memory category")
        categories.add(cat)
        n = integer(item["numel"], "storage numel")
        size = integer(item["itemsize"], "storage itemsize", 1)
        b = integer(item["bytes"], "storage bytes")
        need(b == n * size, "storage byte arithmetic mismatch")
        totals[cat] += b
        if cat.endswith("parameters"):
            parameters += n
    need(categories == CATEGORIES, "missing memory category, including explicit zero categories")
    result = {
        "parameter_numel": parameters,
        "parameter_bytes": totals["encoder_parameters"] + totals["head_parameters"],
        "centroid_bytes": totals["centroid_state"], "replay_bytes": totals["replay_state"],
        "optimizer_bytes": totals["optimizer_state"], "other_bytes": totals["other_persistent_state"],
        "checkpoint_retained_bytes": sum(totals.values()) - totals["optimizer_state"],
    }
    for name in ("optimizer_peak_bytes", "host_peak_rss_bytes"):
        result[name] = integer(value[name], name)
    need(result["host_peak_rss_bytes"] > 0, "host peak was not measured")
    need(result["optimizer_peak_bytes"] >= result["optimizer_bytes"], "optimizer peak below snapshot")
    cuda = ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
    if phase["device"] == "cpu":
        need(all(value[x] is None for x in cuda) and value["device_na_reason"] == "cpu_only_phase",
             "CPU CUDA measures must be explicit N/A")
    else:
        need(value["device_na_reason"] is None, "CUDA incorrectly N/A")
        for name in cuda:
            integer(value[name], name, 1)
        need(value[cuda[1]] >= value[cuda[0]], "CUDA reserved peak below allocated peak")
    result.update({name: value[name] for name in cuda})
    return result


def validate(p, report, root):
    arms = protocol_check(p)
    keys(report, {"schema_version", "evidence_kind", "protocol_sha256", "binding_artifacts", "records"}, "report")
    need(type(report["schema_version"]) is int and report["schema_version"] == 1, "report schema")
    need(report["evidence_kind"] == p["evidence_kind"], "synthetic/real evidence mismatch")
    need(report["protocol_sha256"] == canonical_hash(p), "stale protocol binding")
    keys(report["binding_artifacts"], LOCKS, "binding artifacts")
    bindings = {}
    for key, ref in report["binding_artifacts"].items():
        need(ref["sha256"] == p["locks"][key], "lock/ref mismatch")
        bindings[key] = artifact(root, ref)
    need(bindings["task_order_sha256"] == p["tasks"], "task artifact mismatch")
    need(bindings["available_labels_sha256"] == p["available_labels"], "label artifact mismatch")
    registry = registry_check(bindings["split_registry_sha256"], p)
    method_specs = bindings["method_specs_sha256"]
    keys(method_specs, arms, "method specs")
    for spec in method_specs.values():
        keys(spec, {"loss_policy", "target_assignment", "sampling_policy", "trainable_parameters",
                    "optimizer_schedule", "checkpoint_policy", "initial_head_state_sha256"}, "method spec")
        for field in set(spec) - {"initial_head_state_sha256"}:
            nonempty(spec[field], field)
        need(spec["checkpoint_policy"] == p["checkpoint_policy"], "method checkpoint policy mismatch")
        keys(spec["initial_head_state_sha256"], {str(s) for s in p["seeds"]}, "head initial state")
        for value in spec["initial_head_state_sha256"].values():
            digest(value)
    expected = {(a, seed, phase["id"]): phase for a, arm in arms.items()
                for seed in p["seeds"] for phase in arm["phase_plan"]}
    seen = set()
    summaries = {}
    shared_pretraining = {}
    need(isinstance(report["records"], list), "records must be list")
    for record in report["records"]:
        keys(record, {"arm", "seed", "phase", "initial_encoder_sha256", "task0_encoder_sha256",
                      "initial_head_state_sha256", "trace", "memory", "counters", "memory_metrics", "measurement_status"}, "record")
        integer(record["seed"], "record seed")
        key = (record["arm"], record["seed"], record["phase"])
        need(key in expected and key not in seen, "missing/duplicate/unknown record identity")
        seen.add(key)
        phase = expected[key]
        need(record["initial_head_state_sha256"] == method_specs[record["arm"]]["initial_head_state_sha256"][str(record["seed"])],
             "head initialization mismatch")
        for field, lock in (("initial_encoder_sha256", "shared_initial_encoder_sha256"),
                            ("task0_encoder_sha256", "shared_task0_encoder_sha256")):
            need(record[field] == p[lock][str(record["seed"])], "unequal initial/shared encoder state")
        need(record["measurement_status"] == ("synthetic" if p["evidence_kind"] == "synthetic"
                                               else "instrumented_actual"), "unmeasured record")
        counts, ids = trace_check(artifact(root, record["trace"]), registry, p, phase)
        memory = memory_check(artifact(root, record["memory"]), phase)
        for submitted, computed in ((record["counters"], counts), (record["memory_metrics"], memory)):
            keys(submitted, computed, "reported metrics")
            for name, value in submitted.items():
                if value is not None:
                    integer(value, "reported metric")
                need(value == computed[name], f"reported metric mismatch: {name}")
        metrics = counts | memory
        if record["phase"] == "shared_pretrain":
            signature = (record["trace"]["sha256"], record["memory"]["sha256"], canonical_hash(metrics))
            previous = shared_pretraining.setdefault(record["seed"], signature)
            need(previous == signature, "shared pretraining evidence/cost differs between arms")
        for name, actual in metrics.items():
            limit = phase["limits"][name]
            need((actual is None and limit is None) or
                 (actual is not None and limit is not None and actual <= limit), f"limit exceeded: {name}")
        summary = summaries.setdefault(key[:2], {"metrics": {}, "rows": set()})
        summary["rows"].update(ids)
        for name, value in metrics.items():
            old = summary["metrics"].get(name)
            if name in COUNTERS:
                summary["metrics"][name] = (old or 0) + value
            elif value is None:
                summary["metrics"].setdefault(name, None)
            else:
                summary["metrics"][name] = max(old or 0, value)
    need(seen == set(expected), "incomplete arm/seed/phase coverage")
    for summary in summaries.values():
        summary["metrics"]["unique_raw_rows"] = len(summary["rows"])
    for contrast in p["contrasts"]:
        a, b = contrast["arms"]
        for seed in p["seeds"]:
            for metric in contrast["match_metrics"]:
                left, right = summaries[(a, seed)]["metrics"][metric], summaries[(b, seed)]["metrics"][metric]
                need(left is not None and right is not None and left == right, f"unmatched axis: {metric}")
    return {"status": "ACCOUNTING_CONSISTENT", "synthetic_only": p["evidence_kind"] == "synthetic",
            "records_checked": len(seen), "submission_authorized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("accounting", nargs="?", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        p = decode(args.protocol.read_bytes())
        protocol_check(p)
        need(args.accounting is not None, "accounting evidence missing")
        result = validate(p, decode(args.accounting.read_bytes()), args.artifact_root)
    except (Blocked, OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc), "submission_authorized": False}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
