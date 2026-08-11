"""Publish one validated formal-v2 SHAP/drift/ETG analysis as a new W&B run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from analyze import validate_output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_analysis(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = validate_output(path.resolve())
    analysis = json.loads((path / "analysis.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "analysis_manifest.json").read_text(encoding="utf-8"))
    if analysis["canonical_sha256"] != manifest["analysis_canonical_sha256"]:
        raise RuntimeError("analysis and manifest canonical hashes disagree")
    if validated.get("canonical_sha256") != manifest.get("canonical_sha256"):
        raise RuntimeError("validated manifest differs from the loaded manifest")
    return analysis, manifest


def load_governance(
    path: Path, *, runtime_dataset_id: str, destination: str
) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("data-governance root must be an object")
    if value.get("schema_version") != "ofra_data_governance_v3":
        raise RuntimeError("unsupported data-governance schema")
    if value.get("dataset") != "MalayaNetwork_GT":
        raise RuntimeError("data-governance canonical dataset mismatch")
    runtime_ids = value.get("runtime_dataset_ids")
    if (
        not isinstance(runtime_ids, list)
        or not runtime_ids
        or any(not isinstance(item, str) or not item for item in runtime_ids)
    ):
        raise RuntimeError("data-governance runtime dataset IDs are incomplete")
    if runtime_dataset_id not in runtime_ids:
        raise RuntimeError("data-governance runtime dataset mismatch")
    if value.get("license") != "CC-BY-4.0":
        raise RuntimeError("MalayaNetwork_GT publication requires CC-BY-4.0 binding")
    if value.get("wandb_destination") != destination:
        raise RuntimeError("data-governance W&B destination mismatch")
    policy = value.get("outbound_policy")
    if not isinstance(policy, dict) or policy.get("mode") != "aggregate_only_public_safe":
        raise RuntimeError("W&B publication is not approved for public-safe aggregates")
    required_text = (
        "source_url", "source_revision", "license_url", "attribution",
        "citation_bibtex", "derivative_description", "wandb_version",
    )
    if any(not isinstance(value.get(field), str) or not value[field].strip()
           for field in required_text):
        raise RuntimeError("data-governance attribution is incomplete")
    if not re.fullmatch(r"[0-9a-f]{40}", value["source_revision"]):
        raise RuntimeError("data-governance source revision is invalid")
    required_policy_lists = (
        "allowed_config_keys", "allowed_governance_config_keys",
        "allowed_performance_protocol_keys",
        "allowed_history_keys", "allowed_summary_prefixes",
        "forbidden_key_fragments", "forbidden_value_fragments",
    )
    if any(not isinstance(policy.get(field), list) or not policy[field]
           for field in required_policy_lists):
        raise RuntimeError("data-governance allow/deny lists are incomplete")
    if not isinstance(policy.get("allowed_tables"), dict) or not policy["allowed_tables"]:
        raise RuntimeError("data-governance table allowlist is incomplete")
    return value


def wandb_settings_kwargs() -> dict[str, Any]:
    """Return the fail-closed W&B telemetry settings bound by the tests."""
    return {
        "disable_git": True,
        "disable_code": True,
        "disable_job_creation": True,
        "console": "off",
        "quiet": True,
        "x_disable_meta": True,
        "x_disable_stats": True,
        "x_disable_machine_info": True,
        "x_save_requirements": False,
    }


def _reject_forbidden(value: Any, policy: dict[str, Any], label: str) -> None:
    key_fragments = [str(item).casefold() for item in policy["forbidden_key_fragments"]]
    value_fragments = [str(item).casefold() for item in policy["forbidden_value_fragments"]]
    if isinstance(value, dict):
        for key, item in value.items():
            folded = str(key).casefold()
            if any(fragment in folded for fragment in key_fragments):
                raise RuntimeError(f"forbidden outbound key in {label}: {key}")
            _reject_forbidden(item, policy, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden(item, policy, f"{label}[{index}]")
    elif isinstance(value, str):
        folded = value.casefold()
        if any(fragment in folded for fragment in value_fragments):
            raise RuntimeError(f"forbidden outbound value in {label}")
        if re.match(r"^(?:[a-z]:[\\/]|/(?:home|scr|tmp|var)/)", value, re.I):
            raise RuntimeError(f"absolute path is forbidden in {label}")


def build_config(
    analysis: dict[str, Any],
    manifest: dict[str, Any],
    governance: dict[str, Any],
    *,
    submission_bindings_sha256: str,
    training_result_sha256: str,
) -> dict[str, Any]:
    return {
        "dataset": str(governance["dataset"]),
        "runtime_dataset_id": str(analysis["dataset"]),
        "seed": int(analysis["seed"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "analysis_schema": analysis["schema_version"],
        "analysis_protocol_sha256": analysis["analysis_protocol_sha256"],
        "analysis_canonical_sha256": analysis["canonical_sha256"],
        "analysis_manifest_canonical_sha256": manifest["canonical_sha256"],
        "submission_bindings_sha256": submission_bindings_sha256,
        "training_result_sha256": training_result_sha256,
        "performance_protocol": {
            "view": "official",
            "arm": "joint_cap3000",
            "checkpoint_axis": "checkpoint index after each class-incremental task",
            "task_axis": "task identifier in the fixed training protocol",
            "overall_accuracy": "correct predictions divided by all official-test rows at the checkpoint",
            "average_task_accuracy": "mean of current task accuracies for all tasks seen at the checkpoint",
            "average_forgetting": "mean over prior tasks of max earlier task accuracy minus current task accuracy; zero at checkpoint 0",
        },
        "data_governance": {
            "source_url": governance["source_url"],
            "source_revision": governance["source_revision"],
            "license": governance["license"],
            "license_url": governance["license_url"],
            "attribution": governance["attribution"],
            "citation_bibtex": governance["citation_bibtex"],
            "derivative_description": governance["derivative_description"],
            "visibility_assumption": governance["visibility_assumption"],
            "outbound_mode": governance["outbound_policy"]["mode"],
        },
    }


def load_training_result(
    path: Path,
    submission_bindings: Path,
    *,
    expected_dataset: str,
    expected_seed: int,
) -> tuple[dict[str, Any], str]:
    bindings = json.loads(submission_bindings.resolve().read_text(encoding="utf-8"))
    record = bindings.get("files", {}).get("training_result")
    if not isinstance(record, dict):
        raise RuntimeError("submission bindings lack training_result")
    resolved = path.resolve()
    if resolved != Path(str(record.get("path"))).resolve():
        raise RuntimeError("training result path differs from the bound path")
    actual = sha256_file(resolved)
    if actual != record.get("sha256"):
        raise RuntimeError("training result SHA-256 differs from the bound hash")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        value.get("dataset") != expected_dataset
        or int(value.get("seed", -1)) != expected_seed
    ):
        raise RuntimeError("training result identity differs from validated analysis")
    return value, actual


def _unit_interval(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RuntimeError(f"{label} must be finite and within [0, 1]")
    return number


def task_accuracy_matrix(result: dict[str, Any]) -> list[list[float | None]]:
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RuntimeError("training result lacks checkpoints")
    size = len(checkpoints)
    rebuilt: list[list[float | None]] = []
    for expected_checkpoint, checkpoint in enumerate(checkpoints):
        if int(checkpoint.get("checkpoint", -1)) != expected_checkpoint:
            raise RuntimeError("training checkpoints are not contiguous")
        metrics = checkpoint["views"]["official"]["arms"]["joint_cap3000"]
        task_accuracy = metrics.get("task_accuracy")
        if not isinstance(task_accuracy, dict):
            raise RuntimeError("checkpoint lacks task_accuracy")
        current = {
            int(key): _unit_interval(
                value, f"task_accuracy[{expected_checkpoint}][{key}]"
            )
            for key, value in task_accuracy.items()
        }
        if sorted(current) != list(range(expected_checkpoint + 1)):
            raise RuntimeError("task accuracy keys do not match the checkpoint axis")
        rebuilt.append(
            [current.get(task) if task <= expected_checkpoint else None for task in range(size)]
        )

    reported = result["summary"]["views"]["official"]["joint_cap3000"].get(
        "task_accuracy_matrix"
    )
    if not isinstance(reported, list) or len(reported) != size:
        raise RuntimeError("reported task accuracy matrix has the wrong shape")
    for checkpoint, (expected_row, reported_row) in enumerate(zip(rebuilt, reported)):
        if not isinstance(reported_row, list) or len(reported_row) != size:
            raise RuntimeError("reported task accuracy matrix has the wrong shape")
        for task, (expected, actual) in enumerate(zip(expected_row, reported_row)):
            if expected is None:
                if actual is not None:
                    raise RuntimeError("reported task accuracy matrix is not triangular")
            else:
                checked = _unit_interval(
                    actual, f"reported_task_accuracy[{checkpoint}][{task}]"
                )
                if abs(checked - expected) > 1e-12:
                    raise RuntimeError(
                        "reported task accuracy matrix differs from checkpoint data"
                    )
    return rebuilt


def performance_rows(result: dict[str, Any]) -> list[dict[str, float | int]]:
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RuntimeError("training result lacks checkpoints")
    matrix = task_accuracy_matrix(result)
    task_history: list[dict[int, float]] = []
    output: list[dict[str, float | int]] = []
    for expected_checkpoint, checkpoint in enumerate(checkpoints):
        if int(checkpoint.get("checkpoint", -1)) != expected_checkpoint:
            raise RuntimeError("training checkpoints are not contiguous")
        metrics = checkpoint["views"]["official"]["arms"]["joint_cap3000"]
        current = {
            task: float(value)
            for task, value in enumerate(matrix[expected_checkpoint])
            if value is not None
        }
        forgetting = []
        for task in range(expected_checkpoint):
            earlier = [row[task] for row in task_history if task in row]
            forgetting.append(max(earlier) - current[task])
        output.append({
            "checkpoint/index": expected_checkpoint,
            "performance/overall_accuracy": _unit_interval(
                metrics["accuracy"], f"accuracy[{expected_checkpoint}]"
            ),
            "performance/average_task_accuracy": sum(current.values()) / len(current),
            "performance/average_forgetting": (
                sum(forgetting) / len(forgetting) if forgetting else 0.0
            ),
            "performance/macro_f1": _unit_interval(
                metrics["macro_f1"], f"macro_f1[{expected_checkpoint}]"
            ),
            "performance/balanced_accuracy": _unit_interval(
                metrics["balanced_accuracy"],
                f"balanced_accuracy[{expected_checkpoint}]",
            ),
        })
        task_history.append(current)
    final = result["summary"]["views"]["official"]["joint_cap3000"]
    checks = {
        "average_task_accuracy": output[-1]["performance/average_task_accuracy"],
        "average_forgetting": output[-1]["performance/average_forgetting"],
        "final_overall_accuracy": output[-1]["performance/overall_accuracy"],
        "final_macro_f1": output[-1]["performance/macro_f1"],
        "final_balanced_accuracy": output[-1]["performance/balanced_accuracy"],
    }
    for field, actual in checks.items():
        if abs(float(final[field]) - float(actual)) > 1e-12:
            raise RuntimeError(f"training summary mismatch: {field}")
    return output


def combined_history_rows(
    analysis: dict[str, Any], result: dict[str, Any]
) -> list[dict[str, Any]]:
    performance = {int(row["checkpoint/index"]): row for row in performance_rows(result)}
    explanation = {
        int(row["explanation/checkpoint"]): row for row in history_rows(analysis)
    }
    output = []
    for checkpoint in sorted(performance):
        row = dict(performance[checkpoint])
        drift = explanation.get(checkpoint)
        row.update({
            "explanation/mean_top15_jaccard": (
                drift["explanation/mean_top15_jaccard"] if drift else None
            ),
            "explanation/silent_drift_events_cumulative": (
                drift["explanation/silent_drift_events_cumulative"] if drift else 0
            ),
            "explanation/eligible_transitions_cumulative": (
                drift["explanation/eligible_transitions_cumulative"] if drift else 0
            ),
            "explanation/silent_drift_rate_cumulative": (
                drift["explanation/silent_drift_rate_cumulative"] if drift else 0.0
            ),
        })
        output.append(row)
    return output


def history_rows(analysis: dict[str, Any]) -> list[dict[str, float | int]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in analysis["transition_rows"]:
        grouped[int(row["to_checkpoint"])].append(row)
    cumulative_events = 0
    cumulative_eligible = 0
    output = []
    for checkpoint, rows in sorted(grouped.items()):
        cumulative_events += sum(bool(row["primary_event"]) for row in rows)
        cumulative_eligible += sum(bool(row["primary_eligible"]) for row in rows)
        output.append(
            {
                "explanation/checkpoint": checkpoint,
                "explanation/mean_top15_jaccard": sum(
                    float(row["jaccard_top15"]) for row in rows
                )
                / len(rows),
                "explanation/silent_drift_events_cumulative": cumulative_events,
                "explanation/eligible_transitions_cumulative": cumulative_eligible,
                "explanation/silent_drift_rate_cumulative": (
                    cumulative_events / cumulative_eligible
                    if cumulative_eligible
                    else 0.0
                ),
            }
        )
    return output


def table_specs(
    analysis: dict[str, Any], training_result: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    checkpoint_columns = [
        "dataset", "seed", "checkpoint", "class_id", "class_name",
        "probe_rows", "background_rows", "recall",
        "jaccard_top15_from_previous", "silent_drift_primary_event",
        "rationale_mass", "random_null_95", "mass_margin", "admitted",
        "etg_state", "etg_action", "top15_features",
    ]
    checkpoint_data = [
        [
            row.get(column)
            if column != "top15_features"
            else ", ".join(row.get("top15_features", []))
            for column in checkpoint_columns
        ]
        for row in analysis["checkpoint_rows"]
    ]
    transition_columns = [
        "dataset", "seed", "from_checkpoint", "to_checkpoint", "class_id",
        "class_name", "recall_before", "recall_after", "delta_recall",
        "jaccard_top5", "jaccard_top10", "jaccard_top15", "jaccard_top20",
        "cosine_similarity", "kendall_tau_b",
        "prediction_flip_rate_all_common_probe_rows", "primary_eligible",
        "primary_event",
    ]
    transition_data = [
        [row.get(column) for column in transition_columns]
        for row in analysis["transition_rows"]
    ]
    sensitivity_columns = [
        "k", "jaccard_threshold", "allowed_recall_drop", "events",
        "eligible_transitions", "rate", "is_primary",
    ]
    sensitivity_data = [
        [row.get(column) for column in sensitivity_columns]
        for row in analysis["threshold_sensitivity"]
    ]
    ledger_columns = [
        "checkpoint", "class_id", "class_name", "state_before", "state_after",
        "action", "rationale_mass", "random_null_95", "mass_margin",
        "mass_admitted", "certified_reference_jaccard", "drift_checkpoint",
    ]
    ledger_data = [
        [row.get(column) for column in ledger_columns]
        for row in analysis["etg_ledger"]
    ]
    action_counts = Counter(row["action"] for row in analysis["etg_ledger"])
    specs = {
        "results/shap_checkpoint_metrics": {
            "columns": checkpoint_columns, "data": checkpoint_data,
        },
        "results/explanation_drift_transitions": {
            "columns": transition_columns, "data": transition_data,
        },
        "results/drift_threshold_sensitivity": {
            "columns": sensitivity_columns, "data": sensitivity_data,
        },
        "results/etg_ledger": {"columns": ledger_columns, "data": ledger_data},
        "results/etg_action_counts": {
            "columns": ["action", "count"], "data": sorted(action_counts.items()),
        },
        "results/etg_final_states": {
            "columns": ["state", "class_count"],
            "data": sorted(analysis["etg_summary"]["final_states"].items()),
        },
    }
    if training_result is not None:
        primary = performance_rows(training_result)
        performance_columns = [
            "checkpoint", "overall_accuracy", "average_task_accuracy",
            "average_forgetting", "macro_f1", "balanced_accuracy",
        ]
        specs["results/primary_performance_by_checkpoint"] = {
            "columns": performance_columns,
            "data": [[
                row["checkpoint/index"],
                row["performance/overall_accuracy"],
                row["performance/average_task_accuracy"],
                row["performance/average_forgetting"],
                row["performance/macro_f1"],
                row["performance/balanced_accuracy"],
            ] for row in primary],
        }
        matrix = task_accuracy_matrix(training_result)
        specs["results/primary_task_accuracy_matrix"] = {
            "columns": ["checkpoint", "task", "task_accuracy"],
            "data": [
                [checkpoint, task, float(accuracy)]
                for checkpoint, values in enumerate(matrix)
                for task, accuracy in enumerate(values)
                if accuracy is not None
            ],
        }
    return specs


def summary_payload(
    analysis: dict[str, Any],
    manifest: dict[str, Any],
    training_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary = analysis["primary_silent_explanation_drift"]
    payload: dict[str, Any] = {
        "governance/shap_status": "computed_formal_v2",
        "governance/etg_status": "computed_strict_etg_v2",
        "drift/unit": "class_by_adjacent_checkpoint_transition",
        "reproducibility/analysis_canonical_sha256": analysis["canonical_sha256"],
        "reproducibility/manifest_canonical_sha256": manifest["canonical_sha256"],
    }
    for field in (
        "k", "jaccard_threshold", "allowed_recall_drop", "events",
        "eligible_transitions", "rate",
    ):
        payload[f"drift/primary_{field}"] = primary[field]
    for field, value in analysis["etg_summary"].items():
        if field != "final_states":
            payload[f"etg/{field}"] = value
    if training_result is not None:
        final = performance_rows(training_result)[-1]
        for key, value in final.items():
            if key != "checkpoint/index":
                payload[f"performance/final_{key.split('/', 1)[1]}"] = value
    return payload


def validate_outbound_payload(
    governance: dict[str, Any],
    *,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    tables: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    policy = governance["outbound_policy"]
    if set(config) != set(policy["allowed_config_keys"]):
        raise RuntimeError("W&B config keys do not match the governance allowlist")
    if set(config["data_governance"]) != set(policy["allowed_governance_config_keys"]):
        raise RuntimeError("W&B governance config keys do not match the allowlist")
    if set(config["performance_protocol"]) != set(policy["allowed_performance_protocol_keys"]):
        raise RuntimeError("W&B performance protocol keys do not match the allowlist")
    allowed_history = set(policy["allowed_history_keys"])
    if any(set(row) != allowed_history for row in history):
        raise RuntimeError("W&B history keys do not match the governance allowlist")
    allowed_tables = policy["allowed_tables"]
    if set(tables) != set(allowed_tables):
        raise RuntimeError("W&B table names do not match the governance allowlist")
    for name, spec in tables.items():
        if list(spec["columns"]) != list(allowed_tables[name]):
            raise RuntimeError(f"W&B table columns do not match the allowlist: {name}")
    prefixes = tuple(str(item) for item in policy["allowed_summary_prefixes"])
    if any(not str(key).startswith(prefixes) for key in summary):
        raise RuntimeError("W&B summary keys do not match the governance allowlist")
    outbound = {"config": config, "history": history, "tables": tables, "summary": summary}
    _reject_forbidden(outbound, policy, "wandb_payload")


def table_payloads(
    wandb: Any,
    analysis: dict[str, Any],
    governance: dict[str, Any] | None = None,
    training_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    specs = table_specs(analysis, training_result)
    if governance is not None:
        allowed = governance["outbound_policy"]["allowed_tables"]
        if set(specs) != set(allowed):
            raise RuntimeError("W&B table names do not match the governance allowlist")
        for name, spec in specs.items():
            if list(spec["columns"]) != list(allowed[name]):
                raise RuntimeError(f"W&B table columns do not match the allowlist: {name}")
    return {
        name: wandb.Table(columns=spec["columns"], data=spec["data"])
        for name, spec in specs.items()
    }


def publish(args: argparse.Namespace) -> str:
    os.environ["WANDB_ERROR_REPORTING"] = "false"
    import wandb

    analysis, manifest = load_analysis(args.analysis_dir)
    runtime_dataset_id = str(analysis["dataset"])
    seed = int(analysis["seed"])
    destination = f"{args.entity}/{args.project}" if args.entity else args.project
    governance = load_governance(
        args.governance,
        runtime_dataset_id=runtime_dataset_id,
        destination=destination,
    )
    training_result, training_result_sha256 = load_training_result(
        args.training_result,
        args.submission_bindings,
        expected_dataset=runtime_dataset_id,
        expected_seed=seed,
    )
    if str(wandb.__version__) != governance["wandb_version"]:
        raise RuntimeError(
            f"W&B version mismatch: {wandb.__version__} != {governance['wandb_version']}"
        )
    config = build_config(
        analysis,
        manifest,
        governance,
        submission_bindings_sha256=sha256_file(args.submission_bindings),
        training_result_sha256=training_result_sha256,
    )
    history = combined_history_rows(analysis, training_result)
    tables = table_specs(analysis, training_result)
    summary = summary_payload(analysis, manifest, training_result)
    validate_outbound_payload(
        governance, config=config, history=history, tables=tables, summary=summary
    )
    canonical_dataset = str(governance["dataset"])
    settings = wandb.Settings(**wandb_settings_kwargs())
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group or f"formal-v2-{canonical_dataset}",
        name=f"{args.run_name_prefix}-{canonical_dataset}-seed{seed}",
        job_type="explanation_etg",
        tags=["formal-v2", "shap", "explanation-drift", "etg", canonical_dataset],
        config=config,
        settings=settings,
        reinit="finish_previous",
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    try:
        for row in history:
            run.log(row, step=int(row["checkpoint/index"]))
        run.log(table_payloads(wandb, analysis, governance, training_result))
        for key, value in summary.items():
            run.summary[key] = value
        return str(getattr(run, "url", ""))
    finally:
        run.finish()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--analysis-dir", type=Path, required=True)
    value.add_argument("--governance", type=Path, required=True)
    value.add_argument("--submission-bindings", type=Path, required=True)
    value.add_argument("--training-result", type=Path, required=True)
    value.add_argument("--project", required=True)
    value.add_argument("--entity")
    value.add_argument("--group")
    value.add_argument("--run-name-prefix", default="formal-v2")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    url = publish(args)
    print(json.dumps({"status": "published", "url": url}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
