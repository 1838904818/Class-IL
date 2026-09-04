"""Pure scope validation for hash-bound CPU-reconstruction attribution evidence."""
from __future__ import annotations

from typing import Any


ATTRIBUTION_TARGET_SCOPE = (
    "hash-bound checkpoint reconstructed on CPU and evaluated on each fixed "
    "true-class probe batch"
)
ATTRIBUTION_LIMITATION = (
    "the attribution result describes the deterministic CPU/class-batch "
    "reconstruction and is not a numerical explanation of archived GPU score "
    "arrays or an alternative full-probe batch shape"
)
PREDICTIVE_METRIC_KEYS = (
    "average_task_accuracy",
    "average_forgetting",
    "final_overall_accuracy",
    "final_macro_f1",
    "final_balanced_accuracy",
)


def attribution_scope_contract() -> dict[str, Any]:
    """Return the exact publication scope for every downstream artifact."""
    return {
        "target": ATTRIBUTION_TARGET_SCOPE,
        "cross_device_equivalence_claimed": False,
        "batch_partition_equivalence_claimed": False,
        "limitation": ATTRIBUTION_LIMITATION,
    }


def validate_attribution_scope(value: object, *, context: str) -> None:
    """Reject scope widening or either numerical-equivalence claim."""
    if value != attribution_scope_contract():
        raise RuntimeError(f"{context} attribution scope mismatch or equivalence claim")


def extract_predictive_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the fixed official/joint_cap3000 performance fields."""
    try:
        block = result["summary"]["views"]["official"]["joint_cap3000"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("missing official/joint_cap3000 predictive metrics") from exc
    metrics: dict[str, Any] = {
        "evaluation_view": "official",
        "arm": "joint_cap3000",
    }
    for key in PREDICTIVE_METRIC_KEYS:
        try:
            value = float(block[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid official/joint_cap3000 metric: {key}") from exc
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"official/joint_cap3000 metric outside [0,1]: {key}")
        metrics[key] = value
    return metrics


def validate_declared_scope(
    result: dict[str, Any],
    protocol: dict[str, Any],
    expected_analysis: dict[str, Any],
    *,
    seed: int,
    dataset: str,
) -> None:
    """Fail closed if training and explanation evidence do not share one scope."""
    if int(result.get("seed", -1)) != seed:
        raise RuntimeError("training result/declared seed mismatch")
    protocol_seeds = [int(value) for value in protocol.get("seeds", [])]
    if seed not in protocol_seeds:
        raise RuntimeError("declared seed is absent from the training protocol")
    if expected_analysis.get("dataset") != dataset:
        raise RuntimeError("expected-gradients analysis dataset mismatch")
    if int(expected_analysis.get("seed", -1)) != seed:
        raise RuntimeError("expected-gradients analysis seed mismatch")
    rows = expected_analysis.get("checkpoint_rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("expected-gradients checkpoint rows are missing")
    if any(int(row.get("seed", -1)) != seed for row in rows):
        raise RuntimeError("expected-gradients checkpoint row seed mismatch")
    validate_attribution_scope(
        expected_analysis.get("attribution_scope"),
        context="expected-gradients analysis",
    )
    cross_device = expected_analysis.get("cpu_reload_vs_saved_gpu_numerical_audit")
    if not isinstance(cross_device, dict) or cross_device.get("equivalence_claimed") is not False:
        raise RuntimeError("expected-gradients cross-device equivalence claim changed")
    partition = expected_analysis.get("batch_partition_numerical_sensitivity")
    if not isinstance(partition, dict) or partition.get("equivalence_claimed") is not False:
        raise RuntimeError("expected-gradients batch-partition equivalence claim changed")
