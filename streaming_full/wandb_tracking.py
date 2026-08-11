"""Optional W&B tracking for auditable streaming OFRA runs.

The tracker consumes structured run events and validated result JSON. It never
implements or invents SHAP, explanation drift, or ETG metrics. Those remain
separate analyses until a versioned explanation method is present.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Sequence


SUMMARY_FIELDS = (
    "average_task_accuracy",
    "average_forgetting",
    "final_overall_accuracy",
    "final_macro_f1",
    "final_balanced_accuracy",
    "final_benign_false_positive_rate",
    "final_attack_detection_recall",
)

CHECKPOINT_FIELD_NAMES = {
    "average_task_accuracy": "average_task_accuracy",
    "average_forgetting": "average_forgetting",
    "final_overall_accuracy": "accuracy",
    "final_macro_f1": "macro_f1",
    "final_balanced_accuracy": "balanced_accuracy",
    "final_benign_false_positive_rate": "benign_false_positive_rate",
    "final_attack_detection_recall": "attack_detection_recall",
}

ALLOWED_VIEWS = {"official"}
ALLOWED_ARMS = {"head_only", "joint_cap3000"}


def _official_arms(views: object) -> dict[str, object]:
    if not isinstance(views, dict) or "official" not in views:
        raise ValueError("W&B payload lacks the official view")
    arms = views["official"]
    if not isinstance(arms, dict):
        raise ValueError("W&B official view lacks arm records")
    selected = {name: value for name, value in arms.items() if name in ALLOWED_ARMS}
    if not selected:
        raise ValueError("W&B payload lacks both approved primary arms")
    return selected


def flatten_checkpoint_event(event: dict[str, object]) -> dict[str, float | int]:
    checkpoint = int(event["checkpoint"])
    seen_classes = event.get("seen_classes")
    if not isinstance(seen_classes, list):
        raise ValueError("checkpoint event lacks seen_classes")
    views = event.get("views")
    if not isinstance(views, dict):
        raise ValueError("checkpoint event lacks views")
    payload: dict[str, float | int] = {
        "checkpoint/index": checkpoint,
    }
    for view_name, arms in (("official", _official_arms(views)),):
        for arm_name, metrics in arms.items():
            if not isinstance(metrics, dict):
                raise ValueError("checkpoint arm record must contain metrics")
            for source_name, display_name in CHECKPOINT_FIELD_NAMES.items():
                if source_name in metrics:
                    value = float(metrics[source_name])
                    payload[f"{view_name}/{display_name}/{arm_name}"] = value
    return payload


def summary_table_rows(result: dict[str, object]) -> list[list[object]]:
    summary = result.get("summary")
    views = summary.get("views") if isinstance(summary, dict) else None
    if not isinstance(views, dict):
        raise ValueError("result summary lacks views")
    rows: list[list[object]] = []
    for view_name, arms in (("official", _official_arms(views)),):
        for arm_name, metrics in arms.items():
            if not isinstance(metrics, dict):
                raise ValueError("result summary arm lacks metrics")
            rows.append(
                [view_name, arm_name]
                + [metrics.get(field) for field in SUMMARY_FIELDS]
            )
    return rows


def task_accuracy_table_rows(result: dict[str, object]) -> list[list[object]]:
    summary = result["summary"]
    rows: list[list[object]] = []
    for view_name, arms in (("official", _official_arms(summary["views"])),):
        for arm_name, metrics in arms.items():
            for checkpoint, values in enumerate(metrics["task_accuracy_matrix"]):
                for task, accuracy in enumerate(values):
                    if accuracy is not None:
                        rows.append(
                            [view_name, arm_name, checkpoint, task, float(accuracy)]
                        )
    return rows


def confusion_table_rows(result: dict[str, object]) -> list[list[object]]:
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("result lacks checkpoints")
    final_checkpoint = checkpoints[-1]
    seen_classes = final_checkpoint["seen_classes"]
    rows: list[list[object]] = []
    views = final_checkpoint["views"]
    if not isinstance(views, dict) or "official" not in views:
        raise ValueError("W&B confusion payload lacks official")
    for view_name, view in (("official", views["official"]),):
        for arm_name, metrics in _official_arms({"official": view["arms"]}).items():
            matrix = metrics["confusion_matrix"]
            class_names = {
                int(record["class_id"]): record["class_name"]
                for record in metrics["per_class"]
            }
            for true_index, true_class in enumerate(seen_classes):
                row_total = int(sum(matrix[true_index]))
                for predicted_index, predicted_class in enumerate(seen_classes):
                    count = int(matrix[true_index][predicted_index])
                    rows.append(
                        [
                            view_name,
                            arm_name,
                            int(true_class),
                            class_names[int(true_class)],
                            int(predicted_class),
                            class_names[int(predicted_class)],
                            count,
                            float(count / row_total) if row_total else 0.0,
                        ]
                    )
    return rows


def monitoring_table_rows(result: dict[str, object]) -> list[list[object]]:
    monitoring = result.get("monitoring")
    if not isinstance(monitoring, dict) or not monitoring.get("enabled"):
        return []
    rows = []
    for checkpoint in monitoring.get("checkpoints", []):
        rows.append(
            [
                checkpoint.get("checkpoint"),
                checkpoint.get("checkpoint_manifest_file_sha256"),
                checkpoint.get("checkpoint_manifest_canonical_sha256"),
            ]
        )
    return rows


class WandbSeedTracker:
    """Create one clear W&B run per dataset seed."""

    def __init__(
        self,
        *,
        project: str,
        output_dir: str | Path,
        entity: str | None = None,
        group: str | None = None,
        run_name_prefix: str | None = None,
        tags: Sequence[str] = (),
        wandb_module: Any | None = None,
    ) -> None:
        if not project.strip() or not entity or not entity.strip():
            raise ValueError("W&B entity and project must be explicit")
        if wandb_module is None:
            import wandb as imported_wandb

            wandb_module = imported_wandb
        self.wandb = wandb_module
        self.project = project
        self.entity = entity
        self.group = group
        self.run_name_prefix = run_name_prefix
        self.tags = list(tags)
        self.output_dir = Path(output_dir).resolve()
        self.run = None
        self.seed: int | None = None

    def _protocol(self) -> dict[str, object]:
        return json.loads((self.output_dir / "protocol.json").read_text(encoding="utf-8"))

    def _start(self, event: dict[str, object]) -> None:
        if self.run is not None:
            raise RuntimeError("W&B received a new seed before the prior seed finished")
        protocol = self._protocol()
        seed = int(event["seed"])
        dataset = str(event["dataset"])
        config = protocol.get("config", {})
        encoder = config.get("encoder_type", "model") if isinstance(config, dict) else "model"
        width = config.get("d_model", "na") if isinstance(config, dict) else "na"
        depth = config.get("n_layers", "na") if isinstance(config, dict) else "na"
        prefix = self.run_name_prefix or "ofra"
        name = f"{prefix}-{dataset}-{encoder}{width}x{depth}-seed{seed}"
        stable_run_id = hashlib.sha256(
            f"{event['protocol_sha256']}:{seed}".encode("utf-8")
        ).hexdigest()[:24]
        init_kwargs: dict[str, object] = {
            "project": self.project,
            "name": name,
            "group": self.group or str(event["protocol_sha256"])[:12],
            "tags": self.tags,
            "job_type": "training",
            "config": {
                "dataset": dataset,
                "seed": seed,
                "protocol_sha256": event["protocol_sha256"],
                "run_config": config,
                "explanation_method": None,
                "shap_status": "not_computed",
                "etg_status": "not_computed",
            },
            "dir": str(self.output_dir),
            "reinit": True,
            "id": stable_run_id,
            "resume": "allow",
            "settings": self.wandb.Settings(
                x_disable_meta=True,
                x_disable_stats=True,
                x_disable_machine_info=True,
                x_save_requirements=False,
                disable_code=True,
                disable_git=True,
                disable_job_creation=True,
                save_code=False,
                console="off",
            ),
        }
        init_kwargs["entity"] = self.entity
        self.run = self.wandb.init(**init_kwargs)
        self.seed = seed
        self.run.summary["governance/shap_status"] = "not_computed"
        self.run.summary["governance/etg_status"] = "not_computed"
        self.run.summary["governance/explanation_method"] = None

    def _log_checkpoint(self, event: dict[str, object]) -> None:
        if self.run is None or self.seed != int(event["seed"]):
            raise RuntimeError("W&B checkpoint event has no matching active seed run")
        self.run.log(
            flatten_checkpoint_event(event),
            step=int(event["checkpoint"]),
        )

    def _log_result(self, seed: int) -> None:
        if self.run is None:
            raise RuntimeError("W&B cannot log a result without an active run")
        result_path = self.output_dir / f"result_seed_{seed}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for view_name, arms in (("official", _official_arms(result["summary"]["views"])),):
            for arm_name, metrics in arms.items():
                for field in SUMMARY_FIELDS:
                    if field in metrics:
                        self.run.summary[f"final/{view_name}/{field}/{arm_name}"] = float(
                            metrics[field]
                        )
        self.run.summary["reproducibility/deterministic_result_sha256"] = result[
            "deterministic_result_sha256"
        ]
        self.run.log(
            {
                "results/summary_table": self.wandb.Table(
                    columns=["view", "arm", *SUMMARY_FIELDS],
                    data=summary_table_rows(result),
                ),
                "results/task_accuracy_matrix": self.wandb.Table(
                    columns=["view", "arm", "checkpoint", "task", "accuracy"],
                    data=task_accuracy_table_rows(result),
                ),
                "results/final_confusion_matrix": self.wandb.Table(
                    columns=[
                        "view",
                        "arm",
                        "true_class_id",
                        "true_class_name",
                        "predicted_class_id",
                        "predicted_class_name",
                        "count",
                        "row_normalized",
                    ],
                    data=confusion_table_rows(result),
                ),
                "monitoring/checkpoint_manifests": self.wandb.Table(
                    columns=[
                        "checkpoint",
                        "manifest_file_sha256",
                        "manifest_canonical_sha256",
                    ],
                    data=monitoring_table_rows(result),
                ),
            }
        )

    def _finish(self, *, exit_code: int) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
        self.run = None
        self.seed = None

    def __call__(self, event: dict[str, object]) -> None:
        event_name = event.get("event")
        if event_name == "start":
            self._start(event)
        elif event_name == "checkpoint":
            self._log_checkpoint(event)
        elif event_name in {"end", "skip"}:
            if self.run is None:
                self._start(event)
            self._log_result(int(event["seed"]))
            self._finish(exit_code=0)
        elif event_name == "fail":
            if self.run is not None:
                self.run.summary["failure/status"] = "failed"
            self._finish(exit_code=1)
        elif event_name == "pause":
            self._finish(exit_code=0)

    def close(self) -> None:
        self._finish(exit_code=1 if self.run is not None else 0)
