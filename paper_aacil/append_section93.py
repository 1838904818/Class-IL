"""Append requested B1 reporting fields from already published measurements."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    path = args.result.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    identity = data["identity"]
    if not data["completed"] or identity["effective_config"]["smoke"]:
        raise ValueError("Completed formal publication required")
    if "section93_reporting" in data:
        raise ValueError("Already appended; do not repeat")
    method = identity["method"]
    if identity["section"] != 81 or method not in ("er_bal_tabt", "ar_tabt"):
        raise ValueError("Registered B1 method required")
    final = data["checkpoints"][-1]
    endpoints = {}
    for alpha in ("0.01", "0.02"):
        point = final["operating_points"][alpha]
        endpoints[alpha] = {"final_attack_detection": point["attack_recall"],
                            "final_normal_fpr": point["normal_fpr"],
                            "attack_detection_definition": point["attack_recall_definition"],
                            "source": "final saved operating point; no new inference or recalibration"}
    epochs = []
    for index, checkpoint in enumerate(data["checkpoints"]):
        if checkpoint["task_index"] != index:
            raise ValueError("Non-contiguous trajectory")
        if method == "er_bal_tabt":
            schedule = checkpoint["schedule"]
            if len(schedule["epoch_history"]) != schedule["epochs_completed"]:
                raise ValueError("Epoch history mismatch")
            record = {"task_index": index, "schedule": {
                k: schedule[k] for k in ("epochs_completed", "best_epoch", "early_stopped", "max_epochs")}}
        else:
            record = {"task_index": index,
                      "gate_schedule": checkpoint["gate_training_schedule"],
                      "family_schedule": checkpoint["family_training_schedule"]}
        epochs.append(record)
    data["section93_reporting"] = {
        "previous_publication_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "endpoints_by_alpha": endpoints, "actual_task_schedules": epochs,
        "measurement_recomputed": False, "statistical_tests_performed": False,
        "scope": "Single-cell descriptive reporting only; S1-S3 await five seeds.",
        "epoch_interpretation": "ER reports completed and restored-best epochs. AR reports actual G/F schedules separately: requested eight epochs is a lower bound with minimum 2000 optimizer steps, not necessarily eight executed epochs. Partial epochs remain explicit.",
    }
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
