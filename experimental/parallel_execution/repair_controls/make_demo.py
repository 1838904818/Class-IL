"""Create labeled SYNTHETIC score arrays for software testing, not research wins."""
import argparse
from pathlib import Path

import numpy as np

from score_core import ARMS
from runner import file_hash, save_json


def prospective_policy(groups, rows):
    """Explicit arbitrary demo choices. None is a validated scientific default."""
    return {
        "schema": "score-repair-policy-v1", "scientific_status": "prospective-not-validated", "arms": list(ARMS),
        "trigger": {"min_rows_per_class": 2, "min_error_increase": 0.0, "min_disagreement": 0.0,
                    "min_drift_excess": 0.0, "state_priority_weight": 1.0, "periodic_every": 2,
                    "periodic_phase": 1, "random_seed": 17, "explanation_method": "synthetic-normalized-vector-v1"},
        "fusion": {"router_weight": 0.2, "epsilon": 1e-8, "identity_atol": 1e-12},
        "r1": {"scale_min": 0.5, "scale_max": 2.0, "offset_min": -2.0, "offset_max": 2.0,
               "steps": 24, "learning_rate": 1.0, "identity_l2": 0.01, "max_fit_seconds": 20.0},
        "acceptance": {"min_rows_per_class": 2, "min_target_improvement": 0.01,
                       "max_old_error_increase": 0.05, "max_new_error_increase": 0.05,
                       "max_missed_attack_increase": None, "max_fpr_increase": None},
        "uncertainty": {"min_groups": 2, "trigger_alpha": 0.05, "acceptance_alpha": 0.05, "subsequent_alpha": 0.05,
                        "trigger_family_size": 80, "acceptance_family_size": 60, "subsequent_family_size": 120},
        "budget": {"unique_labels_per_arm": 3 * rows, "max_repairs_per_arm": 1,
                   "fit_steps_per_arm": 24, "acceptance_score_rows_per_arm": 2 * rows,
                   "diagnostic_cells_per_arm": 1000000},
        "semantics": {"kind": "application", "attack_classes": []},
    }


def create_demo(output, seed=7, groups=12):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    classes, old = ["application-a", "application-b", "application-c"], ["application-a", "application-b"]
    manifest = {"schema": "score-repair-bundle-v1", "source_kind": "synthetic-software-test", "increment": 1,
                "classes": classes, "old_classes": old, "partitions": {},
                "synthetic_seed": seed, "claimed_research_improvement": False}
    for role in ("trigger", "fit", "acceptance", "subsequent"):
        n = groups * len(classes)
        labels = classes * groups
        group_ids = [f"demo-{seed}-{role}-group-{g}" for g in range(groups) for _ in classes]
        row_ids = [f"demo-{seed}-{role}-row-{i}" for i in range(n)]
        old_h = rng.normal(-0.5, 1.0, (n, len(old)))
        new_h = rng.normal(-0.5, 1.0, (n, len(classes)))
        router = rng.normal(0, 1.0, (n, len(classes)))
        old_router = router[:, :len(old)].copy()
        for i, label in enumerate(labels):
            k = classes.index(label)
            new_h[i, k] += 1.4
            if k < len(old):
                old_h[i, k] += 1.8
        # Unstructured, signed variation; outcomes are measured, never supplied.
        new_h[:, 0] += rng.normal(-0.3, 0.5, n)
        part = {"role": role, "row_ids": row_ids, "group_ids": group_ids,
                "new_head_logits": new_h.tolist(), "new_router_raw": router.tolist(),
                "old_head_logits": old_h.tolist(), "old_router_raw": old_router.tolist()}
        if role != "subsequent":
            part["labels"] = labels
        if role == "trigger":
            part["explanations"] = {}
            for c in old:
                a = rng.normal(size=(4, n, 5))
                d = a + rng.normal(0, 0.5, a.shape)
                part["explanations"][c] = {"method": "synthetic-normalized-vector-v1", "independent_draws_by_group": True,
                                            "A": a.tolist(), "D": d.tolist(), "source": "random arrays, not SHAP"}
        path = output / f"{role}.json"
        save_json(path, part)
        manifest["partitions"][role] = {"file": path.name, "sha256": file_hash(path), "row_ids": row_ids,
                                        "group_ids": group_ids, "observed_increment": 2 if role == "subsequent" else 1,
                                        "labels_available_increment": None if role == "subsequent" else 1}
        if role == "subsequent":
            label_path = output / "sealed-evaluation-labels.json"
            save_json(label_path, {"row_ids": row_ids, "group_ids": group_ids, "labels": labels, "observed_increment": 2})
            manifest["sealed_labels_sha256"] = file_hash(label_path)
    save_json(output / "manifest.json", manifest)
    save_json(output / "policy.json", prospective_policy(groups, groups * len(classes)))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--groups", type=int, default=12)
    args = parser.parse_args()
    create_demo(args.output, args.seed, args.groups)
    print("Created synthetic inputs only; no repair or performance result was prefilled.")
