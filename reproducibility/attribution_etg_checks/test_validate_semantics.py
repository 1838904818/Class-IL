import copy
import tempfile
from pathlib import Path
import unittest

from validate_semantics import METHODS, THRESHOLDS, canonical, checked_json, rate, validate


def seal(artifact):
    artifact["canonical_sha256"] = canonical({k: v for k, v in artifact.items() if k != "canonical_sha256"})


def fixture(*, declining=False):
    features = {"dataset": "malaya-network-gt", "feature_count": 77,
                "feature_columns": [f"feature_{i}" for i in range(77)],
                "class_order": [f"class_{i}" for i in range(10)]}
    predictive = {"average_task_accuracy": 0.6, "average_forgetting": 0.05,
                  "final_overall_accuracy": 0.5, "final_macro_f1": 0.4, "final_balanced_accuracy": 0.5}
    training = {"seed": 42, "dataset": "malaya-network-gt", "checkpoints": [],
                "summary": {"views": {"official": {"joint_cap3000": predictive}}}}
    rows, transitions = [], []
    for t in range(5):
        recall = 0.9 - t * 0.2 if declining else 0.9
        training["checkpoints"].append({"checkpoint": t, "seen_classes": list(range(2*(t+1))),
            "views": {"official": {"arms": {"joint_cap3000": {"per_class": [
                {"class_id": c, "recall": recall} for c in range(2*(t+1))]}}}}})
        for c in range(2*(t+1)):
            first = t == c//2
            rows.append({"dataset": "malaya-network-gt", "seed": 42, "checkpoint": t,
                "class_id": c, "class_name": f"class_{c}", "probe_rows": 128, "background_rows": 64,
                "recall": recall, "rationale_mass": 0.2, "random_null_95": 0.1, "mass_margin": 0.1,
                "admitted": True, "top15_indices": list(range(15)),
                "top15_features": features["feature_columns"][:15], "etg_state": "CERTIFIED_STABLE",
                "etg_action": "admission_certified" if first else "monitor_no_change"})
        if t:
            before = 0.9 - (t-1)*0.2 if declining else 0.9
            for c in range(2*t):
                transitions.append({"from_checkpoint": t-1, "to_checkpoint": t, "class_id": c,
                    "recall_before": before, "recall_after": recall, "delta_recall": recall-before,
                    "jaccard_top15": 1.0, "primary_eligible": not declining, "primary_event": False})
    artifact = {"schema_version": "ofra_attribution_robustness_v3", "dataset": "malaya-network-gt", "seed": 42,
        "status": "completed_cpu_reconstruction_single_seed_analysis", "score_target": "joint_cap3000 class margin",
        "attribution_scope": {"target": "hash-bound checkpoint reconstructed on CPU and evaluated on each fixed true-class probe batch",
                              "cross_device_equivalence_claimed": False, "batch_partition_equivalence_claimed": False},
        "thresholds": copy.deepcopy(THRESHOLDS), "checkpoint_rows": {}, "transition_rows": {}, "method_summaries": {},
        "predictive_metrics": {"evaluation_view": "official", "arm": "joint_cap3000", **predictive}}
    for method in METHODS:
        artifact["checkpoint_rows"][method] = copy.deepcopy(rows)
        artifact["transition_rows"][method] = copy.deepcopy(transitions)
        artifact["method_summaries"][method] = {"checkpoint_class_rows": 30, "admitted_rows": 30,
            "silent_drift_events": 0, "eligible_transitions": 0 if declining else 20}
    seal(artifact)
    return artifact, training, features


class SemanticTests(unittest.TestCase):
    def reject(self, mutation, message):
        a, t, f = fixture()
        mutation(a)
        seal(a)
        with self.assertRaisesRegex(ValueError, message):
            validate(a, t, f)

    def test_complete_source_registry(self):
        output = validate(*fixture())
        self.assertEqual(output["checkpoint_class_rows_per_method"], 30)
        self.assertEqual(output["transitions_per_method"], 20)

    def test_jointly_missing_class_cannot_hide_in_equal_method_scopes(self):
        self.reject(lambda a: [a["checkpoint_rows"][m].pop(0) for m in METHODS], "coverage")

    def test_jointly_missing_transition(self):
        self.reject(lambda a: [a["transition_rows"][m].pop(0) for m in METHODS], "coverage")

    def test_duplicate_key(self):
        self.reject(lambda a: a["checkpoint_rows"][METHODS[0]].append(a["checkpoint_rows"][METHODS[0]][0]), "Duplicate")

    def test_string_false_is_not_a_boolean(self):
        self.reject(lambda a: a["transition_rows"][METHODS[0]][0].update(primary_event="False"), "Invalid decision")

    def test_event_without_feature_change(self):
        self.reject(lambda a: a["transition_rows"][METHODS[0]][0].update(primary_event=True), "silent drift")

    def test_wrong_eligible_flag(self):
        self.reject(lambda a: a["transition_rows"][METHODS[0]][0].update(primary_eligible=False), "eligibility")

    def test_wrong_recomputed_jaccard(self):
        self.reject(lambda a: a["transition_rows"][METHODS[0]][0].update(jaccard_top15=0.2), "Jaccard")

    def test_source_recall_not_self_declared(self):
        self.reject(lambda a: a["checkpoint_rows"][METHODS[0]][0].update(recall=0.8), "official class recall")

    def test_duplicate_feature(self):
        self.reject(lambda a: a["checkpoint_rows"][METHODS[0]][0].update(top15_indices=[0]*15), "top-15")

    def test_feature_names_bound_to_schema(self):
        self.reject(lambda a: a["checkpoint_rows"][METHODS[0]][0].update(top15_features=["wrong"]*15), "Feature labels")

    def test_thresholds_fixed_not_only_equal_between_seeds(self):
        self.reject(lambda a: a["thresholds"].update(silent_drift_jaccard=0.6), "threshold")

    def test_ledger_states_replayed(self):
        self.reject(lambda a: a["checkpoint_rows"][METHODS[0]][0].update(etg_state="UNEXPLAINABLE"), "ledger")

    def test_nonfinite_mass_rejected(self):
        a, t, f = fixture()
        a["checkpoint_rows"][METHODS[0]][0]["rationale_mass"] = float("nan")
        with self.assertRaises(ValueError):
            validate(a, t, f)

    def test_zero_eligibility_is_na_not_zero(self):
        output = validate(*fixture(declining=True))
        self.assertTrue(all(r["rate"] is None for r in output["method_rates"].values()))
        self.assertTrue(all(r["eligible_transitions"]["rate"] is None for r in output["drift_agreement_denominators"].values()))

    def test_drift_and_next_checkpoint_recertification(self):
        a, t, f = fixture()
        rows = a["checkpoint_rows"][METHODS[0]]
        target = next(r for r in rows if r["checkpoint"] == 1 and r["class_id"] == 0)
        target.update(top15_indices=list(range(15, 30)), top15_features=f["feature_columns"][15:30],
                      etg_state="DRIFTED", etg_action="human_review_escalation")
        next(r for r in rows if r["checkpoint"] == 2 and r["class_id"] == 0).update(etg_action="strict_recertified")
        for row in a["transition_rows"][METHODS[0]]:
            if row["class_id"] == 0 and row["to_checkpoint"] in (1, 2):
                row.update(jaccard_top15=0, primary_event=True)
        a["method_summaries"][METHODS[0]]["silent_drift_events"] = 2
        seal(a)
        self.assertEqual(validate(a, t, f)["method_rates"][METHODS[0]]["events"], 2)

    def test_event_count_cannot_exceed_eligible_count(self):
        with self.assertRaises(ValueError):
            rate(1, 0)

    def test_file_hash_required(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "example.json"
            p.write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                checked_json(p, "0"*64)


if __name__ == "__main__":
    unittest.main()
