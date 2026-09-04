import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_validate_semantics import fixture, seal
from validate_aggregate import describe, verify
from validate_semantics import METHODS, SEEDS, canonical

# Golden comparison uses the byte-preserved runtime, never a remote job.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attribution_etg_v9"))
from formal_v2_explanation_etg.aggregate_attribution_robustness import (
    aggregate as original_aggregate, compute_agreement, compute_method_summaries,
)
from formal_v2_explanation_etg.attribution_scope import attribution_scope_contract


def golden():
    """Synthetic arithmetic fixtures, not semantic or model evidence."""
    artifacts = {}
    with tempfile.TemporaryDirectory() as directory:
        paths = {}
        for index, seed in enumerate(SEEDS):
            a = fixture()[0]
            a["seed"] = seed
            a["attribution_scope"] = attribution_scope_contract()
            a["source_bindings"] = {"streaming_manifest_file_sha256": "a"*64,
                "feature_schema_file_sha256": "b"*64, "script_sha256": "c"*64}
            a["predictive_metrics"]["final_macro_f1"] = (index+1)/10
            for method in METHODS:
                for row in a["checkpoint_rows"][method]:
                    row["seed"] = seed
                for i, row in enumerate(a["transition_rows"][method]):
                    row.update(primary_eligible=i < 2*(index+1), primary_event=i == 0)
            a["agreement"] = compute_agreement(a, seed)
            a["method_summaries"] = compute_method_summaries(a, seed)
            seal(a)
            artifacts[seed] = a
            paths[seed] = Path(directory) / f"seed-{seed}.json"
            paths[seed].write_text(json.dumps(a), encoding="utf-8")
        result = original_aggregate(paths)
        hashes = {seed: result["artifact_bindings"][str(seed)]["file_sha256"] for seed in SEEDS}
    return result, artifacts, hashes


class AggregateValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = golden()

    def inputs(self):
        return copy.deepcopy(self.base)

    def reject_aggregate_change(self, change, message):
        aggregate, artifacts, hashes = self.inputs()
        change(aggregate)
        seal(aggregate)
        with self.assertRaisesRegex(ValueError, message):
            verify(aggregate, artifacts, hashes)

    def test_matches_registered_aggregate_arithmetic(self):
        self.assertEqual(verify(*self.inputs())["seeds"], list(SEEDS))

    def test_missing_seed_cannot_emit_completion(self):
        aggregate, artifacts, hashes = self.inputs()
        artifacts.pop(42)
        with self.assertRaisesRegex(ValueError, "Exactly registered"):
            verify(aggregate, artifacts, hashes)

    def test_altered_seed_canonical_hash(self):
        aggregate, artifacts, hashes = self.inputs()
        artifacts[1]["thresholds"]["top_k"] = 10
        with self.assertRaisesRegex(ValueError, "thresholds"):
            verify(aggregate, artifacts, hashes)

    def test_final_metric_mean_recomputed(self):
        self.reject_aggregate_change(lambda a: a["predictive_performance"]["seed_statistics"]["final_macro_f1"].update(mean=0.9), "arithmetic")

    def test_transition_count_cannot_be_seed_n(self):
        self.reject_aggregate_change(lambda a: a["all_method_seed_statistics"]["all_method_admission_agreement"].update(n=100), "value")

    def test_interval_recomputed(self):
        self.reject_aggregate_change(lambda a: a["predictive_performance"]["seed_statistics"]["final_macro_f1"].update(t95_ci_lower=0.2), "arithmetic")

    def test_pooled_counts_recomputed(self):
        self.reject_aggregate_change(lambda a: a["method_conditioned_results"][METHODS[0]]["pooled_counts"].update(eligible_transitions=999), "value")

    def test_seed_mean_not_pooled_event_rate(self):
        def mutation(a):
            block = a["method_conditioned_results"][METHODS[0]]
            pooled = block["pooled_counts"]
            block["silent_drift_rate_seed_summary"]["mean"] = pooled["silent_drift_events"] / pooled["eligible_transitions"]
        self.reject_aggregate_change(mutation, "arithmetic")

    def test_input_hash_binding_required(self):
        aggregate, artifacts, hashes = self.inputs()
        hashes[1] = "0"*64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify(aggregate, artifacts, hashes)

    def test_cross_seed_code_mismatch_rejected(self):
        aggregate, artifacts, hashes = self.inputs()
        artifacts[2]["source_bindings"]["script_sha256"] = "e"*64
        seal(artifacts[2])
        aggregate["artifact_bindings"]["2"]["canonical_sha256"] = artifacts[2]["canonical_sha256"]
        seal(aggregate)
        with self.assertRaisesRegex(ValueError, "input/code"):
            verify(aggregate, artifacts, hashes)

    def test_undefined_rate_is_not_zero_or_dropped(self):
        with self.assertRaisesRegex(ValueError, "Undefined"):
            describe([0.1, 0.2, None, 0.3, 0.4])

    def test_four_values_are_not_five(self):
        with self.assertRaisesRegex(ValueError, "Exactly five"):
            describe([0.1]*4)

    def test_known_mean_and_sample_sd(self):
        result = describe([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(result["mean"], 0.3)
        self.assertAlmostEqual(result["sample_sd"], 0.15811388300841897)

    def test_descriptive_interval_display_clipping(self):
        result = describe([0, 0, 0, 0, 1])
        self.assertEqual(result["t95_ci_lower"], 0.0)

    def test_cli_partial_registry_rejected_before_reading_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "partial.json"
            inputs.write_text(json.dumps([{"seed": 1}, {"seed": 2}]))
            output = root / "should_not_exist.json"
            process = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("validate_aggregate.py")),
                "--inputs", str(inputs), "--aggregate", str(root/"missing.json"),
                "--aggregate-sha256", "0"*64, "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Exactly five input records", process.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
