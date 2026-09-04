import json
import re
import tempfile
import unittest
from pathlib import Path

from formal_v2_explanation_etg.aggregate_attribution_robustness import canonical_sha256
from formal_v2_explanation_etg.analyze import (
    CONFIRMATION,
    SURROGATE_DISTANCE_SQUARED_FLOOR,
    validate_output,
)
from formal_v2_explanation_etg.preflight_score_fidelity import (
    CONFIRMATION as PREFLIGHT_CONFIRMATION,
)
from formal_v2_explanation_etg.attribution_robustness import (
    PRIMARY_ALLOWED_RECALL_DROP,
    PRIMARY_JACCARD_THRESHOLD,
    validate_completed_output,
)
from formal_v2_explanation_etg.test_aggregate_attribution_robustness import artifact
from streaming_full.data import sha256_file


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def build_expected_output(path: Path, bindings: dict) -> None:
    path.mkdir()
    protocol = {
        "schema_version": "formal_v2_explanation_etg_v2",
        "dataset": "malaya-network-gt",
        "seed": 1,
        "score_target": "g_c(x)=joint_cap3000_c(x)-max_{f!=c}joint_cap3000_f(x)",
        "numerical_fidelity": {
            "gradient_surrogate": {
                "distance_squared_floor": SURROGATE_DISTANCE_SQUARED_FLOOR,
                "scope": "gradient surrogate only",
                "exact_forward_value": "unchanged by straight-through value correction",
                "nonfinite_policy": "fail closed before attribution; never replace non-finite values with zero",
            }
        },
        "attribution": {"nsamples": 64, "cpu_threads": 8},
        "stability": {
            "primary": {"k": 15, "jaccard_threshold": 0.70, "allowed_recall_drop": 0.05}
        },
        "bindings": bindings,
    }
    protocol["analysis_protocol_sha256"] = canonical_sha256(protocol)
    analysis = {
        "schema_version": "formal_v2_explanation_etg_v2",
        "dataset": "malaya-network-gt",
        "seed": 1,
        "analysis_protocol_sha256": protocol["analysis_protocol_sha256"],
        "counts": {
            "class_checkpoint_rows": 1,
            "attribution_gradient_targets_checked": 1,
        },
        "attribution_gradient_fidelity": {
            "targets_checked": 1,
            "nonfinite_count": 0,
            "surrogate_distance_squared_floor": SURROGATE_DISTANCE_SQUARED_FLOOR,
            "scope": "gradient surrogate only; exact forward values are unchanged",
            "rows": [
                {
                    "checkpoint": 0,
                    "class_id": 0,
                    "probe_rows": 1,
                    "feature_dim": 2,
                    "output_max_abs": 1.0,
                    "gradient_max_abs": 0.5,
                    "gradient_nonfinite_count": 0,
                    "surrogate_distance_squared_floor": SURROGATE_DISTANCE_SQUARED_FLOOR,
                    "exact_forward_value_changed": False,
                }
            ],
        },
    }
    analysis["canonical_sha256"] = canonical_sha256(analysis)
    write_json(path / "analysis_protocol.json", protocol)
    write_json(path / "analysis.json", analysis)
    write_json(path / "progress.json", {"status": "complete", "active": None})
    (path / "attributions.npz").write_bytes(b"bound-test-attributions")
    files = {}
    for member in sorted(path.iterdir()):
        files[member.name] = {"sha256": sha256_file(member), "bytes": member.stat().st_size}
    manifest = {
        "schema_version": "formal_v2_explanation_etg_v2",
        "dataset": "malaya-network-gt",
        "seed": 1,
        "analysis_protocol_sha256": protocol["analysis_protocol_sha256"],
        "analysis_canonical_sha256": analysis["canonical_sha256"],
        "input_bindings": bindings,
        "files": files,
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    write_json(path / "analysis_manifest.json", manifest)


def build_robustness_output(path: Path, sources: dict, expected_analysis: Path, expected_arrays: Path) -> None:
    path.mkdir()
    mean_arrays = path / "attribution_robustness_mean_attributions.npz"
    mean_arrays.write_bytes(b"bound-test-mean-attributions")
    value = artifact(1)
    value.update({
        "source_training_job": 388991,
        "source_etg_job": None,
        "methods": {
            "expected_gradients": {
                "source": "completed formal analysis",
                "analysis_file_sha256": sha256_file(expected_analysis),
                "attributions_file_sha256": sha256_file(expected_arrays),
            },
            "integrated_gradients": {
                "status": "skipped_by_prespecified_three_method_protocol"
            },
            "feature_ablation": {"definition": "test"},
            "gradient_x_input": {"definition": "test"},
        },
        "thresholds": {
            "top_k": 15,
            "silent_drift_jaccard": PRIMARY_JACCARD_THRESHOLD,
            "allowed_recall_drop": PRIMARY_ALLOWED_RECALL_DROP,
            "admission": "selected-feature deletion mass exceeds the same fixed random-control 95th percentile used by Expected Gradients",
        },
        "source_bindings": sources,
        "output_bindings": {
            "mean_attributions_file_sha256": sha256_file(mean_arrays)
        },
    })
    value["canonical_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "canonical_sha256"}
    )
    write_json(path / "attribution_robustness.json", value)


class CompletedOutputResumeTests(unittest.TestCase):
    def test_submitted_sbatch_confirmations_match_both_runtime_guards(self):
        operation_root = Path(__file__).resolve().parents[1]
        sbatch = operation_root / "scheduler_contract.sbatch.txt"
        values = re.findall(r"--confirm\s+([^\s\\]+)", sbatch.read_text(encoding="utf-8"))
        self.assertEqual(values, [PREFLIGHT_CONFIRMATION, CONFIRMATION])

    def test_expected_complete_directory_reuses_only_with_exact_current_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "expected-gradients-etg"
            bindings = {"training_result_file_sha256": "a" * 64, "analyzer_file_sha256": "b" * 64}
            build_expected_output(output, bindings)
            record = validate_output(
                output,
                expected_seed=1,
                expected_dataset="malaya-network-gt",
                expected_bindings=bindings,
                expected_shap_nsamples=64,
                expected_cpu_threads=8,
            )
            self.assertEqual(record["seed"], 1)
            with self.assertRaisesRegex(RuntimeError, "current source evidence"):
                validate_output(output, expected_bindings={**bindings, "analyzer_file_sha256": "c" * 64})

    def test_expected_complete_directory_rejects_unbound_member(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "expected-gradients-etg"
            build_expected_output(output, {"source": "bound"})
            (output / "unexpected.bin").write_bytes(b"not signed")
            with self.assertRaisesRegex(RuntimeError, "membership"):
                validate_output(output)

    def test_secondary_interruption_directory_reuses_only_after_full_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result_seed_1.json"
            protocol = root / "protocol.json"
            stream = root / "streaming_manifest.json"
            schema = root / "feature_schema.json"
            expected_analysis = root / "analysis.json"
            expected_arrays = root / "attributions.npz"
            write_json(result, {"summary": {"views": {"official": {"joint_cap3000": {
                "average_task_accuracy": 0.301,
                "average_forgetting": 0.04,
                "final_overall_accuracy": 0.55,
                "final_macro_f1": 0.21,
                "final_balanced_accuracy": 0.24,
            }}}}})
            for index, member in enumerate((protocol, stream, schema, expected_analysis, expected_arrays)):
                member.write_bytes(f"source-{index}".encode("ascii"))
            script = Path(__file__).with_name("attribution_robustness.py")
            sources = {
                "training_result_file_sha256": sha256_file(result),
                "training_protocol_file_sha256": sha256_file(protocol),
                "streaming_manifest_file_sha256": sha256_file(stream),
                "feature_schema_file_sha256": sha256_file(schema),
                "script_sha256": sha256_file(script),
            }
            output = root / "attribution-robustness"
            build_robustness_output(output, sources, expected_analysis, expected_arrays)
            record = validate_completed_output(
                output,
                dataset="malaya-network-gt",
                seed=1,
                source_training_job=388991,
                source_etg_job=None,
                result_path=result,
                training_protocol_path=protocol,
                streaming_manifest_path=stream,
                feature_schema_path=schema,
                expected_analysis_path=expected_analysis,
                expected_attributions_path=expected_arrays,
            )
            self.assertEqual(record["source_training_job"], 388991)
            expected_arrays.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "array binding mismatch"):
                validate_completed_output(
                    output,
                    dataset="malaya-network-gt",
                    seed=1,
                    source_training_job=388991,
                    source_etg_job=None,
                    result_path=result,
                    training_protocol_path=protocol,
                    streaming_manifest_path=stream,
                    feature_schema_path=schema,
                    expected_analysis_path=expected_analysis,
                    expected_attributions_path=expected_arrays,
                )


if __name__ == "__main__":
    unittest.main()
