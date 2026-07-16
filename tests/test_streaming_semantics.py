from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from streaming_full.data import (
    APPLICATION_CLASSIFICATION,
    CLASS_INCREMENTAL,
    GENERIC_METRIC_PROFILE,
    INTRUSION_DETECTION,
    NIDS_METRIC_PROFILE,
    dataset_logical_fingerprints,
    load_manifest,
    sha256_file,
)
from streaming_full.validation import (
    ARMS,
    RunConfig,
    _aggregate,
    _metrics_from_confusion,
    _validate_evaluation_instrumentation,
    run_manifest,
)
from streaming_full.summarize import summarize_campaign


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_shard(path: Path, values: np.ndarray) -> dict[str, object]:
    np.save(path, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return {
        "path": path.name,
        "rows": int(len(values)),
        "sha256": sha256_file(path),
    }


def _make_manifest(
    root: Path,
    *,
    semantic_fields: dict[str, object] | None,
) -> Path:
    classes = []
    for class_id, name in enumerate(("Web", "Video", "Messaging")):
        center = np.asarray([class_id * 2.0, -class_id], dtype=np.float32)
        train = np.vstack([center + [offset, -offset] for offset in (0.0, 0.1, 0.2, 0.3)])
        test = np.vstack([center + [offset, offset] for offset in (0.05, 0.15)])
        classes.append(
            {
                "id": class_id,
                "name": name,
                "train": [
                    _write_shard(root / f"train_{class_id}.npy", train)
                ],
                "test": [_write_shard(root / f"test_{class_id}.npy", test)],
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": "semantic-test",
        "feature_dim": 2,
        "normal_class_id": 0,
        "tasks": [[0, 1], [2]],
        "classes": classes,
    }
    if semantic_fields is not None:
        manifest.update(semantic_fields)
    path = root / "streaming_manifest.json"
    _write_json(path, manifest)
    return path


def _config() -> RunConfig:
    return RunConfig(
        pretrain_epochs=0,
        epochs_per_task=1,
        batch_size=4,
        eval_batch_size=2,
        shuffle_block_rows=2,
        learning_rate=1e-3,
        d_model=4,
        n_layers=1,
        lora_rank=2,
        lora_alpha=2.0,
        negative_ratio=1,
        exemplar_capacity=1,
        exemplar_candidate_capacity=3,
        router_cap_samples=3,
        router_lambda_quantile=0.3,
        router_max_centroids=2,
        device="cpu",
        deterministic=True,
        verify_shard_hashes=True,
        verbose=False,
    )


class StreamingSemanticTests(unittest.TestCase):
    def test_legacy_manifest_defaults_to_fail_closed_nids_semantics(self):
        with tempfile.TemporaryDirectory(prefix="ofra_legacy_semantics_") as temporary:
            manifest = load_manifest(
                _make_manifest(Path(temporary), semantic_fields=None)
            )
        self.assertEqual(manifest.problem_type, INTRUSION_DETECTION)
        self.assertEqual(manifest.metric_profile, NIDS_METRIC_PROFILE)
        self.assertEqual(manifest.task_semantics, CLASS_INCREMENTAL)
        self.assertEqual(manifest.normal_class_id, 0)

    def test_semantic_fields_reject_partial_mismatched_or_non_null_generic_modes(self):
        invalid_records = (
            {"problem_type": APPLICATION_CLASSIFICATION},
            {
                "problem_type": APPLICATION_CLASSIFICATION,
                "metric_profile": NIDS_METRIC_PROFILE,
                "normal_class_id": None,
            },
            {
                "problem_type": APPLICATION_CLASSIFICATION,
                "metric_profile": GENERIC_METRIC_PROFILE,
                "normal_class_id": 0,
            },
            {
                "problem_type": INTRUSION_DETECTION,
                "metric_profile": GENERIC_METRIC_PROFILE,
                "normal_class_id": 0,
            },
        )
        for index, semantic_fields in enumerate(invalid_records):
            with self.subTest(index=index), tempfile.TemporaryDirectory(
                prefix="ofra_invalid_semantics_"
            ) as temporary:
                path = _make_manifest(Path(temporary), semantic_fields=semantic_fields)
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_generic_metrics_and_aggregation_omit_nids_only_fields(self):
        matrix = np.asarray([[2, 1, 0], [0, 2, 0], [1, 0, 2]], dtype=np.int64)
        metrics = _metrics_from_confusion(
            matrix,
            [0, 1, 2],
            {0: "Web", 1: "Video", 2: "Messaging"},
            None,
            metric_profile=GENERIC_METRIC_PROFILE,
        )
        self.assertNotIn("binary_detection", metrics)
        self.assertAlmostEqual(metrics["accuracy"], 6 / 8)
        self.assertIn("macro_f1", metrics)
        self.assertIn("balanced_accuracy", metrics)

        with tempfile.TemporaryDirectory(prefix="ofra_generic_run_") as temporary:
            root = Path(temporary)
            manifest_path = _make_manifest(
                root,
                semantic_fields={
                    "problem_type": APPLICATION_CLASSIFICATION,
                    "metric_profile": GENERIC_METRIC_PROFILE,
                    "task_semantics": CLASS_INCREMENTAL,
                    "normal_class_id": None,
                },
            )
            manifest = load_manifest(manifest_path)
            fingerprints = dataset_logical_fingerprints(manifest)
            nids_identity = replace(
                manifest,
                problem_type=INTRUSION_DETECTION,
                metric_profile=NIDS_METRIC_PROFILE,
                normal_class_id=0,
            )
            self.assertNotEqual(
                fingerprints["train_logical_sha256"],
                dataset_logical_fingerprints(nids_identity)["train_logical_sha256"],
            )

            output = run_manifest(
                manifest_path,
                seeds=[7],
                output_dir=root / "result",
                config=_config(),
            )
            protocol = output["protocol"]
            result = output["results"][0]
            self.assertEqual(protocol["problem_type"], APPLICATION_CLASSIFICATION)
            self.assertEqual(protocol["metric_profile"], GENERIC_METRIC_PROFILE)
            self.assertEqual(protocol["task_semantics"], CLASS_INCREMENTAL)
            self.assertEqual(protocol["normal_class"]["class_id"], None)
            self.assertEqual(
                protocol["evaluation"]["training_input_logical_sha256"],
                fingerprints["train_logical_sha256"],
            )
            for checkpoint in result["checkpoints"]:
                for arm_metrics in checkpoint["views"]["official"]["arms"].values():
                    self.assertNotIn("binary_detection", arm_metrics)
            tampered = copy.deepcopy(result)
            tampered["checkpoints"][0]["views"]["official"]["arms"][
                "head_only"
            ]["binary_detection"] = {}
            with self.assertRaisesRegex(RuntimeError, "NIDS-only"):
                _validate_evaluation_instrumentation(
                    tampered,
                    root / "tampered.json",
                    expected_metric_profile=GENERIC_METRIC_PROFILE,
                    expected_normal_class_id=None,
                )
            for arm in ARMS:
                summary = result["summary"]["views"]["official"][arm]
                self.assertNotIn("final_benign_false_positive_rate", summary)
                self.assertNotIn("final_attack_detection_recall", summary)
            aggregate = _aggregate([result])
            for arm in ARMS:
                self.assertNotIn(
                    "final_benign_false_positive_rate",
                    aggregate["views"]["official"][arm],
                )

            campaign_root = root / "campaign"
            run_manifest(
                manifest_path,
                seeds=[0, 1, 2, 3, 4],
                output_dir=campaign_root / "semantic-test",
                config=_config(),
            )
            campaign = summarize_campaign(
                campaign_root,
                campaign_root / "paired.json",
                expected_seeds=[0, 1, 2, 3, 4],
                expected_datasets=["semantic-test"],
            )
            dataset_report = campaign["datasets"]["semantic-test"]
            self.assertEqual(
                dataset_report["metric_profile"], GENERIC_METRIC_PROFILE
            )
            self.assertNotIn(
                "final_benign_false_positive_rate",
                dataset_report["metric_registry"],
            )
            comparison = dataset_report["views"]["official"][
                "primary_arm_comparisons"
            ]["joint_uncapped_minus_joint_cap3000"]
            self.assertNotIn(
                "final_attack_detection_recall", comparison["metrics"]
            )


if __name__ == "__main__":
    unittest.main()
