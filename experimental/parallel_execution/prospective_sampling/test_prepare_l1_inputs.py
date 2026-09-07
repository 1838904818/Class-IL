"""Small synthetic-only cross-line fit arrays -> frozen encoder -> L1 inputs."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

import sampling_builder as b
import prepare_l1_inputs as bridge
import test_sampling_builder as fixtures


BASE = Path(__file__).resolve().parent
FAIR = BASE.parent / "fair_comparison"
sys.path.insert(0, str(FAIR))
import materialize_embeddings as exporter
import l1_runner as runner
import torch


class CrossLineTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.BuilderTests(methodName="runTest")
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        f = self.fx
        source = f.json("catalog.json", f.source_catalog())
        self.contracts = f.root / "contracts"
        b.adapt_replayids(source, self.contracts, expected_sha256=b.sha256_file(source))
        self.test_directory = f.root / "official-test-verified"
        b.verify_official_test(self.contracts / "official_test.json", f.source, self.test_directory, chunk_rows=3)
        self.prefixes = {}
        for arm in ("P", "O"):
            policy = dict(f.policy, arm=arm, offline_normal_cap=20 if arm == "O" else None)
            policy_path = f.json(arm + "-policy.json", policy)
            prefix = []
            for increment in range(2):
                output = f.root / f"{arm}-{increment}"
                b.derive(self.contracts / f"increment_{increment:02d}.json", f.source, output, policy_path,
                         previous=prefix[-1] if prefix else None, chunk_rows=3)
                prefix.append(output)
            self.prefixes[arm] = prefix
        first = b.load_increment(self.prefixes["P"][0])
        # This deliberately untrained random checkpoint is only an interface
        # fixture. No optimizer, pretraining or real evidence is fabricated.
        torch.manual_seed(17)
        self.model = exporter.MLPEncoder(3, 4, 1)
        cohort = np.concatenate([c["x"] for c in b.iter_partition(self.prefixes["P"][0], "transform_fit", 3)])
        mean, scale = cohort.astype(np.float64).mean(0), cohort.astype(np.float64).std(0)
        state = {f"encoder_{i}": value.detach().numpy() for i, value in enumerate(self.model.state_dict().values())}
        self.state_path = f.root / "fixture-state.npz"
        np.savez(self.state_path, **state, mean=mean, scale=scale)
        self.metadata = {"checkpoint": 0, "dataset": "synthetic", "seen_classes": [0, 1], "feature_dim": 3,
                         "architecture": {"encoder_type": "mlp", "d_model": 4, "n_layers": 1},
                         "state_schema": {"encoder": {name: f"encoder_{i}" for i, name in enumerate(self.model.state_dict())},
                                          "normalization": {"mean": "mean", "scale": "scale"}},
                         "inference_state_sha256": b.sha256_file(self.state_path),
                         "common_transform_fit_sha256": first["common_transform_fit_sha256"],
                         "normalization_fit_row_ids_sha256": first["collections"]["transform_fit"]["row_ids"]["sha256"],
                         "encoder_training_row_ids_sha256": first["collections"]["transform_fit"]["row_ids"]["sha256"],
                         "evidence_kind": "synthetic", "synthetic_untrained_fixture": True}
        self.metadata_path = f.json("fixture-metadata.json", self.metadata)
        self.config_path = FAIR / "pilot_config.json"

    def prepare(self, name="raw-pair", **changes):
        options = dict(p_prefix=self.prefixes["P"], o_prefix=self.prefixes["O"], test_directory=self.test_directory,
                       source_root=self.fx.source, checkpoint_metadata=self.metadata_path, checkpoint_state=self.state_path,
                       config_path=self.config_path, output=self.fx.root / name, evidence_kind="synthetic", chunk_rows=3)
        options.update(changes)
        return bridge.prepare_pair(**options)

    def test_actual_two_arm_materializer_and_l1_loader_preserve_fit_identity(self):
        result = self.prepare()
        self.assertEqual(result["status"], "RAW_EXPORT_PAIR_COMPLETE")
        self.assertFalse(result["model_training_performed"])
        self.assertTrue(result["offline_container_not_online_revelation"])
        states = []
        test_hashes = []
        for arm in ("P", "O"):
            raw_root = self.fx.root / "raw-pair" / arm
            spec = bridge.verify_export_input(raw_root / "export-input.json")
            expected_ids = [rid for d in self.prefixes[arm] for c in b.iter_partition(d, "fit", 2) for rid in c["row_ids"]]
            forbidden = {rid for d in self.prefixes[arm] for part in ("calibration", "omitted")
                         for c in b.iter_partition(d, part, 2) for rid in c["row_ids"]}
            with b.mapped(raw_root / spec["raw_splits"]["train"]["row_ids"]["path"]) as ids:
                self.assertEqual(ids.tolist(), expected_ids)
                self.assertFalse(set(ids.tolist()) & forbidden)
            before = {r["path"]: b.sha256_file(self.fx.source / r["path"])
                      for r in bridge.verified_test(self.test_directory)[0]["source_shards"]}
            output_manifest = exporter.export(raw_root / "export-input.json", self.fx.root / (arm + "-embeddings"),
                                              device_name="cpu", batch_size=3)
            manifest, arrays, _ = runner.load_inputs(output_manifest)
            try:
                self.assertEqual(arrays["train"]["row_ids"].tolist(), expected_ids)
                np.testing.assert_array_equal(arrays["train"]["group_ids"], arrays["train"]["row_ids"])
                self.assertEqual(len(arrays["test"]["labels"]), 8)
                self.assertEqual(manifest["tasks"], [[0, 1], [2, 3]])
                states.append(manifest["encoder_binding"]["state_sha256"])
                test_hashes.append(manifest["splits"]["test"])
                with b.mapped(raw_root / "train-raw_features.npy") as raw, np.load(self.state_path) as checkpoint:
                    normalized = torch.from_numpy(((raw.astype(np.float64) - checkpoint["mean"]) / checkpoint["scale"]).astype(np.float32))
                    with torch.inference_mode():
                        expected = self.model(normalized).numpy()
                    np.testing.assert_allclose(arrays["train"]["embeddings"], expected, rtol=1e-6, atol=1e-6)
            finally:
                for split in arrays.values():
                    for array in split.values():
                        b.close_array(array)
            self.assertEqual(before, {path: b.sha256_file(self.fx.source / path) for path in before})
        self.assertEqual(states[0], states[1])
        self.assertEqual(test_hashes[0], test_hashes[1])

    def test_missing_or_changed_common_cohort_checkpoint_is_rejected(self):
        for field in ("common_transform_fit_sha256", "normalization_fit_row_ids_sha256", "encoder_training_row_ids_sha256"):
            bad = copy.deepcopy(self.metadata)
            bad.pop(field)
            path = self.fx.json("bad-" + field + ".json", bad)
            with self.assertRaises(b.Blocked):
                self.prepare("bad-" + field, checkpoint_metadata=path)
        with self.assertRaises(b.Blocked):
            self.prepare(evidence_kind="real")
        with self.assertRaises(b.Blocked):
            self.prepare(p_prefix=self.prefixes["P"][:1], o_prefix=self.prefixes["O"][:1])
        config = b.read_json(self.config_path)
        bad_config = self.fx.json("invalid-config.json", dict(config, learning_rate=-1))
        with self.assertRaises(b.Blocked):
            self.prepare(config_path=bad_config)
        original = bridge.validate_prefix
        def no_support(*args, **kwargs):
            result = original(*args, **kwargs)
            result[0]["classes"][1]["fit_rows"] = 0
            return result
        with patch.object(bridge, "validate_prefix", side_effect=no_support), self.assertRaisesRegex(b.Blocked, "nonempty fitting support"):
            self.prepare("empty-support")

    def test_actual_common_task0_training_bridge_export_and_l1_fit(self):
        import pretrain_task0 as pretrain
        config = b.read_json(FAIR / "task0_pilot_config.json")
        config.update(batch_size=64, checkpoint_every_steps=1,
                      architecture={"encoder_type": "mlp", "d_model": 4, "n_layers": 1})
        pretrain_config = self.fx.json("tiny-pretrain-config.json", config)
        task0 = self.prefixes["P"][0]
        task0_sha = b.sha256_file(task0 / "manifest.json")
        trained = self.fx.root / "actually-trained-common-task0"
        result = pretrain.run(task0, task0_sha, pretrain_config, trained, evidence_kind="synthetic", device_name="cpu")
        self.assertEqual(result["status"], "COMPLETE")
        cost = b.read_json(trained / "pretrain-cost.json")
        self.assertEqual(cost["status"], "MEASURED")
        self.assertGreater(cost["measurements"]["optimizer_steps"], 0)
        self.assertEqual(cost["measurements"]["raw_row_presentations"], 18)
        trained_metadata = b.read_json(trained / "checkpoint.json")
        self.assertNotIn("synthetic_untrained_fixture", trained_metadata)
        l1_config = b.read_json(self.config_path)
        l1_config.update(epochs=1, batch_size=64, eval_batch_size=64, rank=2, negative_ratio=1, exemplar_capacity=2)
        l1_config_path = self.fx.json("tiny-l1-config.json", l1_config)
        self.prepare("trained-raw-pair", checkpoint_metadata=trained / "checkpoint.json",
                     checkpoint_state=trained / "inference-state.npz", pretrain_cost=trained / "pretrain-cost.json",
                     config_path=l1_config_path)
        for arm in ("P", "O"):
            raw = self.fx.root / "trained-raw-pair" / arm / "export-input.json"
            bridge.verify_export_input(raw)
            embedded = exporter.export(raw, self.fx.root / (arm + "-trained-embeddings"), device_name="cpu", batch_size=8)
            output = self.fx.root / (arm + "-tiny-l1-fit")
            fitted = runner.run(embedded, output, device_name="cpu")
            self.assertEqual(fitted["status"], "COMPLETE")
            self.assertEqual(fitted["evidence_kind"], "synthetic")
            audited = runner.audit(output, embedded)
            self.assertEqual(audited["status"], "AUDIT_PASS")
            # One unit is recorded for each arm/task pair (2 arms x 2 tasks).
            self.assertEqual(audited["training_units"], 4)

    def test_pair_failure_cannot_be_consumed_and_cannot_overwrite(self):
        original = bridge.materialize_arm
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise b.Blocked("synthetic injected second-arm failure")
            return original(*args, **kwargs)
        with patch.object(bridge, "materialize_arm", side_effect=fail_second), self.assertRaises(b.Blocked):
            self.prepare("failed-pair")
        self.assertTrue((self.fx.root / "failed-pair" / "FAILED.json").is_file())
        self.assertFalse((self.fx.root / "failed-pair" / "COMPLETE.json").exists())
        with self.assertRaises((b.Blocked, FileNotFoundError)):
            bridge.verify_export_input(self.fx.root / "failed-pair" / "P" / "export-input.json")
        with self.assertRaises(b.Blocked):
            self.prepare("failed-pair")
        calls.clear()
        def replace_bound_prefix(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(True)
            if len(calls) == 1:
                directory = self.prefixes["P"][0]
                manifest = b.read_json(directory / "manifest.json")
                manifest["binding"]["input_sha256"] = "f" * 64
                (directory / "manifest.json").write_bytes(b.canonical(manifest) + b"\n")
                marker = b.read_json(directory / "COMPLETE.json")
                marker["manifest_sha256"] = b.sha256_file(directory / "manifest.json")
                marker["files"]["manifest.json"] = marker["manifest_sha256"]
                (directory / "COMPLETE.json").write_bytes(b.canonical(marker) + b"\n")
                b.load_increment(directory)  # Internally consistent, but not the originally bound input.
            return result
        with patch.object(bridge, "materialize_arm", side_effect=replace_bound_prefix), self.assertRaisesRegex(b.Blocked, "externally bound"):
            self.prepare("replaced-prefix")
        self.assertFalse((self.fx.root / "replaced-prefix" / "COMPLETE.json").exists())

    def test_completed_raw_closure_rejects_unbound_file_and_changed_arrays(self):
        self.prepare()
        root = self.fx.root / "raw-pair" / "P"
        self.fx.json("raw-pair/P/extra.json", {"unbound": True})
        with self.assertRaises(b.Blocked):
            bridge.verify_export_input(root / "export-input.json")


if __name__ == "__main__":
    unittest.main()
