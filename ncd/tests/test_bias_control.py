"""Synthetic controller fixtures only; these are not Section 23 results."""
from contextlib import redirect_stderr, redirect_stdout
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ncd import run_bias_control as control


def config():
    return {"base_config": "ncd/configs/base.json", "datasets": control.DATASETS,
            "best_methods": control.BEST_METHODS, "development_seeds": [5, 6],
            "reporting_seeds": [1, 2, 3, 4, 42], "epochs_per_task": 8,
            "min_steps_per_task": 2000, "batch_size": 256,
            "candidates": {"wa": [{}], "bic": [{"steps": 100}, {"steps": 300}],
                           "ssil": [{"tkd_weight": .5}, {"tkd_weight": 1.0}],
                           "cbf": [{"epochs": 10}, {"epochs": 30}]}}


def setup_root(root):
    for name in control.SOURCE_PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source\n", encoding="utf-8")
    path = root / "ncd/configs/section23.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    control.save(path, config())
    control.save(root / "ncd/configs/base.json", {"lambda_files": {"default": "ncd/configs/lambda.json"}})
    control.save(root / "ncd/configs/lambda.json", {"selection_seeds": [5, 6]})
    return path


def write_child(out, identity, bound, score=.4):
    directory = out / control.cell_path(identity)
    directory.mkdir(parents=True, exist_ok=True)
    actual_identity = dict(identity)
    actual_identity["base_method"] = actual_identity.pop("base")
    actual_identity.update({key: bound[key] for key in ("git_commit", "source_hashes", "config_sha256", "config_input_hashes")})
    actual_identity["tasks"] = [[0], [1]]
    role = ("training_holdout_not_official_ID_test" if identity["phase"] == "development"
            else "official_ID_test_not_used_for_selection")
    actual_identity["evaluation_role"] = role
    if identity["phase"] == "formal":
        actual_identity["frozen_selection"] = {"sha256": control.digest(out / "selection.json")}
    identity_hash = hashlib.sha256(json.dumps(actual_identity, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    checkpoints = []
    for task in range(2):
        weight = directory / "checkpoints" / f"checkpoint_{task:03d}.pt"
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b"Synthetic bytes, not a trained model")
        row = {"task_index": task, "checkpoint_file": weight.relative_to(directory).as_posix(),
               "checkpoint_sha256": control.digest(weight), "identity_hash": identity_hash,
               "evaluation_role": role, "id": {"macro_f1": score}}
        control.save(weight.with_suffix(".json"), row)
        checkpoints.append(row)
    path = directory / "result.json"
    control.save(path, {"completed": True, "identity": actual_identity, "checkpoints": checkpoints})
    return path


class GridTests(unittest.TestCase):
    def test_atomic_save_retries_transient_file_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'result.json'
            original = Path.replace
            attempts = []
            def replace(path, target):
                attempts.append(target)
                if len(attempts) <= 2:
                    raise PermissionError('synthetic transient read lock')
                return original(path, target)
            with patch.object(Path, 'replace', replace), patch.object(control.time, 'sleep') as sleep:
                control.save(destination, {'valid': True})
            self.assertEqual(control.load(destination), {'valid': True})
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(len(attempts), 3)

    def test_atomic_save_stops_on_persistent_file_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(Path, 'replace', side_effect=PermissionError('persistent')) as replace:
                with patch.object(control.time, 'sleep') as sleep:
                    with self.assertRaises(PermissionError):
                        control.save(Path(temporary) / 'result.json', {'valid': True})
            self.assertEqual(replace.call_count, 41)
            self.assertEqual(sleep.call_count, 40)

    def test_source_paths_match_runner_without_importing_training(self):
        tree = ast.parse((control.ROOT / "ncd/run_bias_correction.py").read_text(encoding="utf-8"))
        paths = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "SOURCE_PATHS" for target in node.targets))
        self.assertEqual(control.SOURCE_PATHS, paths)

    def test_development_grid_is_exact_and_disjoint(self):
        cells = list(control.development_cells(config()))
        self.assertEqual(len(cells), 84)
        self.assertEqual(len({tuple(cell.values()) for cell in cells}), 84)
        self.assertEqual({cell["seed"] for cell in cells}, {5, 6})
        self.assertEqual({cell["phase"] for cell in cells}, {"development"})
        self.assertTrue(all(not cell["smoke"] for cell in cells))

    def test_smoke_is_24_development_candidate_zero_cells(self):
        cells = list(control.development_cells(config(), smoke=True))
        self.assertEqual(len(cells), 24)
        self.assertEqual({cell["seed"] for cell in cells}, {5})
        self.assertEqual({cell["candidate"] for cell in cells}, {0})
        self.assertTrue(all(cell["smoke"] and cell["phase"] == "development" for cell in cells))

    def test_formal_grid_uses_selection_and_report_seeds(self):
        rows = []
        for dataset in control.DATASETS:
            for base in ("replay", control.BEST_METHODS[dataset]):
                for correction in control.CORRECTIONS:
                    rows.append({"dataset": dataset, "base": base, "correction": correction,
                                 "selected_candidate": 0 if correction == "wa" else 1})
        cells = list(control.formal_cells(config(), {"selections": rows}))
        self.assertEqual(len(cells), 120)
        self.assertEqual({cell["seed"] for cell in cells}, {1, 2, 3, 4, 42})
        self.assertTrue(all(cell["candidate"] == (0 if cell["correction"] == "wa" else 1) for cell in cells))

    def test_config_rejects_seed_leak_and_unbounded_grid(self):
        for key, value in (("development_seeds", [5, 6, 42]), ("reporting_seeds", [1, 2, 5]),
                           ("best_methods", {**control.BEST_METHODS, "NSL-KDD": "replay"}),
                           ("datasets", control.DATASETS + ["CSE-CIC-IDS2018"])):
            changed = copy.deepcopy(config())
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.validate_config(changed)
        changed = copy.deepcopy(config())
        changed["candidates"]["wa"].append({})
        with self.assertRaises(ValueError):
            control.validate_config(changed)
        for key, value in (("epochs_per_task", 7), ("min_steps_per_task", 500), ("batch_size", 384)):
            changed = copy.deepcopy(config())
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.validate_config(changed)

    def test_command_uses_relative_paths_and_passes_smoke(self):
        identity = control.cell_identity("NSL-KDD", "replay", "wa", 0, 5, "development", True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = control.child_command(identity, root / "config.json", root / "result", root)
        self.assertEqual(command[command.index("--config") + 1], "config.json")
        self.assertEqual(command[command.index("--output-dir") + 1], "result")
        self.assertIn("--smoke", command)

    def test_formal_command_requires_selection(self):
        identity = control.cell_identity("NSL-KDD", "replay", "wa", 0, 1, "formal")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "frozen development selection"):
                control.child_command(identity, root / "config.json", root / "result", root)
            command = control.child_command(identity, root / "config.json", root / "result", root, root / "selection.json")
            self.assertEqual(command[command.index("--selection") + 1], "selection.json")


class FilesystemTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = setup_root(self.root)
        self.env = patch.dict(os.environ, {"NCD_SOURCE_COMMIT": "a" * 40})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.bound = control.provenance(self.config_path, self.root)
        self.out = self.root / "results"
        self.out.mkdir()
        self.identity = control.cell_identity("NSL-KDD", "replay", "wa", 0, 5, "development")

    def test_complete_child_is_validated_without_unpickling(self):
        path = write_child(self.out, self.identity, self.bound, .42)
        proof = control.validate_child(path, self.identity, self.bound)
        self.assertEqual(proof["macro_f1"], .42)
        self.assertEqual(proof["checkpoint_count"], 2)

    def test_identity_source_config_and_role_mismatches_fail(self):
        cases = [("seed", 42), ("base_method", "derpp"), ("git_commit", "b" * 40),
                 ("config_sha256", "b" * 64), ("source_hashes", {}),
                 ("evaluation_role", "official_ID_test_not_used_for_selection")]
        for key, value in cases:
            path = write_child(self.out, self.identity, self.bound)
            result = control.load(path)
            result["identity"][key] = value
            control.save(path, result)
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.validate_child(path, self.identity, self.bound)

    def test_missing_or_changed_checkpoint_and_sidecar_fail(self):
        path = write_child(self.out, self.identity, self.bound)
        weight = path.parent / "checkpoints/checkpoint_001.pt"
        weight.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint file"):
            control.validate_child(path, self.identity, self.bound)
        path = write_child(self.out, self.identity, self.bound)
        control.save(weight.with_suffix(".json"), {})
        with self.assertRaisesRegex(ValueError, "sidecar"):
            control.validate_child(path, self.identity, self.bound)

    def test_incomplete_can_resume_but_never_select(self):
        path = write_child(self.out, self.identity, self.bound)
        result = control.load(path)
        result["completed"] = False
        control.save(path, result)
        self.assertIsNone(control.validate_child(path, self.identity, self.bound, require_complete=False))
        with self.assertRaisesRegex(ValueError, "complete result"):
            control.validate_child(path, self.identity, self.bound)

    def test_missing_tasks_or_invalid_metric_fail(self):
        path = write_child(self.out, self.identity, self.bound)
        result = control.load(path)
        result["checkpoints"].pop()
        control.save(path, result)
        with self.assertRaisesRegex(ValueError, "checkpoint count"):
            control.validate_child(path, self.identity, self.bound)
        path = write_child(self.out, self.identity, self.bound, 1.1)
        with self.assertRaisesRegex(ValueError, "finite proportion"):
            control.validate_child(path, self.identity, self.bound)

    def test_source_drift_rejects_resume(self):
        (self.root / control.SOURCE_PATHS[0]).write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source/config changed"):
            control.provenance(self.config_path, self.root, self.bound)

    def test_base_config_and_lambda_drift_are_bound(self):
        for relative in ("ncd/configs/base.json", "ncd/configs/lambda.json"):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(path=relative), self.assertRaisesRegex(ValueError, "source/config changed"):
                control.provenance(self.config_path, self.root, self.bound)
            path.write_bytes(original)

    def test_resume_keeps_commit_without_reading_new_head(self):
        with patch.dict(os.environ, {"NCD_SOURCE_COMMIT": ""}), patch.object(control, "source_commit", side_effect=AssertionError("must not read HEAD")):
            self.assertEqual(control.provenance(self.config_path, self.root, self.bound), self.bound)
        with patch.dict(os.environ, {"NCD_SOURCE_COMMIT": "b" * 40}), self.assertRaisesRegex(ValueError, "original run"):
            control.provenance(self.config_path, self.root, self.bound)

    def test_duplicate_controller_lock_fails(self):
        path = self.out / ".execution.lock"
        with control.execution_lock(path):
            with self.assertRaises(OSError):
                with control.execution_lock(path):
                    pass
        with control.execution_lock(path):
            pass

    def populate_development(self):
        for identity in control.development_cells(config()):
            # Candidate 1 wins BiC/CBF; SS-IL ties and must choose candidate 0.
            score = .4 + (.1 * identity["candidate"] if identity["correction"] != "ssil" else 0)
            write_child(self.out, identity, self.bound, score)

    def test_selection_uses_only_dev_results_with_tie_break(self):
        self.populate_development()
        # Poison formal paths. If selection opens them, JSON parsing will fail.
        formal = self.out / "formal"
        formal.mkdir()
        (formal / "result.json").write_text("not JSON", encoding="utf-8")
        selection = control.freeze_selection(config(), self.out, self.bound)
        self.assertEqual(len(selection["selections"]), 24)
        for row in selection["selections"]:
            self.assertEqual(row["selected_candidate"], 1 if row["correction"] in {"bic", "cbf"} else 0)
            self.assertTrue(all(e["seed"] in {5, 6} for c in row["candidates"] for e in c["development_results"]))
        self.assertEqual(control.freeze_selection(config(), self.out, self.bound), selection)

    def test_frozen_selection_or_development_change_fails(self):
        self.populate_development()
        control.freeze_selection(config(), self.out, self.bound)
        write_child(self.out, self.identity, self.bound, .9)
        with self.assertRaisesRegex(ValueError, "frozen selection"):
            control.freeze_selection(config(), self.out, self.bound)

    def test_nonzero_child_stops_and_preserves_attempt_logs(self):
        class FailedProcess:
            pid = 999999

            def wait(self):
                return 7

        with patch.object(control.subprocess, "Popen", return_value=FailedProcess()) as popen, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = control.run(self.config_path, self.out, smoke=True, root=self.root)
        self.assertEqual(code, 1)
        self.assertEqual(popen.call_count, 1)
        record = control.load(self.out / "result.json")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["current_cell"]["exit_code"], 7)
        for key in ("stdout", "stderr"):
            self.assertTrue((self.out / record["current_cell"][key]).is_file())

    def test_smoke_end_to_end_and_resume_do_not_launch_formal(self):
        def fake_child(identity, config_path, out, bound, record, environment, root):
            self.assertEqual(environment["NCD_SOURCE_COMMIT"], "a" * 40)
            path = write_child(out, identity, bound)
            return control.validate_child(path, identity, bound, selection_sha256=record.get("selection_sha256"))

        with patch.object(control, "run_cell", side_effect=fake_child) as runner, redirect_stdout(io.StringIO()):
            code = control.run(self.config_path, self.out, smoke=True, root=self.root)
        self.assertEqual(code, 0)
        self.assertEqual(runner.call_count, 24)
        record = control.load(self.out / "result.json")
        self.assertEqual(record["completed_cells"], {"development": 24, "formal": 0})
        self.assertFalse((self.out / "selection.json").exists())
        with patch.object(control.subprocess, "Popen", side_effect=AssertionError("completed cells must skip")), redirect_stdout(io.StringIO()):
            self.assertEqual(control.run(self.config_path, self.out, smoke=True, root=self.root), 0)
        with self.assertRaisesRegex(ValueError, "different output roots"):
            control.run(self.config_path, self.out, smoke=False, root=self.root)

    def test_formal_end_to_end_waits_for_all_development(self):
        visited = []

        def fake_child(identity, config_path, out, bound, record, environment, root):
            if identity["phase"] == "formal":
                self.assertEqual(record["completed_cells"]["development"], 84)
                self.assertTrue((out / "selection.json").is_file())
            visited.append(identity)
            path = write_child(out, identity, bound, .4 + .1 * identity["candidate"])
            return control.validate_child(path, identity, bound, selection_sha256=record.get("selection_sha256"))

        with patch.object(control, "run_cell", side_effect=fake_child), redirect_stdout(io.StringIO()):
            self.assertEqual(control.run(self.config_path, self.out, root=self.root), 0)
        self.assertEqual(len(visited), 204)
        self.assertEqual([row["phase"] for row in visited], ["development"] * 84 + ["formal"] * 120)
        record = control.load(self.out / "result.json")
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["completed_cells"], {"development": 84, "formal": 120})


if __name__ == "__main__":
    unittest.main()
