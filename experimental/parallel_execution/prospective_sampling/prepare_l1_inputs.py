"""Offline O/P fit-only array bridge to the shared frozen-encoder exporter.

No encoder fitting, model training, resampling, network or submission is used.
The aggregate export container is explicitly offline, not an online emulator.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import shutil
import sqlite3
import sys

import numpy as np

import sampling_builder as b


RAW_KEYS = ("raw_features", "labels", "row_ids", "available_tasks", "group_ids")
CONFIG_KEYS = {"seed", "epochs", "batch_size", "eval_batch_size", "rank", "lora_alpha", "learning_rate",
               "weight_decay", "negative_ratio", "minority_threshold", "focal_alpha", "focal_gamma",
               "exemplar_capacity", "exemplar_selection", "checkpoint_policy", "profile_label"}


def validate_config(config):
    b.keys(config, CONFIG_KEYS, "L1 configuration")
    b.natural(config["seed"])
    for key in ("epochs", "batch_size", "eval_batch_size", "rank", "negative_ratio", "minority_threshold", "exemplar_capacity"):
        b.natural(config[key], True)
    for key, minimum, maximum in (("weight_decay", 0, float("inf")), ("focal_gamma", 0, float("inf")),
                                  ("learning_rate", 1e-15, float("inf")), ("lora_alpha", 1e-15, float("inf")),
                                  ("focal_alpha", 0, 1)):
        value = config[key]
        b.need(type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum, "invalid L1 numeric config")
    b.need(config["exemplar_selection"] == "uniform_without_replacement"
           and config["checkpoint_policy"] == "fixed_last_epoch"
           and config["profile_label"] == "prospective_pilot_defaults_not_optimized_not_preregistered", "unsupported L1 pilot policy")


def verified_test(directory):
    directory = b.no_links(directory)
    marker = b.read_json(directory / "COMPLETE.json")
    b.keys(marker, {"schema_version", "status", "manifest_sha256", "files"}, "test completion")
    b.need(type(marker["schema_version"]) is int and marker["schema_version"] == 1
           and marker["status"] == "OFFICIAL_TEST_VERIFIED", "official test verification incomplete")
    actual = set()
    for path in directory.rglob("*"):
        b.no_links(path)
        if path.is_file() and path.relative_to(directory).as_posix() != "COMPLETE.json":
            actual.add(path.relative_to(directory).as_posix())
    b.need(actual == set(marker["files"]), "test verification closure changed")
    for name, sha in marker["files"].items():
        b.need(b.sha256_file(b.safe_path(directory, name)) == b.digest(sha), "test verification artifact changed")
    b.need(marker["manifest_sha256"] == marker["files"].get("manifest.json"), "test manifest binding changed")
    data = b.read_json(directory / "manifest.json")
    b.need(data["kind"] == "official_test_verification" and data["status"] == "COMPLETE"
           and data["test_sampled"] is False and data["participated_in_train_selection"] is False, "test contract not full-support")
    return data, marker["manifest_sha256"]


def verify_export_input(manifest_path):
    """Verify immutable arm closure and its final O/P transaction commitment."""
    path = b.no_links(manifest_path)
    root = path.parent
    b.need(path.name == "export-input.json", "unexpected raw export manifest name")
    marker = b.read_json(root / "COMPLETE.json")
    b.keys(marker, {"status", "evidence_kind", "manifest_sha256", "files"}, "raw completion")
    b.need(marker["status"] == "RAW_EXPORT_INPUT_COMPLETE", "raw export incomplete")
    actual = set()
    for entry in root.rglob("*"):
        b.no_links(entry)
        if entry.is_file() and entry.relative_to(root).as_posix() != "COMPLETE.json":
            actual.add(entry.relative_to(root).as_posix())
    b.need(actual == set(marker["files"]), "raw export closure changed")
    for name, sha in marker["files"].items():
        b.need(b.sha256_file(b.safe_path(root, name)) == b.digest(sha), "raw export bytes changed")
    b.need(marker["manifest_sha256"] == marker["files"].get(path.name), "raw export manifest binding changed")
    parent = b.read_json(root.parent / "COMPLETE.json")
    b.need(parent["status"] == "RAW_EXPORT_PAIR_COMPLETE" and parent["evidence_kind"] == marker["evidence_kind"],
           "O/P pair transaction incomplete")
    b.need(root.name in {"P", "O"} and parent[root.name + "_export_manifest_sha256"] == marker["manifest_sha256"],
           "raw export not bound to pair")
    return b.read_json(path)


def validate_prefix(directories, arm):
    b.need(directories, "full completed increment prefix required")
    result = []
    previous_sha = None
    for i, directory in enumerate(directories):
        data = b.load_increment(directory)
        b.need(data["increment"] == i and data["policy"]["arm"] == arm, "wrong or incomplete arm prefix")
        b.need(data["binding"]["previous_manifest_sha256"] == previous_sha, "prefix chain mismatch")
        if result:
            for key in ("dataset_id", "namespace", "feature_dim", "feature_dtype", "group_mode", "policy", "common_transform_fit_sha256"):
                b.need(data[key] == result[0][key], "prefix contract changed")
        previous_sha = b.sha256_file(Path(directory) / "manifest.json")
        result.append(data)
    return result


def ref(root, filename):
    return {"path": filename, "sha256": b.sha256_file(root / filename)}


def raw_arrays(root, split, count, feature_dim):
    specs = {"raw_features": ("float32", (count, feature_dim)), "labels": ("int64", (count,)),
             "row_ids": ("S64", (count,)), "available_tasks": ("int64", (count,)), "group_ids": ("S64", (count,))}
    return {key: b.new_array(root / f"{split}-{key}.npy", dtype, shape) for key, (dtype, shape) in specs.items()}


def close_all(arrays):
    for array in arrays.values():
        b.close_array(array)


def insert_identity(db, rows, groups, split):
    try:
        db.executemany("INSERT INTO identities VALUES(?,?,?)", [(bytes(rid).decode("ascii"), bytes(gid).decode("ascii"), split)
                                                               for rid, gid in zip(rows, groups)])
    except sqlite3.IntegrityError as exc:
        raise b.Blocked("duplicate raw row identity across train/test") from exc


def materialize_arm(root, directories, manifests, test, source_root, metadata, state_path, metadata_path,
                    config, evidence_kind, pretrain_cost_path, chunk_rows, expected_prefix_hashes):
    root = b.fresh_directory(root)
    b.write_json(root / "STARTED.json", {"status": "INCOMPLETE", "stage": "offline_fit_only_export_input"})
    first = manifests[0]
    tasks = [[c["class_id"] for c in data["classes"]] for data in manifests]
    introduction = {cid: task for task, classes in enumerate(tasks) for cid in classes}
    db = sqlite3.connect(root / "identity-audit.sqlite")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("CREATE TABLE identities(rid TEXT PRIMARY KEY,gid TEXT,split TEXT)")
    arrays = {}
    try:
        count = sum(data["collections"]["fit"]["rows"] for data in manifests)
        arrays = raw_arrays(root, "train", count, first["feature_dim"])
        written = 0
        for directory, data, expected in zip(directories, manifests, expected_prefix_hashes):
            for chunk in b.iter_partition(directory, "fit", chunk_rows, expected):
                n = len(chunk["labels"])
                target = slice(written, written + n)
                arrays["raw_features"][target] = chunk["x"]
                for key in RAW_KEYS[1:]:
                    arrays[key][target] = chunk[key]
                insert_identity(db, chunk["row_ids"], chunk["group_ids"], "train")
                written += n
        b.need(written == count, "fit-only materialization count mismatch")
        close_all(arrays)
        arrays = {}
        test_count = sum(record["rows"] for record in test["source_shards"])
        arrays = raw_arrays(root, "test", test_count, first["feature_dim"])
        written = 0
        for record in test["source_shards"]:
            cid = record["class_id"]
            b.need(cid in introduction and record["dtype"] == "float32", "test class/dtype mismatch")
            desc = {key: record[key] for key in ("path", "sha256", "rows")}
            path, _ = b.checked_input(source_root, desc, "features", first["feature_dim"], cid, chunk_rows)
            with b.mapped(path) as x:
                for start in range(0, len(x), chunk_rows):
                    stop = min(start + chunk_rows, len(x))
                    target = slice(written, written + stop - start)
                    ids = np.array([b.row_identity(first["namespace"], record["path"], offset) for offset in range(start, stop)], dtype="S64")
                    arrays["raw_features"][target] = x[start:stop]
                    arrays["labels"][target] = cid
                    arrays["available_tasks"][target] = introduction[cid]
                    arrays["row_ids"][target] = ids
                    arrays["group_ids"][target] = ids
                    insert_identity(db, ids, ids, "test")
                    written += stop - start
        b.need(written == test_count == test["total_rows"], "official test support changed")
        close_all(arrays)
        arrays = {}
        b.need(db.execute("SELECT gid FROM identities GROUP BY gid HAVING count(DISTINCT split)>1 LIMIT 1").fetchone() is None,
               "provided train/test group identity collision")
        db.commit()
        db.close()
        db = None
        for source, name in ((metadata_path, "checkpoint-metadata.json"), (state_path, "checkpoint-state.npz")):
            shutil.copyfile(b.no_links(source), root / name)
            b.need(b.sha256_file(source) == b.sha256_file(root / name), "checkpoint source changed during copy")
        pretrain = None
        if pretrain_cost_path is not None:
            shutil.copyfile(b.no_links(pretrain_cost_path), root / "pretrain-cost.json")
            pretrain = ref(root, "pretrain-cost.json")
        raw = {split: {key: ref(root, f"{split}-{key}.npy") for key in RAW_KEYS} for split in ("train", "test")}
        result = {"schema_version": 1, "dataset": first["dataset_id"], "evidence_kind": evidence_kind,
                  "tasks": tasks, "group_mode": first["group_mode"], "config": config, "allow_aggregate_wandb": False,
                  "checkpoint_metadata": ref(root, "checkpoint-metadata.json"), "checkpoint_state": ref(root, "checkpoint-state.npz"),
                  "pretrain_cost_receipt": pretrain, "raw_splits": raw}
        lineage = {"schema_version": 1, "kind": "offline_fit_only_raw_export_lineage", "evidence_kind": evidence_kind,
                   "sampling_arm": first["policy"]["arm"], "common_transform_fit_sha256": first["common_transform_fit_sha256"],
                   "normalization_fit_row_ids_sha256": metadata["normalization_fit_row_ids_sha256"],
                   "input_manifest_sha256": expected_prefix_hashes,
                   "train_selection": "concatenated immutable fit partitions only; no resampling; no calibration or omitted rows",
                   "train_identity_conversion": "none: S64 row/group IDs copied unchanged",
                   "test_identity_rule": "source-row-v1 plus namespace, original relative test path, original row offset",
                   "test_source_shards": test["source_shards"], "source_namespace": first["namespace"],
                   "capture_independent_established": False, "offline_container_not_online_revelation": True,
                   "model_training_performed": False}
        b.write_json(root / "LINEAGE.json", lineage)
        b.write_json(root / "export-input.json", result)
        files = {path.relative_to(root).as_posix(): b.sha256_file(path) for path in root.rglob("*") if path.is_file()}
        b.write_json(root / "COMPLETE.json", {"status": "RAW_EXPORT_INPUT_COMPLETE", "evidence_kind": evidence_kind,
                                              "manifest_sha256": b.sha256_file(root / "export-input.json"), "files": files})
        return result
    except Exception as exc:
        close_all(arrays)
        if db is not None:
            db.close()
        if not (root / "COMPLETE.json").exists():
            b.write_json(root / "FAILED.json", {"status": "INCOMPLETE_FAILED", "error_type": type(exc).__name__})
        raise


def prepare_pair(p_prefix, o_prefix, test_directory, source_root, checkpoint_metadata, checkpoint_state, config_path,
                 output, *, evidence_kind, pretrain_cost=None, chunk_rows=8192):
    b.natural(chunk_rows, True)
    implementation = {"prepare_l1_inputs.py": b.sha256_file(__file__), "sampling_builder.py": b.sha256_file(b.__file__)}
    b.need(evidence_kind in {"synthetic", "real"}, "explicit evidence kind required")
    pm, om = validate_prefix(p_prefix, "P"), validate_prefix(o_prefix, "O")
    initial_prefix_hashes = {arm: [b.sha256_file(Path(directory) / "manifest.json") for directory in prefix]
                             for arm, prefix in (("P", p_prefix), ("O", o_prefix))}
    b.need(len(pm) == len(om), "O/P prefix lengths differ")
    for p, o in zip(pm, om):
        b.need(all(c["fit_rows"] > 0 for data in (p, o) for c in data["classes"]),
               "L1 requires nonempty fitting support for every class")
        for field in ("dataset_id", "namespace", "increment", "feature_dim", "feature_dtype", "group_mode", "common_transform_fit_sha256"):
            b.need(p[field] == o[field], "O/P data or transform contract differs")
        b.need(p["binding"]["input_sha256"] == o["binding"]["input_sha256"], "O/P did not use identical available training contracts")
        b.need(p["collections"]["calibration"] == o["collections"]["calibration"], "O/P calibration changed")
        b.need(p["policy"]["seed"] == o["policy"]["seed"]
               and p["policy"]["calibration_numerator"] == o["policy"]["calibration_numerator"]
               and p["policy"]["calibration_denominator"] == o["policy"]["calibration_denominator"], "O/P sampler differs")
    first = pm[0]
    b.need(first["feature_dtype"] == "float32", "L1 exporter requires float32: no silent precision conversion")
    b.need(first["group_mode"] == "row_identity_proxy", "selected source has no official-test group sidecars; cannot claim provided_disjoint")
    b.need(re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", first["dataset_id"]), "unsafe exporter dataset name")
    test, test_sha = verified_test(test_directory)
    b.need(test["dataset_id"] == first["dataset_id"] and test["feature_dim"] == first["feature_dim"], "test dataset mismatch")
    train_classes = {c["class_id"] for data in pm for c in data["classes"]}
    b.need(train_classes == {record["class_id"] for record in test["source_shards"]}, "incomplete training prefix or unmatched official test support")
    b.need(all(sum(record["rows"] for record in test["source_shards"] if record["class_id"] == cid) > 0 for cid in train_classes),
           "L1 requires nonempty official test support for every class")
    metadata, metadata_sha = b.read_bound_json(checkpoint_metadata)
    state_sha = b.sha256_file(checkpoint_state)
    cost_sha = b.sha256_file(pretrain_cost) if pretrain_cost is not None else None
    b.need(metadata["inference_state_sha256"] == state_sha, "checkpoint state binding mismatch")
    b.need(metadata["checkpoint"] == 0 and metadata["seen_classes"] == [c["class_id"] for c in first["classes"]]
           and metadata["feature_dim"] == first["feature_dim"] and metadata.get("dataset") == first["dataset_id"], "checkpoint is not this dataset's shared Task 0")
    b.need(metadata.get("common_transform_fit_sha256") == first["common_transform_fit_sha256"], "checkpoint lacks the actual common Task-0 transform-cohort binding")
    expected_ids = first["collections"]["transform_fit"]["row_ids"]["sha256"]
    b.need(metadata.get("normalization_fit_row_ids_sha256") == expected_ids,
           "normalization was not bound to the common Task-0 fitting IDs")
    b.need(metadata.get("encoder_training_row_ids_sha256") == expected_ids,
           "encoder pretraining was not bound to the common Task-0 fitting IDs")
    b.need(metadata.get("evidence_kind") == evidence_kind, "checkpoint evidence kind mismatch")
    config, config_sha = b.read_bound_json(config_path)
    validate_config(config)
    output = b.fresh_directory(output)
    b.write_json(output / "STARTED.json", {"status": "INCOMPLETE", "stage": "offline_O_P_export_pair"})
    try:
        left = materialize_arm(output / "P", p_prefix, pm, test, source_root, metadata, checkpoint_state, checkpoint_metadata,
                               config, evidence_kind, pretrain_cost, chunk_rows, initial_prefix_hashes["P"])
        right = materialize_arm(output / "O", o_prefix, om, test, source_root, metadata, checkpoint_state, checkpoint_metadata,
                                config, evidence_kind, pretrain_cost, chunk_rows, initial_prefix_hashes["O"])
        b.need(left["checkpoint_metadata"] == right["checkpoint_metadata"] and left["checkpoint_state"] == right["checkpoint_state"],
               "O/P checkpoint changed between materializations")
        b.need(left["raw_splits"]["test"] == right["raw_splits"]["test"], "O/P official-test export differs")
        b.need(left["checkpoint_metadata"]["sha256"] == metadata_sha
               and left["checkpoint_state"]["sha256"] == state_sha
               and b.sha256_file(checkpoint_metadata) == metadata_sha
               and b.sha256_file(checkpoint_state) == state_sha
               and b.sha256_file(config_path) == config_sha, "shared upstream checkpoint/config changed")
        if pretrain_cost is not None:
            b.need(b.sha256_file(pretrain_cost) == cost_sha
                   and left["pretrain_cost_receipt"]["sha256"] == cost_sha
                   and right["pretrain_cost_receipt"]["sha256"] == cost_sha, "pretrain receipt changed")
        for record in test["source_shards"]:
            b.need(b.sha256_file(b.safe_path(source_root, record["path"])) == record["sha256"],
                   "official test source changed during export input creation")
        b.need(verified_test(test_directory)[1] == test_sha, "official test verification changed")
        for arm, prefix in (("P", p_prefix), ("O", o_prefix)):
            for directory, expected in zip(prefix, initial_prefix_hashes[arm]):
                b.load_increment(directory, expected)
        b.need(implementation == {"prepare_l1_inputs.py": b.sha256_file(__file__), "sampling_builder.py": b.sha256_file(b.__file__)},
               "bridge implementation changed during materialization")
        result = {"schema_version": 1, "status": "RAW_EXPORT_PAIR_COMPLETE", "evidence_kind": evidence_kind,
                  "P_export_manifest_sha256": b.sha256_file(output / "P" / "export-input.json"),
                  "O_export_manifest_sha256": b.sha256_file(output / "O" / "export-input.json"),
                  "source_checkpoint_state_sha256": state_sha, "common_transform_fit_sha256": first["common_transform_fit_sha256"],
                  "official_test_verification_sha256": test_sha, "offline_container_not_online_revelation": True,
                  "implementation_sha256": implementation,
                  "model_training_performed": False}
        b.write_json(output / "COMPLETE.json", result)
        return result
    except Exception as exc:
        b.write_json(output / "FAILED.json", {"status": "INCOMPLETE_FAILED", "error_type": type(exc).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-prefix", nargs="+", type=Path, required=True)
    parser.add_argument("--o-prefix", nargs="+", type=Path, required=True)
    for name in ("test-directory", "source-root", "checkpoint-metadata", "checkpoint-state", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evidence-kind", choices=("real", "synthetic"), required=True)
    parser.add_argument("--pretrain-cost", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        result = prepare_pair(args.p_prefix, args.o_prefix, args.test_directory, args.source_root,
                              args.checkpoint_metadata, args.checkpoint_state, args.config, args.output,
                              evidence_kind=args.evidence_kind, pretrain_cost=args.pretrain_cost, chunk_rows=args.chunk_rows)
        print(b.canonical(result).decode())
        return 0
    except (b.Blocked, ValueError, OSError, TypeError, KeyError, sqlite3.Error) as exc:
        print(b.canonical({"status": "BLOCKED", "error_type": type(exc).__name__}).decode())
        return 2


if __name__ == "__main__":
    sys.exit(main())
