"""Increment-available array builder: bounded-memory I/O, immutable outputs.

This module never trains, submits jobs, or opens a network connection. Real
dataset execution requires a separately authorized, measured compute stage.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys

import numpy as np


DOMAIN = "ofra-prospective-sampling-reference-v1"
PARTITIONS = ("fit", "calibration", "omitted", "transform_fit")
ARRAY_NAMES = ("x", "labels", "row_ids", "group_ids", "available_tasks", "source_index")
DTYPES = {"float32", "float64"}


class Blocked(ValueError):
    """Invalid or insufficient binding: do not consume any incomplete output."""


def need(condition, message):
    if not condition:
        raise Blocked(message)


def keys(value, expected, name):
    need(isinstance(value, dict) and set(value) == set(expected), f"invalid {name} keys")


def natural(value, positive=False):
    need(type(value) is int and value >= int(positive), "invalid nonnegative integer")
    return value


def digest(value):
    need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA-256")
    return value


def token(value):
    need(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. /-]{0,199}", value), "invalid stable identity")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def no_links(path):
    path = Path(path).absolute()
    for item in reversed((path,) + tuple(path.parents)):
        if item.exists() or item.is_symlink():
            st = item.lstat()
            need(not item.is_symlink() and not (getattr(st, "st_file_attributes", 0) & 0x400), "symlink or reparse point prohibited")
    return path


def safe_path(root, raw):
    need(isinstance(raw, str) and raw and "\\" not in raw and ":" not in raw, "unsafe relative path")
    relative = Path(raw)
    need(not relative.is_absolute() and all(x not in ("", ".", "..") for x in raw.split("/")), "unsafe relative path")
    root = no_links(root).resolve()
    path = no_links(root / relative)
    need(path.resolve().is_relative_to(root) and path != root, "path escapes root")
    return path


def sha256_file(path):
    path = no_links(path)
    need(path.is_file(), "bound file missing")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = no_links(path)
    need(path.stat().st_size <= 4 * 1024 * 1024, "JSON contract exceeds bounded metadata size")
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def number(value):
        value = float(value)
        need(math.isfinite(value), "nonfinite JSON number")
        return value
    def constant(_):
        raise Blocked("nonfinite JSON constant")
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                      parse_float=number, parse_constant=constant)


def write_json(path, value):
    path = no_links(path)
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_bound_json(path):
    before = sha256_file(path)
    value = read_json(path)
    need(sha256_file(path) == before, "JSON contract changed while reading")
    return value, before


def fresh_directory(path):
    path = no_links(path)
    need(path.parent.is_dir(), "output parent must already exist")
    need(not path.exists(), "output exists; overwrite and partial resume prohibited")
    path.mkdir()
    return path


def close_array(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        if getattr(array, "mode", "r") != "r":
            array.flush()
        mapping.close()


@contextmanager
def mapped(path):
    no_links(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        yield array
    finally:
        close_array(array)


def array_ref(root, path, rows):
    with mapped(path) as array:
        return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
                "rows": rows, "shape": list(array.shape), "dtype": array.dtype.str}


def input_ref(ref):
    keys(ref, {"path", "sha256", "rows"}, "input array reference")
    digest(ref["sha256"])
    natural(ref["rows"])


def checked_input(root, ref, kind, feature_dim, class_id, chunk_rows):
    input_ref(ref)
    path = safe_path(root, ref["path"])
    need(path.suffix == ".npy" and sha256_file(path) == ref["sha256"], "input file hash mismatch")
    with mapped(path) as array:
        need(len(array) == ref["rows"], "input row count mismatch")
        if kind == "features":
            need(array.ndim == 2 and array.shape[1] == feature_dim and array.dtype.name in DTYPES
                 and array.dtype.isnative and array.flags.c_contiguous, "invalid feature shape/dtype/layout")
        else:
            need(array.ndim == 1, "invalid sidecar shape")
            if kind == "labels":
                need(array.dtype.kind in "iu", "labels must be integer")
            else:
                need(array.dtype == np.dtype("S64"), "identity sidecar must use S64")
        for start in range(0, len(array), chunk_rows):
            block = array[start:start + chunk_rows]
            if kind == "features":
                need(bool(np.isfinite(block).all()), "nonfinite source features")
            elif kind == "labels":
                need(bool((block == class_id).all()), "class label sidecar mismatch")
            else:
                for value in block:
                    digest(bytes(value).decode("ascii"))
        dtype = array.dtype.name
    need(sha256_file(path) == ref["sha256"], "input changed during validation")
    return path, dtype


def validate_policy(policy):
    keys(policy, {"schema_version", "arm", "seed", "offline_normal_cap", "calibration_numerator",
                  "calibration_denominator", "transform_fit_policy"}, "policy")
    need(policy["schema_version"] == 1 and type(policy["schema_version"]) is int, "policy schema")
    need(policy["arm"] in {"O", "P"}, "only explicit offline O and prospective P arms supported")
    natural(policy["seed"])
    n, d = natural(policy["calibration_numerator"]), natural(policy["calibration_denominator"], True)
    need(n < d, "calibration fraction outside [0,1)")
    if policy["arm"] == "O":
        natural(policy["offline_normal_cap"], True)
    else:
        need(policy["offline_normal_cap"] is None, "P must not receive an offline cap")
    need(policy["transform_fit_policy"] == "common_task0_prospective_fit", "unsupported transform-fit policy")


def validate_input(manifest):
    keys(manifest, {"schema_version", "kind", "dataset_id", "namespace", "increment", "feature_dim",
                    "normal_class_id", "group_mode", "classes"}, "train increment")
    need(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1, "input schema")
    need(manifest["kind"] == "available_train_increment", "future/test/full-world manifest prohibited")
    token(manifest["dataset_id"])
    token(manifest["namespace"])
    natural(manifest["increment"])
    natural(manifest["feature_dim"], True)
    natural(manifest["normal_class_id"])
    need(manifest["group_mode"] in {"row_identity_proxy", "provided_disjoint"}, "unknown grouping contract")
    classes = manifest["classes"]
    need(isinstance(classes, list) and classes, "current classes missing")
    ids, shard_ids = set(), set()
    for cls in classes:
        keys(cls, {"class_id", "name", "introduced_increment", "partition", "shards"}, "class cohort")
        cid = natural(cls["class_id"])
        need(cid not in ids, "duplicate class")
        ids.add(cid)
        token(cls["name"])
        need(cls["introduced_increment"] == manifest["increment"] and type(cls["introduced_increment"]) is int,
             "future or historical class cohort")
        need(cls["partition"] == "train", "only training source accepted")
        need(isinstance(cls["shards"], list) and cls["shards"], "class shards missing")
        for shard in cls["shards"]:
            keys(shard, {"shard_id", "features", "row_ids", "group_ids", "labels"}, "train shard")
            token(shard["shard_id"])
            need(shard["shard_id"] not in shard_ids, "duplicate stable shard identity")
            shard_ids.add(shard["shard_id"])
            for kind in ("features", "row_ids", "group_ids", "labels"):
                if shard[kind] is not None:
                    input_ref(shard[kind])
                    need(shard[kind]["rows"] == shard["features"]["rows"], "sidecar row mismatch")
            need(shard["features"] is not None, "features required")
            if shard["row_ids"] is None:
                need(shard["shard_id"] == shard["features"]["path"], "generated IDs require the canonical source relative path as shard_id")
            need((shard["group_ids"] is not None) == (manifest["group_mode"] == "provided_disjoint"), "group claim/sidecar mismatch")
    normal = manifest["normal_class_id"]
    need((normal in ids) == (manifest["increment"] == 0), "normal class must occur exactly at Task 0")


def rank(seed, stage, class_id, row_id):
    return hashlib.sha256(json.dumps([DOMAIN, seed, stage, class_id, row_id],
                                    ensure_ascii=True, separators=(",", ":")).encode()).digest()


def row_identity(namespace, shard_id, offset):
    # Never use a global manifest hash, future counts, class list, or test hash.
    return object_hash(["source-row-v1", namespace, shard_id, offset])


def cal_count(rows, policy):
    if rows <= 1 or policy["calibration_numerator"] == 0:
        return 0
    return min(rows - 1, max(1, rows * policy["calibration_numerator"] // policy["calibration_denominator"]))


def database(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA mmap_size=0")
    db.execute("PRAGMA threads=1")
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("CREATE TABLE IF NOT EXISTS rows (rid TEXT PRIMARY KEY, gid TEXT NOT NULL, cid INTEGER NOT NULL, "
               "inc INTEGER NOT NULL, sid TEXT NOT NULL, slot INTEGER NOT NULL, offset INTEGER NOT NULL, "
               "cal_rank BLOB NOT NULL, fit_rank BLOB NOT NULL, part TEXT NOT NULL, transform_fit INTEGER NOT NULL DEFAULT 0)")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS source_identity ON rows(sid,offset)")
    db.execute("CREATE INDEX IF NOT EXISTS cal_order ON rows(cid,cal_rank,rid)")
    db.execute("CREATE INDEX IF NOT EXISTS fit_order ON rows(cid,fit_rank,rid)")
    return db


def ingest(db, manifest, root, policy, chunk_rows):
    source_files, source_shards, dtype = [], [], None
    inc, dim = manifest["increment"], manifest["feature_dim"]
    for cls in sorted(manifest["classes"], key=lambda c: c["class_id"]):
        cid = cls["class_id"]
        need(db.execute("SELECT 1 FROM rows WHERE cid=? LIMIT 1", (cid,)).fetchone() is None, "historical class rewrite")
        for shard in sorted(cls["shards"], key=lambda s: s["shard_id"]):
            slot = len(source_shards)
            paths = {}
            for kind in ("features", "row_ids", "group_ids", "labels"):
                ref = shard[kind]
                if ref is not None:
                    path, observed_dtype = checked_input(root, ref, kind, dim, cid, chunk_rows)
                    paths[kind] = path
                    source_files.append(ref)
                    if kind == "features":
                        need(dtype is None or dtype == observed_dtype, "inconsistent feature dtype")
                        dtype = observed_dtype
            source_shards.append({"class_id": cid, "shard_id": shard["shard_id"], **shard})
            ids = np.load(paths["row_ids"], mmap_mode="r", allow_pickle=False) if "row_ids" in paths else None
            groups = np.load(paths["group_ids"], mmap_mode="r", allow_pickle=False) if "group_ids" in paths else None
            try:
                for start in range(0, shard["features"]["rows"], chunk_rows):
                    records = []
                    for offset in range(start, min(start + chunk_rows, shard["features"]["rows"])):
                        rid = bytes(ids[offset]).decode("ascii") if ids is not None else row_identity(manifest["namespace"], shard["shard_id"], offset)
                        gid = bytes(groups[offset]).decode("ascii") if groups is not None else rid
                        records.append((rid, gid, cid, inc, shard["shard_id"], slot, offset,
                                        rank(policy["seed"], "calibration", cid, rid), rank(policy["seed"], "normal-fit", cid, rid), "pool"))
                    db.executemany("INSERT INTO rows(rid,gid,cid,inc,sid,slot,offset,cal_rank,fit_rank,part) VALUES(?,?,?,?,?,?,?,?,?,?)", records)
                    db.commit()
            except sqlite3.IntegrityError as exc:
                raise Blocked("row ID or original source identity collision") from exc
            finally:
                if ids is not None:
                    close_array(ids)
                if groups is not None:
                    close_array(groups)
    return source_files, source_shards, dtype


def select_rows(db, manifest, policy, previous):
    summaries = []
    for cls in sorted(manifest["classes"], key=lambda c: c["class_id"]):
        cid = cls["class_id"]
        n = db.execute("SELECT count(*) FROM rows WHERE cid=?", (cid,)).fetchone()[0]
        calibration = cal_count(n, policy)
        db.execute("UPDATE rows SET part='calibration' WHERE rid IN (SELECT rid FROM rows WHERE cid=? ORDER BY cal_rank,rid LIMIT ?)", (cid, calibration))
        summaries.append({"class_id": cid, "name": cls["name"], "introduced_increment": manifest["increment"],
                          "source_rows": n, "calibration_rows": calibration, "fit_pool_rows": n - calibration})
    if previous is None:
        attacks = [c["fit_pool_rows"] for c in summaries if c["class_id"] != manifest["normal_class_id"]]
        need(attacks and max(attacks) > 0, "no nonempty Task-0 attack fit pool")
        prospective_cap = max(attacks)
        cap = prospective_cap if policy["arm"] == "P" else policy["offline_normal_cap"]
        need(cap >= prospective_cap, "O must contain P Task-0 fit cohort for the common transform")
    else:
        cap, prospective_cap = previous["normal_cap"], previous["prospective_task0_cap"]
    for summary in summaries:
        cid = summary["class_id"]
        limit = cap if cid == manifest["normal_class_id"] else summary["fit_pool_rows"]
        if manifest["increment"] == 0:
            transform_limit = prospective_cap if cid == manifest["normal_class_id"] else summary["fit_pool_rows"]
            db.execute("UPDATE rows SET transform_fit=1 WHERE rid IN (SELECT rid FROM rows WHERE cid=? AND part='pool' ORDER BY fit_rank,rid LIMIT ?)", (cid, transform_limit))
        db.execute("UPDATE rows SET part='fit' WHERE rid IN (SELECT rid FROM rows WHERE cid=? AND part='pool' ORDER BY fit_rank,rid LIMIT ?)", (cid, limit))
        db.execute("UPDATE rows SET part='omitted' WHERE cid=? AND part='pool'", (cid,))
        counts = dict(db.execute("SELECT part,count(*) FROM rows WHERE cid=? GROUP BY part", (cid,)))
        summary["fit_rows"] = counts.get("fit", 0)
        summary["omitted_rows"] = counts.get("omitted", 0)
        need(sum(counts.values()) == summary["source_rows"] and counts.get("calibration", 0) == summary["calibration_rows"], "selection conservation failed")
        summary["warning"] = "no_calibration_support" if summary["calibration_rows"] == 0 else None
    if manifest["group_mode"] == "provided_disjoint":
        collision = db.execute("SELECT gid FROM rows WHERE part IN ('fit','calibration') GROUP BY gid HAVING count(DISTINCT part)>1 LIMIT 1").fetchone()
        need(collision is None, "provided group crosses fitting/calibration boundary; grouped redesign required")
    db.commit()
    return summaries, cap, prospective_cap


def new_array(path, dtype, shape):
    need(not path.exists(), "output array already exists")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def materialize(db, output, name, manifest, shards, source_root, dtype, chunk_rows):
    where = "inc=? AND transform_fit=1" if name == "transform_fit" else "inc=? AND part=?"
    params = (manifest["increment"],) if name == "transform_fit" else (manifest["increment"], name)
    count = db.execute(f"SELECT count(*) FROM rows WHERE {where}", params).fetchone()[0]
    folder = output / name
    folder.mkdir()
    arrays = {}
    specs = {"labels": ("<i8", (count,)), "row_ids": ("S64", (count,)), "group_ids": ("S64", (count,)),
             "available_tasks": ("<i8", (count,)), "source_index": ("<i8", (count, 2))}
    if name != "omitted":
        specs["x"] = (np.dtype(dtype), (count, manifest["feature_dim"]))
    try:
        for key, (array_dtype, shape) in specs.items():
            arrays[key] = new_array(folder / f"{key}.npy", array_dtype, shape)
        cursor = db.execute(f"SELECT rid,gid,cid,inc,slot,offset FROM rows WHERE {where} ORDER BY cid,sid,offset", params)
        written = 0
        while True:
            rows = cursor.fetchmany(chunk_rows)
            if not rows:
                break
            end = written + len(rows)
            arrays["row_ids"][written:end] = [r[0] for r in rows]
            arrays["group_ids"][written:end] = [r[1] for r in rows]
            arrays["labels"][written:end] = [r[2] for r in rows]
            arrays["available_tasks"][written:end] = [r[3] for r in rows]
            arrays["source_index"][written:end] = [(r[4], r[5]) for r in rows]
            if "x" in arrays:
                start = 0
                while start < len(rows):
                    slot = rows[start][4]
                    stop = start + 1
                    while stop < len(rows) and rows[stop][4] == slot:
                        stop += 1
                    ref = shards[slot]["features"]
                    path = safe_path(source_root, ref["path"])
                    with mapped(path) as source:
                        arrays["x"][written + start:written + stop] = source[[r[5] for r in rows[start:stop]]]
                    start = stop
            written = end
        need(written == count, "materialized row count mismatch")
    finally:
        for array in arrays.values():
            close_array(array)
    return {"rows": count, **{key: array_ref(output, folder / f"{key}.npy", count) if key in specs else None for key in ARRAY_NAMES}}


def complete(output, manifest, status):
    write_json(output / "manifest.json", manifest)
    files = {}
    for path in sorted(output.rglob("*")):
        no_links(path)
        if path.is_file():
            files[path.relative_to(output).as_posix()] = sha256_file(path)
    marker = {"schema_version": 1, "status": status, "manifest_sha256": files["manifest.json"], "files": files}
    write_json(output / "COMPLETE.json", marker)
    return manifest


def load_increment(directory, expected_manifest_sha256=None):
    """Verify a complete immutable increment and every output file before use."""
    directory = no_links(directory)
    marker = read_json(directory / "COMPLETE.json")
    keys(marker, {"schema_version", "status", "manifest_sha256", "files"}, "completion marker")
    need(type(marker["schema_version"]) is int and marker["schema_version"] == 1
         and marker["status"] == "TRAIN_INCREMENT_COMPLETE", "not a complete train increment")
    need(isinstance(marker["files"], dict) and "manifest.json" in marker["files"], "completion closure missing")
    actual = set()
    for path in directory.rglob("*"):
        no_links(path)
        if path.is_file() and path.relative_to(directory).as_posix() != "COMPLETE.json":
            actual.add(path.relative_to(directory).as_posix())
    need(actual == set(marker["files"]), "output closure changed or partial files present")
    for raw, expected in marker["files"].items():
        digest(expected)
        need(sha256_file(safe_path(directory, raw)) == expected, "completed artifact hash mismatch")
    need(marker["files"]["manifest.json"] == marker["manifest_sha256"], "manifest binding mismatch")
    if expected_manifest_sha256 is not None:
        need(digest(expected_manifest_sha256) == marker["manifest_sha256"], "externally bound manifest changed")
    manifest = read_json(directory / "manifest.json")
    keys(manifest, {"schema_version", "kind", "status", "dataset_id", "namespace", "increment", "feature_dim",
                    "normal_class_id", "group_mode", "feature_dtype", "capture_independent_established",
                    "group_boundary_checked", "policy", "binding", "normal_cap", "prospective_task0_cap",
                    "common_transform_fit_sha256", "source_shards", "classes", "historical_classes",
                    "collections", "sampler_domain", "official_test_used_for_selection", "model_training_performed"}, "completed manifest")
    need(manifest.get("kind") == "derived_train_increment" and manifest.get("status") == "COMPLETE", "invalid completed manifest")
    need(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1, "derived schema")
    natural(manifest["increment"])
    natural(manifest["feature_dim"], True)
    need(manifest["feature_dtype"] in DTYPES, "derived feature dtype")
    need(manifest["official_test_used_for_selection"] is False and manifest["model_training_performed"] is False
         and manifest["capture_independent_established"] is False, "unsupported derived evidence claim")
    validate_policy(manifest["policy"])
    keys(manifest["collections"], PARTITIONS, "collections")
    for part, collection in manifest["collections"].items():
        need(part in PARTITIONS, "unknown collection")
        if collection is None:
            need(part == "transform_fit" and manifest["increment"] > 0, "unexpected missing collection")
            continue
        keys(collection, {"rows", *ARRAY_NAMES}, "collection")
        natural(collection["rows"])
        for name in ARRAY_NAMES:
            ref = collection[name]
            if ref is None:
                need(part == "omitted" and name == "x", "unexpected missing array")
                continue
            keys(ref, {"path", "sha256", "rows", "shape", "dtype"}, "derived array reference")
            need(ref["sha256"] == marker["files"].get(ref["path"]), "collection not bound to closure")
            with mapped(safe_path(directory, ref["path"])) as array:
                need(list(array.shape) == ref["shape"] and array.dtype.str == ref["dtype"]
                     and len(array) == collection["rows"] == ref["rows"], "collection shape/dtype mismatch")
                expected_shape = (collection["rows"], manifest["feature_dim"]) if name == "x" else (
                    (collection["rows"], 2) if name == "source_index" else (collection["rows"],))
                expected_dtype = manifest["feature_dtype"] if name == "x" else ("S64" if name in {"row_ids", "group_ids"} else "int64")
                need(array.shape == expected_shape and array.dtype == np.dtype(expected_dtype), "derived semantic array type mismatch")
    return manifest


def iter_partition(directory, partition="fit", chunk_rows=8192, expected_manifest_sha256=None):
    """Yield bounded copied chunks of raw x, labels, IDs and provenance arrays."""
    natural(chunk_rows, True)
    need(partition in PARTITIONS, "unknown partition")
    manifest = load_increment(directory, expected_manifest_sha256)
    collection = manifest["collections"][partition]
    need(collection is not None, "transform-fit collection exists at Task 0 only")
    arrays = {}
    try:
        for name in ARRAY_NAMES:
            if collection[name] is not None:
                arrays[name] = np.load(safe_path(directory, collection[name]["path"]), mmap_mode="r", allow_pickle=False)
        for start in range(0, collection["rows"], chunk_rows):
            yield {key: np.array(array[start:start + chunk_rows], copy=True) for key, array in arrays.items()}
    finally:
        for array in arrays.values():
            close_array(array)


def derive(input_path, source_root, output, policy_path, *, previous=None, resume=False, chunk_rows=8192):
    natural(chunk_rows, True)
    manifest, input_sha = read_bound_json(input_path)
    policy, policy_sha = read_bound_json(policy_path)
    validate_input(manifest)
    validate_policy(policy)
    source_root = no_links(source_root)
    need(source_root.is_dir(), "source root missing")
    previous_data = load_increment(previous) if previous is not None else None
    previous_sha = sha256_file(Path(previous) / "manifest.json") if previous is not None else None
    need((previous_data is None) == (manifest["increment"] == 0), "prefix required exactly for later increments")
    if previous_data is not None:
        need(previous_data["increment"] + 1 == manifest["increment"], "nonconsecutive prefix")
        need(previous_data["policy"] == policy, "policy changed after Task 0")
        need(previous_data["binding"]["implementation_sha256"] == sha256_file(__file__), "implementation changed across prefix")
        for field in ("dataset_id", "namespace", "feature_dim", "normal_class_id", "group_mode"):
            need(previous_data[field] == manifest[field], "dataset contract changed across prefix")
        prior = {c["class_id"] for c in previous_data["historical_classes"]}
        need(not prior.intersection(c["class_id"] for c in manifest["classes"]), "historical class rewrite")
    binding = {"input_sha256": input_sha, "policy_sha256": policy_sha,
               "implementation_sha256": sha256_file(__file__), "previous_manifest_sha256": previous_sha}
    output = no_links(output)
    if output.exists():
        need(resume, "output exists; explicit completed-output resume required")
        existing = load_increment(output)
        need(existing["binding"] == binding, "resume input/policy/code/prefix changed")
        for cls in manifest["classes"]:
            for shard in cls["shards"]:
                for kind in ("features", "row_ids", "group_ids", "labels"):
                    if shard[kind] is not None:
                        checked_input(source_root, shard[kind], kind, manifest["feature_dim"], cls["class_id"], chunk_rows)
        for cls in manifest["classes"]:
            for shard in cls["shards"]:
                for kind in ("features", "row_ids", "group_ids", "labels"):
                    if shard[kind] is not None:
                        ref = shard[kind]
                        need(sha256_file(safe_path(source_root, ref["path"])) == ref["sha256"], "source changed during resume validation")
        need(sha256_file(input_path) == input_sha and sha256_file(policy_path) == policy_sha
             and sha256_file(__file__) == binding["implementation_sha256"], "resume contract or implementation changed during validation")
        if previous is not None:
            load_increment(previous, previous_sha)
        return existing
    output = fresh_directory(output)
    write_json(output / "STARTED.json", {"status": "INCOMPLETE", **binding})
    write_json(output / "input.json", manifest)
    write_json(output / "policy.json", policy)
    db = None
    try:
        ledger = output / "selection.sqlite"
        if previous_data is not None:
            shutil.copyfile(safe_path(previous, "selection.sqlite"), ledger)
        db = database(ledger)
        files, shards, dtype = ingest(db, manifest, source_root, policy, chunk_rows)
        need(previous_data is None or previous_data["feature_dtype"] == dtype, "feature dtype changed across prefix")
        classes, cap, prospective_cap = select_rows(db, manifest, policy, previous_data)
        collections = {}
        for partition in PARTITIONS:
            collections[partition] = None if partition == "transform_fit" and manifest["increment"] > 0 else materialize(
                db, output, partition, manifest, shards, source_root, dtype, chunk_rows)
        db.commit()
        db.close()
        db = None
        # Rehash after extraction. Any mutation invalidates the entire directory.
        for ref in files:
            need(sha256_file(safe_path(source_root, ref["path"])) == ref["sha256"], "source changed during derivation")
        need(sha256_file(__file__) == binding["implementation_sha256"], "implementation changed during derivation")
        need(sha256_file(input_path) == input_sha and sha256_file(policy_path) == policy_sha,
             "input or policy contract changed during derivation")
        if previous_data is not None:
            load_increment(previous)
            need(sha256_file(Path(previous) / "manifest.json") == previous_sha, "prefix changed during derivation")
        transform = object_hash(collections["transform_fit"]) if previous_data is None else previous_data["common_transform_fit_sha256"]
        result = {"schema_version": 1, "kind": "derived_train_increment", "status": "COMPLETE",
                  **{key: manifest[key] for key in ("dataset_id", "namespace", "increment", "feature_dim", "normal_class_id", "group_mode")},
                  "feature_dtype": dtype, "capture_independent_established": False,
                  "group_boundary_checked": manifest["group_mode"] == "provided_disjoint",
                  "policy": policy, "binding": binding, "normal_cap": cap, "prospective_task0_cap": prospective_cap,
                  "common_transform_fit_sha256": transform, "source_shards": shards,
                  "classes": classes, "historical_classes": (previous_data["historical_classes"] if previous_data else []) + classes,
                  "collections": collections, "sampler_domain": DOMAIN,
                  "official_test_used_for_selection": False, "model_training_performed": False}
        complete(output, result, "TRAIN_INCREMENT_COMPLETE")
        load_increment(output)
        return result
    except Exception as exc:
        if db is not None:
            db.close()
        if not (output / "COMPLETE.json").exists() and not (output / "FAILED.json").exists():
            write_json(output / "FAILED.json", {"status": "INCOMPLETE_FAILED", "error_type": type(exc).__name__})
        raise


def adapt_replayids(source_manifest, output, *, expected_sha256, namespace="replayids-contract-source-v2"):
    """Metadata-only isolation adapter. No arrays read; no sampler is called.

    Its full-source provenance remains in a separate audit, never train input.
    Default labels, order and paths come from the bound existing source.
    """
    digest(expected_sha256)
    need(sha256_file(source_manifest) == expected_sha256, "upstream manifest hash mismatch")
    source, observed_sha = read_bound_json(source_manifest)
    need(observed_sha == expected_sha256, "upstream manifest changed before read")
    keys(source, {"classes", "dataset", "feature_dim", "metric_profile", "normal_class_id", "problem_type",
                  "schema_version", "source", "task_semantics", "tasks"}, "ReplayIDS source")
    need(type(source["schema_version"]) is int and source["schema_version"] == 1
         and source["task_semantics"] == "class_incremental", "source schema/semantics")
    all_classes = {}
    for cls in source["classes"]:
        keys(cls, {"id", "name", "train", "test"}, "ReplayIDS class")
        cid = natural(cls["id"])
        need(cid not in all_classes, "duplicate source class")
        for partition in ("train", "test"):
            need(isinstance(cls[partition], list) and cls[partition], "source shards missing")
            for ref in cls[partition]:
                input_ref(ref)
        all_classes[cid] = cls
    flat = [cid for task in source["tasks"] for cid in task]
    need(len(flat) == len(set(flat)) and set(flat) == set(all_classes) and source["normal_class_id"] in source["tasks"][0], "invalid source task order")
    output = fresh_directory(output)
    emitted = []
    for increment, task in enumerate(source["tasks"]):
        manifest = {"schema_version": 1, "kind": "available_train_increment", "dataset_id": source["dataset"],
                    "namespace": namespace, "increment": increment, "feature_dim": source["feature_dim"],
                    "normal_class_id": source["normal_class_id"], "group_mode": "row_identity_proxy", "classes": []}
        for cid in sorted(task):
            cls = all_classes[cid]
            manifest["classes"].append({"class_id": cid, "name": cls["name"], "introduced_increment": increment, "partition": "train",
                                        "shards": [{"shard_id": ref["path"], "features": ref, "row_ids": None, "group_ids": None, "labels": None} for ref in cls["train"]]})
        validate_input(manifest)
        filename = f"increment_{increment:02d}.json"
        write_json(output / filename, manifest)
        emitted.append({"path": filename, "sha256": sha256_file(output / filename)})
    tests = {"schema_version": 1, "kind": "sealed_official_test", "dataset_id": source["dataset"],
             "feature_dim": source["feature_dim"], "classes": [{"class_id": cid, "name": cls["name"], "test": cls["test"]} for cid, cls in sorted(all_classes.items())]}
    write_json(output / "official_test.json", tests)
    audit = {"schema_version": 1, "kind": "metadata_adapter_audit", "source_manifest_sha256": expected_sha256,
             "train_increment_contracts": emitted, "official_test_contract_sha256": sha256_file(output / "official_test.json"),
             "arrays_read": False, "sampling_performed": False, "capture_groups_available": False}
    need(sha256_file(source_manifest) == expected_sha256, "upstream manifest changed during adaptation")
    write_json(output / "ADAPTER_COMPLETE.json", audit)
    return audit


def verify_official_test(input_path, source_root, output, *, chunk_rows=8192):
    """Separate full-support read-only source verification, never sampling."""
    natural(chunk_rows, True)
    contract, input_sha = read_bound_json(input_path)
    keys(contract, {"schema_version", "kind", "dataset_id", "feature_dim", "classes"}, "official test contract")
    need(type(contract["schema_version"]) is int and contract["schema_version"] == 1 and contract["kind"] == "sealed_official_test", "test contract required")
    natural(contract["feature_dim"], True)
    token(contract["dataset_id"])
    need(isinstance(contract["classes"], list) and contract["classes"], "test classes missing")
    seen, records = set(), []
    for cls in contract["classes"]:
        keys(cls, {"class_id", "name", "test"}, "test class")
        cid = natural(cls["class_id"])
        need(cid not in seen, "duplicate official test class")
        seen.add(cid)
        token(cls["name"])
        need(isinstance(cls["test"], list) and cls["test"], "official test shards missing")
        for ref in cls["test"]:
            _, dtype = checked_input(source_root, ref, "features", contract["feature_dim"], cid, chunk_rows)
            records.append({"class_id": cid, "dtype": dtype, **ref})
    for record in records:
        need(sha256_file(safe_path(source_root, record["path"])) == record["sha256"], "official test changed during verification")
    need(sha256_file(input_path) == input_sha, "official test contract changed during verification")
    output = fresh_directory(output)
    result = {"schema_version": 1, "kind": "official_test_verification", "status": "COMPLETE",
              "input_sha256": input_sha, "dataset_id": contract["dataset_id"], "feature_dim": contract["feature_dim"],
              "source_shards": records, "total_rows": sum(r["rows"] for r in records),
              "test_sampled": False, "source_modified": False, "participated_in_train_selection": False}
    return complete(output, result, "OFFICIAL_TEST_VERIFIED")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    derivation = commands.add_parser("derive")
    for name in ("input", "source-root", "output", "policy"):
        derivation.add_argument("--" + name, type=Path, required=True)
    derivation.add_argument("--previous", type=Path)
    derivation.add_argument("--resume", action="store_true")
    derivation.add_argument("--chunk-rows", type=int, default=8192)
    adapter = commands.add_parser("adapt-replayids")
    adapter.add_argument("--source-manifest", type=Path, required=True)
    adapter.add_argument("--expected-sha256", required=True)
    adapter.add_argument("--output", type=Path, required=True)
    test = commands.add_parser("verify-test")
    for name in ("input", "source-root", "output"):
        test.add_argument("--" + name, type=Path, required=True)
    test.add_argument("--chunk-rows", type=int, default=8192)
    check = commands.add_parser("check")
    check.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "derive":
            result = derive(args.input, args.source_root, args.output, args.policy, previous=args.previous,
                            resume=args.resume, chunk_rows=args.chunk_rows)
        elif args.command == "adapt-replayids":
            result = adapt_replayids(args.source_manifest, args.output, expected_sha256=args.expected_sha256)
        elif args.command == "verify-test":
            result = verify_official_test(args.input, args.source_root, args.output, chunk_rows=args.chunk_rows)
        else:
            result = load_increment(args.directory)
        print(json.dumps({"status": result.get("status", "METADATA_PREPARED"), "kind": result["kind"], "training_performed": False}))
        return 0
    except (Blocked, OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__, "reason": str(exc), "training_performed": False}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
