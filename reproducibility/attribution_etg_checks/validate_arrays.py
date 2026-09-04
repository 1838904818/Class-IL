"""Verify stored attribution rankings and probe IDs, without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from validate_semantics import METHODS, canonical, checked_json, keyed, require, validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path, expected_keys):
    require(path.stat().st_size < 8_000_000, "NPZ exceeds this mean-attribution contract")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)), "Duplicate NPZ member")
        require(set(names) == {name + ".npy" for name in expected_keys}, "NPZ member registry mismatch")
        require(sum(entry.file_size for entry in entries) < 32_000_000, "Expanded NPZ exceeds contract")
        require(all(not entry.is_dir() and not (entry.flag_bits & 1) for entry in entries), "Invalid NPZ entry")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in expected_keys}


def ranking(mean_abs, mean_signed):
    require(mean_abs.shape == mean_signed.shape == (77,), "Attribution vector shape mismatch")
    require(mean_abs.dtype.kind == mean_signed.dtype.kind == "f", "Attributions must be real floating arrays")
    require(np.isfinite(mean_abs).all() and np.isfinite(mean_signed).all(), "Non-finite attribution vector")
    require((mean_abs >= 0).all(), "Negative mean absolute attribution")
    scale = np.maximum(mean_abs.astype(np.float64), 1e-12)
    tolerance = 16 * max(np.finfo(mean_abs.dtype).eps, np.finfo(mean_signed.dtype).eps) * scale
    require(np.all(np.abs(mean_signed.astype(np.float64)) <= mean_abs.astype(np.float64) + tolerance),
            "Signed mean exceeds mean absolute attribution")
    # Independent Python sorting: feature index breaks ties, matching stable order.
    return sorted(range(77), key=lambda index: (-float(mean_abs[index]), index))[:15]


def verify_rows(artifact, analysis, probes, expected_arrays, secondary_arrays):
    rows = artifact["checkpoint_rows"]
    expected_keys = {(int(r["checkpoint"]), int(r["class_id"])) for r in rows[METHODS[0]]}
    analysis_rows = keyed(analysis["checkpoint_rows"], ("checkpoint", "class_id"), expected_keys, "Expected Gradients analysis")
    require(analysis["seed"] == artifact["seed"] and analysis["dataset"] == artifact["dataset"], "Analysis scope mismatch")
    require(analysis["attribution_scope"] == artifact["attribution_scope"], "Analysis target scope mismatch")
    require(probes.get("dataset") == artifact["dataset"] and probes.get("raw_feature_rows_persisted") is False, "Probe manifest scope mismatch")
    require(probes["streaming_manifest_sha256"] == artifact["source_bindings"]["streaming_manifest_file_sha256"], "Probe data manifest mismatch")
    test = probes["official_test"]["samples"]
    background = probes["task0_train_background"]["samples"]
    require(probes["task0_train_background"]["source_classes"] == [0, 1], "Background is not Task 0")
    require(len(background) > 0 and len(test) > 0, "Empty probe manifest")
    for split, samples, classes in (("official_test", test, set(range(10))), ("task0_train_background", background, {0, 1})):
        seen = set()
        for sample in samples:
            sample_id = sample["sample_id_sha256"]
            require(isinstance(sample_id, str) and re.fullmatch("[0-9a-f]{64}", sample_id) is not None, "Invalid sample identity")
            require(sample_id not in seen, "Duplicate probe identity")
            seen.add(sample_id)
            require(type(sample["class_id"]) is int and sample["class_id"] in classes, "Unexpected probe class")
            require(sample["split"] == split and sample["dataset"] == artifact["dataset"], "Probe split mismatch")
            require(sample["streaming_manifest_sha256"] == probes["streaming_manifest_sha256"], "Probe row data binding mismatch")
    require(not ({s["sample_id_sha256"] for s in test} & {s["sample_id_sha256"] for s in background}), "Test/background identity overlap")
    ids_by_class = {c: [s["sample_id_sha256"] for s in test if s["class_id"] == c] for c in range(10)}
    checked = 0
    for method in METHODS:
        for row in rows[method]:
            checkpoint, class_id = row["checkpoint"], row["class_id"]
            core = f"checkpoint_{checkpoint:03d}_class_{class_id:03d}"
            prefix = core if method == METHODS[0] else method + "__" + core
            arrays = expected_arrays if method == METHODS[0] else secondary_arrays
            require(ranking(arrays[prefix + "_mean_abs"], arrays[prefix + "_mean_signed"]) == row["top15_indices"], "Stored ranking differs from attribution arrays")
            require(row["probe_rows"] == len(ids_by_class[class_id]), "Probe row count mismatch")
            require(row["background_rows"] == len(background), "Background count mismatch")
            if method == METHODS[0]:
                saved = arrays[core + "_sample_id_sha256"]
                require(saved.dtype == np.dtype("S64") and saved.ndim == 1, "Probe identity array dtype/shape mismatch")
                require(saved.tolist() == [s.encode("ascii") for s in ids_by_class[class_id]], "Probe identity/order mismatch")
                original = analysis_rows[(checkpoint, class_id)]
                for field in ("recall", "rationale_mass", "random_null_95", "mass_margin", "admitted", "top15_indices", "top15_features", "etg_state", "etg_action", "probe_rows", "background_rows"):
                    require(row[field] == original[field], "Expected-Gradients row changed during robustness assembly")
            checked += 1
    return {"status": "stored_arrays_verified_single_seed_not_campaign_completion", "seed": artifact["seed"],
            "rankings_checked": checked, "mean_vectors_checked": checked*2,
            "probe_identity_arrays_checked": len(expected_keys), "unique_test_probes": len(test),
            "background_samples": len(background), "limitation": "Checks stored vectors and bound manifest identities; does not recompute sample-level attributions, prove source-row contents, or replay deletion controls."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, required=True, help="Local flat export of the seed files listed in README")
    parser.add_argument("--protected-manifest-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.seed_dir / "SHA256SUMS"
    require(digest(manifest_path) == args.protected_manifest_sha256, "Protected manifest hash mismatch")
    manifest = {}
    for line in manifest_path.read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        require(re.fullmatch("[0-9a-f]{64}", sha) is not None and name not in manifest, "Invalid checksum manifest")
        manifest[name] = sha
    # Only this fixed whitelist is read. Paths in the manifest are never executed.
    mapping = {"artifact": ("attribution_robustness.json", "./robustness/attribution_robustness.json"),
               "analysis": ("analysis.json", "./expected-gradients-etg/analysis.json"),
               "eg_arrays": ("attributions.npz", "./expected-gradients-etg/attributions.npz"),
               "secondary_arrays": ("attribution_robustness_mean_attributions.npz", "./robustness/attribution_robustness_mean_attributions.npz")}
    files, hashes = {}, {}
    for name, (local, registered) in mapping.items():
        files[name], hashes[name] = args.seed_dir/local, manifest[registered]
        require(digest(files[name]) == hashes[name], "Protected input hash mismatch")
    artifact = json.loads(files["artifact"].read_text())
    seed = artifact["seed"]
    training_file = args.seed_dir / f"result_seed_{seed}.json"
    hashes["training"] = manifest[f"./result_seed_{seed}.json"]
    training = checked_json(training_file, hashes["training"])
    require(artifact["source_bindings"]["training_result_file_sha256"] == hashes["training"], "Training binding mismatch")
    hashes["features"] = artifact["source_bindings"]["feature_schema_file_sha256"]
    features = checked_json(args.features, hashes["features"])
    validate(artifact, training, features)
    analysis = json.loads(files["analysis"].read_text())
    for value in (analysis,):
        require(value["canonical_sha256"] == canonical({k: v for k, v in value.items() if k != "canonical_sha256"}), "Analysis canonical hash mismatch")
    bindings = artifact["methods"]["expected_gradients"]
    require(bindings["analysis_file_sha256"] == hashes["analysis"] and bindings["attributions_file_sha256"] == hashes["eg_arrays"], "Expected-Gradients binding mismatch")
    require(artifact["output_bindings"]["mean_attributions_file_sha256"] == hashes["secondary_arrays"], "Secondary array binding mismatch")
    hashes["probes"] = training["monitoring"]["probe_manifest"]["file_sha256"]
    probes = checked_json(args.seed_dir / "probe_manifest.json", hashes["probes"])
    require(probes["canonical_sha256"] == canonical({k: v for k, v in probes.items() if k != "canonical_sha256"}), "Probe canonical hash mismatch")
    cores = [f"checkpoint_{r['checkpoint']:03d}_class_{r['class_id']:03d}" for r in artifact["checkpoint_rows"][METHODS[0]]]
    expected_keys = {core+suffix for core in cores for suffix in ("_mean_abs", "_mean_signed", "_sample_id_sha256")}
    secondary_keys = {method+"__"+core+suffix for method in METHODS[1:] for core in cores for suffix in ("_mean_abs", "_mean_signed")}
    output = verify_rows(artifact, analysis, probes, read_npz(files["eg_arrays"], expected_keys), read_npz(files["secondary_arrays"], secondary_keys))
    output["input_sha256"] = {"protected_manifest": args.protected_manifest_sha256, **hashes}
    output["validator_sha256"] = digest(Path(__file__))
    output["semantic_validator_sha256"] = digest(Path(__file__).with_name("validate_semantics.py"))
    output["canonical_sha256"] = canonical(output)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"STORED_ARRAY_CHECKS=PASS seed={seed}; rankings={output['rankings_checked']}; no model inference")


if __name__ == "__main__":
    main()
