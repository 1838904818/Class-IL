"""Adapt four hash-bound native OFRA receipts into a sealed score-level bundle.

No model inference or attribution is executed. Real exports remain pending
external fidelity/lineage/governance evidence; COMPLETE is not scientific PASS.
"""
import argparse
from pathlib import Path

import numpy as np

from score_core import Blocked, check, sigmoid
from runner import (bound_path, file_hash, read_json, reject_links, save_json,
                    source_binding, validate_manifest)


EXPECTED_ARRAYS = {f"{time}_{kind}" for time in ("old", "new") for kind in ("head", "head_logits", "raw", "row_ids")} | {"group_ids"}


def bound_file(root, ref):
    check(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "native reference must have path and sha256")
    path = bound_path(root, ref["path"])
    check(file_hash(path) == ref["sha256"], "native input hash mismatch")
    return path


def strings(values):
    check(values.ndim == 1 and values.dtype.kind in "US", "opaque string row/group array required")
    result = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values.tolist()]
    check(all(result), "empty row/group identifier")
    return result


def load_export(root, reference, tolerance):
    receipt_path = bound_file(root, reference)
    receipt = read_json(receipt_path)
    complete = read_json(bound_path(receipt_path.parent, "COMPLETE.json"))
    check(receipt.get("schema") == "native-ofra-score-pair-v1" and receipt.get("status") == "COMPLETE" and
          complete.get("status") == "COMPLETE" and complete.get("receipt_sha256") == reference["sha256"] and
          complete.get("arrays") == receipt.get("arrays"), "incomplete or inconsistent native export receipt")
    check(receipt.get("evidence_kind") in ("synthetic", "real") and receipt.get("labels_read") is False and
          receipt.get("attributions_computed") is False, "unexpected native export scope")
    check(set(receipt["arrays"]) == EXPECTED_ARRAYS, "raw affinities or native head margins missing; never reconstruct from z scores")
    arrays = {name: np.load(bound_file(receipt_path.parent, ref), allow_pickle=False) for name, ref in receipt["arrays"].items()}
    old, new = receipt["checkpoints"]["old"], receipt["checkpoints"]["new"]
    reference_checkpoint = receipt["checkpoints"]["reference"]
    check(reference_checkpoint["checkpoint"] == 0 and old["checkpoint"] < new["checkpoint"] and
          old["dataset"] == new["dataset"] == reference_checkpoint["dataset"] and
          old["seed"] == new["seed"] == reference_checkpoint["seed"], "invalid native checkpoint chronology/dataset/seed")
    classes, old_classes = new["classes"], old["classes"]
    check(isinstance(classes, list) and isinstance(old_classes, list) and len(set(classes)) == len(classes) and
          len(set(old_classes)) == len(old_classes) and set(old_classes) < set(classes) and
          all(isinstance(c, str) and c for c in classes + old_classes), "invalid checkpoint class axes")
    rows = strings(arrays["new_row_ids"])
    check(rows == strings(arrays["old_row_ids"]) and len(set(rows)) == len(rows), "old/new row alignment mismatch")
    groups = strings(arrays["group_ids"])
    check(len(groups) == len(rows) == receipt["rows"], "native row/group count mismatch")
    for when, axis in (("old", old_classes), ("new", classes)):
        for kind in ("head", "head_logits", "raw"):
            value = arrays[f"{when}_{kind}"]
            check(value.dtype.kind == "f" and value.shape == (len(rows), len(axis)) and np.isfinite(value).all(), "invalid native score array")
        probabilities = arrays[f"{when}_head"]
        check(np.all((probabilities >= 0) & (probabilities <= 1)), "native head probabilities out of range")
        check(np.max(np.abs(sigmoid(arrays[f"{when}_head_logits"]) - probabilities)) <= tolerance,
              "native head margin/probability conversion failed declared tolerance; not a historical GPU certificate")
    return receipt, {"row_ids": rows, "group_ids": groups,
                     "old_head_logits": arrays["old_head_logits"].tolist(), "new_head_logits": arrays["new_head_logits"].tolist(),
                     "old_router_raw": arrays["old_raw"].tolist(), "new_router_raw": arrays["new_raw"].tolist()}


def adapt(spec_path, output):
    spec_path, output = Path(spec_path), Path(output)
    spec = read_json(spec_path)
    check(set(spec) == {"schema", "increment", "probability_margin_tolerance", "partitions"} and
          spec["schema"] == "native-score-repair-adapter-v1", "adapter specification schema mismatch")
    check(isinstance(spec["increment"], int) and spec["increment"] >= 0 and
          isinstance(spec["probability_margin_tolerance"], (int, float)) and
          np.isfinite(spec["probability_margin_tolerance"]) and spec["probability_margin_tolerance"] > 0,
          "explicit increment and margin/probability tolerance required")
    check(set(spec["partitions"]) == {"trigger", "fit", "acceptance", "subsequent"}, "four exported partitions required")
    reject_links(output)
    check(not output.exists(), "adapter output must be a fresh directory")
    loaded, reference, kinds = {}, None, set()
    for role in ("trigger", "fit", "acceptance", "subsequent"):
        entry = spec["partitions"][role]
        check(set(entry) == {"receipt", "labels", "origin", "observed_increment", "labels_available_increment"}, "partition adapter fields must be exact")
        if role != "subsequent":
            check(entry["origin"] == "increment-available-training", "official test or held-out labels cannot trigger, fit or accept a repair")
        else:
            check(entry["origin"] in ("prospective-heldout", "retrospective-official-test"), "explicit held-out evaluation origin required")
        receipt, part = load_export(spec_path.parent, entry["receipt"], spec["probability_margin_tolerance"])
        kinds.add(receipt["evidence_kind"])
        if reference is None:
            reference = receipt["checkpoints"]
        check(receipt["checkpoints"] == reference, "different model/class/seed/dataset checkpoint lineage across partitions")
        labels = read_json(bound_file(spec_path.parent, entry["labels"]))
        check(set(labels) == {"row_ids", "group_ids", "labels", "origin"} and labels["origin"] == entry["origin"], "label-source origin mismatch")
        check(labels["row_ids"] == part["row_ids"] and labels["group_ids"] == part["group_ids"] and
              len(labels["labels"]) == len(part["row_ids"]), "explicit label alignment required")
        y = [str(x) for x in labels["labels"]]
        check(set(y) == set(reference["new"]["classes"]), "missing/unknown labels in native partition")
        part["role"] = role
        if role != "subsequent":
            part["labels"] = y
        loaded[role] = (entry, receipt, part, y)
    check(len(kinds) == 1, "cannot mix synthetic and real export lineages")
    manifest = {"schema": "score-repair-bundle-v1", "increment": spec["increment"],
                "source_kind": "synthetic-software-test" if kinds == {"synthetic"} else "score-bundle-pending-fidelity",
                "classes": reference["new"]["classes"], "old_classes": reference["old"]["classes"], "partitions": {},
                "native_export_receipts": {}, "adapter_schema": spec["schema"], "adapter_spec_sha256": file_hash(spec_path),
                "historical_gpu_fidelity_verified_by_adapter": False, "attributions_computed_by_adapter": False}
    # Validate all metadata and disjointness before creating outputs.
    for role, (entry, receipt, part, y) in loaded.items():
        manifest["partitions"][role] = {"file": f"{role}.json", "sha256": "0" * 64, "row_ids": part["row_ids"],
                                        "group_ids": part["group_ids"], "observed_increment": entry["observed_increment"],
                                        "labels_available_increment": entry["labels_available_increment"], "origin": entry["origin"]}
        manifest["native_export_receipts"][role] = {"receipt_sha256": entry["receipt"]["sha256"],
                                                   "manifest_sha256": receipt["manifest_sha256"],
                                                   "historical_gpu_fidelity_verified": receipt["historical_gpu_fidelity_verified"],
                                                   "training_lineage_verified": receipt["training_lineage_verified"]}
    manifest["sealed_labels_sha256"] = "0" * 64
    validate_manifest(manifest)
    output.mkdir(parents=True)
    for role, (entry, receipt, part, y) in loaded.items():
        path = output / f"{role}.json"
        save_json(path, part)
        manifest["partitions"][role]["sha256"] = file_hash(path)
        if role == "subsequent":
            label_path = output / "sealed-evaluation-labels.json"
            save_json(label_path, {"row_ids": part["row_ids"], "group_ids": part["group_ids"], "labels": y,
                                   "observed_increment": entry["observed_increment"]})
            manifest["sealed_labels_sha256"] = file_hash(label_path)
    save_json(output / "manifest.json", manifest)
    save_json(output / "ADAPTER_RECEIPT.json", {"status": "COMPLETE_SCORE_ADAPTATION_ONLY", "manifest_sha256": file_hash(output / "manifest.json"),
                                               "source_binding_sha256": source_binding(manifest), "source_kind": manifest["source_kind"],
                                               "real_experiment_authorized": False})
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("spec", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = adapt(args.spec, args.output)
    print(result["source_kind"])
