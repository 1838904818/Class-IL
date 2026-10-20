"""Export a complete Section57 development dataset without model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import SECTION, private_fields, read, sanitize, write_derived
from export_section46_development import COUNTS, PACKAGE_SHA, SEEDS, WRAPPER_COMMIT, read_shared, require

METHOD = "aacil_akd_wa"
DATASETS = ("UNSW-NB15", "NSL-KDD", "CIC-IDS-2017")
LAUNCHER = "ae705f199b4cb059b9743ae6444c1bd73c9ab094"
LAUNCHER_SHA = "f582dbcc44b17d3aeb32466ecc25e832ef2d9e87bfaa846930dab12b9555845e"
ENTRY_SHA = "f9abc725d49a01d709633dadbb2b478d827ebd247eacfc732c1b2b5ebc8349be"


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def publication(value, digest, path, root):
    # This entrypoint records an absolute data root inside effective_config.
    # Explicitly disclose this derived identity; never pretend it is byte-identical
    # or replace the original execution hashes referenced by saved checkpoints.
    changes = []
    body = {k: v for k, v in value.items() if k != "identity"}
    result = sanitize(body, changed=changes)
    identity = value.get("identity")
    if identity is not None:
        identity_changes = []
        result["identity"] = sanitize(identity, changed=identity_changes)
        changes.extend("identity." + k for k in identity_changes)
    result["publication_derivation"] = {
        "original_json": "section57_development/" + path.relative_to(root).as_posix(),
        "original_json_sha256": digest,
        "redacted_fields": changes,
        "original_identity_canonical_sha256": canonical(identity) if identity else None,
        "published_identity_canonical_sha256": canonical(result["identity"]) if identity else None,
        "execution_hashes_preserved": True,
        "identity_hash_semantics": "Saved identity/config hashes refer to the original private execution identity, not its path-redacted public representation.",
        "changes": "Only private command/path metadata is normalized, including the effective_config data root; numeric measurements and saved execution hashes are unchanged.",
        "checkpoint_reference_base": "Original execution directory; weights are not exported.",
    }
    require(not list(private_fields(result)), "Private metadata remains")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    paths = sorted((root / "attempts").glob("*/status.json"))
    require(len(paths) == 1, "Multiple attempts require explicit cost reconciliation")
    job, snapshot = read_shared(paths[0])
    planned = [[d, s] for d in DATASETS for s in SEEDS]
    require(job["planned_cells"] == planned and job["smoke"] is False
            and job["resume"] is False and job["status"] in ("running", "complete"),
            "First formal Section57 assignment required")
    bindings = {"launcher_commit": LAUNCHER, "launcher_sha256": LAUNCHER_SHA,
                "training_entrypoint_sha256": ENTRY_SHA,
                "dependency_package_sha256": PACKAGE_SHA,
                "dependency_wrapper_commit": WRAPPER_COMMIT}
    require(all(job.get(k) == v for k, v in bindings.items()), "Assignment source mismatch")
    cells = job["cells"]
    require(len(cells) <= 15 and [[c["dataset"], c["seed"]] for c in cells] == planned[:len(cells)]
            and all(c["method"] == METHOD and c["exit_code"] == 0 for c in cells),
            "Successful serial prefix required")
    chosen = [c for c in cells if c["dataset"] == args.dataset]
    require([c["seed"] for c in chosen] == list(SEEDS)
            and all(c.get("ended_at") for c in chosen)
            and (job.get("current_cell") is None or job["current_cell"]["dataset"] != args.dataset),
            "Requested five-seed dataset is not complete")
    slug = args.dataset.lower().replace("-", "_")
    destinations = [SECTION / "publication_v1" / slug / f"{METHOD}_seed{s}" for s in SEEDS]
    require(not any(p.exists() for p in destinations), "Already exported; never replace status snapshots")
    prepared, common = [], None
    for cell, destination in zip(chosen, destinations):
        seed = cell["seed"]
        folder = root / slug / "cells" / METHOD / f"seed_{seed}"
        invocation, inv_hash = read(folder / "section57_local_invocation.json")
        require(all(invocation.get(k) == v for k, v in bindings.items())
                and invocation["actual_entrypoint"] == "ncd/run_section57_local.py"
                and invocation["training_entrypoint"] == "ncd/run_aacil_akd.py"
                and (invocation["dataset"], invocation["method"], invocation["seed"]) == (args.dataset, METHOD, seed)
                and invocation["smoke"] is False and invocation["selection_changed"] is False
                and invocation["CSE18_model_metrics_accessed"] is False,
                "Invocation mismatch")
        marker, marker_hash = read(folder / "section57_result.json")
        matrix, matrix_hash = read(folder / "raw/result.json")
        require(marker["completed"] is True and marker["smoke"] is False
                and marker["invocation"] == invocation and marker["raw_matrix_sha256"] == matrix_hash
                and matrix["execution_status"] == "complete" and len(matrix["cells"]) == 1
                and matrix["expected"] == {"datasets": [args.dataset], "methods": [METHOD], "seeds": [seed]},
                "Completion binding mismatch")
        path = folder / "raw" / slug / f"{METHOD}_seed{seed}" / "result.json"
        raw, raw_hash = read(path)
        ident, cfg = raw["identity"], raw["identity"]["effective_config"]
        require(matrix["cells"][0]["result_sha256"] == raw_hash and raw["completed"] is True
                and raw["selection_only"] is False
                and (ident["dataset"], ident["method"], ident["seed"]) == (args.dataset, METHOD, seed)
                and ident["git_commit"] == LAUNCHER
                and ident["source_hashes"]["ncd/run_aacil_akd.py"] == ENTRY_SHA,
                "Raw identity mismatch")
        fixed = {"lambda_g": .5, "derpp_alpha": 1., "derpp_beta": 1., "hidden": 128,
                 "batch_size": 256, "epochs_per_task": 8, "learning_rate": .001,
                 "buffer_per_class": 200, "min_steps_per_task": 2000,
                 "normal_training_rows": 50000, "cpu_threads": 2, "device": "cuda",
                 "section57": True, "smoke": False, "selection_only": False,
                 "post_unsealing_design": True, "independent_dataset_confirmation": False,
                 "gate_kd_mask_policy": "replayed_old_attack_only",
                 "gate_kd_benign": False, "gate_kd_current_new_attack": False}
        require(all(cfg.get(k) == v for k, v in fixed.items()), "Fixed config mismatch")
        for split, count in zip(("train", "test"), COUNTS[args.dataset]):
            require(ident["data_binding"]["x_" + split]["shape"][0] == count
                    and ident["data_binding"]["y_" + split]["shape"][0] == count, "Full data required")
        shared = {k: ident[k] for k in ("source_hashes", "effective_config_sha256",
                                      "data_binding", "tasks", "class_order")}
        common = shared if common is None else common
        require(shared == common and raw["tasks"] == ident["tasks"]
                and len(raw["checkpoints"]) == len(raw["tasks"]), "Cross-seed binding mismatch")
        exported = publication(raw, raw_hash, path, root)
        exported["section57_execution"] = {
            "assignment_status_snapshot_sha256": snapshot, "assignment_attempt": paths[0].parent.name,
            "assignment_started_at": job["started_at"], "assignment_ended_at": job.get("ended_at"),
            "assignment_status_at_export": job["status"], "dataset_completed_at": chosen[-1]["ended_at"],
            "cell_attempt": {k: v for k, v in cell.items() if k != "command"},
            "local_invocation": sanitize(invocation), "local_invocation_sha256": inv_hash,
            "completion_marker_sha256": marker_hash, "raw_matrix_sha256": matrix_hash,
            "complete_dataset_only": True, "other_dataset_metrics_read": False,
            "training_inference_statistics_recomputed": False,
            "descriptive_development_dataset": True, "selection_changed": False,
            "cost_scope": "Raw and parent times are nested, not additive; local GPU includes desktop activity.",
        }
        prepared.append((destination / "result.json", exported))
        for index, cp in enumerate(raw["checkpoints"]):
            require(cp["task_index"] == index and cp["primary_alpha"] == .02
                    and cp["checkpoint_file"] == f"checkpoints/checkpoint_{index:03d}.pt"
                    and cp["gate_kd_mask_policy"] == "replayed_old_attack_only"
                    and cp["gate_kd_benign_rows"] == cp["gate_kd_new_attack_rows"] == 0
                    and cp["effective_lambda_g"] == .5
                    and all("confusion_matrix" in cp["operating_points"][a] for a in ("0.01", "0.02", "0.05")),
                    "Saved trajectory mismatch")
            sidepath = path.parent / Path(cp["checkpoint_file"]).with_suffix(".json")
            sidecar, digest = read(sidepath)
            require(sidecar == cp, "Sidecar mismatch")
            prepared.append((destination / sidepath.relative_to(path.parent),
                             publication(sidecar, digest, sidepath, root)))
    require(all(not list(private_fields(v)) for _, v in prepared), "Private publication metadata")
    for path, value in prepared:
        write_derived(path, value)
    print(json.dumps({"dataset": args.dataset, "json_files": len(prepared),
                      "training_inference_statistics_recomputed": False}))


if __name__ == "__main__":
    main()
