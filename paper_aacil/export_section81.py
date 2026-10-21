"""Export one completed B1 ER cell; no inference or aggregate recomputation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import private_fields, read, sanitize, write_derived
from export_section46_development import require
from export_section57 import canonical

ROOT = Path(__file__).resolve().parents[1]


def public(value, digest, relative):
    changes = []
    result = sanitize({k: v for k, v in value.items() if k != "identity"}, changed=changes)
    if "identity" in value:
        identity_changes = []
        result["identity"] = sanitize(value["identity"], changed=identity_changes)
        changes.extend("identity." + k for k in identity_changes)
    result["publication_derivation"] = {
        "original_json": "section81/" + relative,
        "original_json_sha256": digest, "redacted_fields": changes,
        "original_identity_canonical_sha256": canonical(value.get("identity")),
        "published_identity_canonical_sha256": canonical(result.get("identity")),
        "execution_hashes_preserved": True,
        "identity_hash_semantics": "Saved hashes refer to the original execution identity, not the path-redacted public identity.",
        "training_inference_statistics_recomputed": False,
    }
    require(not list(private_fields(result)), "Private metadata remains")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    job, job_hash = read(source / "job_exit.json")
    attempt, attempt_hash = read(source / "run/attempt.json")
    require(job["status"] == "complete" and job["exit_code"] == 0
            and attempt["status"] == "complete" and attempt["smoke"] is False,
            "Completed formal cell required")
    raw, raw_hash = read(source / "run/cell/result.json")
    require(raw["completed"] is True and raw_hash == job["result_sha256"]
            == attempt["result_sha256"], "Result completion mismatch")
    identity = raw["identity"]
    cfg = identity["effective_config"]
    require(identity["section"] == 81 and identity["scientific_result"] is True
            and identity["method"] == job["method"] == "er_bal_tabt"
            and identity["dataset"] == job["dataset"] == attempt["dataset"]
            and identity["seed"] == job["seed"] == attempt["seed"]
            and identity["memory_setting"] == job["memory_setting"] == attempt["memory_setting"],
            "B1 identity mismatch")
    require(identity["dataset"] in ("UNSW-NB15", "CIC-IDS-2017")
            and identity["seed"] in (1, 2, 3, 4, 42), "Unregistered cell")
    fixed = dict(device="cuda", smoke=False, epochs_per_task=100, batch_size=256,
                 optimizer="AdamW", learning_rate=5e-4, weight_decay=1e-5,
                 normal_training_rows=50000, cpu_threads=2)
    require(all(cfg.get(k) == v for k, v in fixed.items()), "ER schedule mismatch")
    require(attempt["provenance"]["effective_config"] == cfg
            and attempt["provenance"]["source_hashes"] == identity["source_hashes"],
            "Attempt source mismatch")
    commit = identity["git_commit"]
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "Invalid source commit")
    for name, expected in identity["source_hashes"].items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == expected
                or hashlib.sha256(blob.replace(b"\n", b"\r\n")).hexdigest() == expected,
                "Published source mismatch: " + name)
    slug = identity["dataset"].lower().replace("-", "_")
    destination = ROOT / "ncd/results/problem_scout/section81/publication_v1" / slug / identity["memory_setting"] / f"er_bal_tabt_seed{identity['seed']}"
    require(not destination.exists(), "Cell already exported")
    prepared = []
    require(len(raw["checkpoints"]) == len(identity["tasks"]), "Incomplete trajectory")
    for index, cp in enumerate(raw["checkpoints"]):
        rel = f"checkpoints/checkpoint_{index:03d}.pt"
        require(cp["task_index"] == index and cp["checkpoint_file"] == rel
                and cp["identity_hash"] == canonical(identity)
                and all(alpha in cp["operating_points"] for alpha in ("0.01", "0.02")),
                "Checkpoint identity mismatch")
        sidepath = Path(rel).with_suffix(".json")
        side, side_hash = read(source / "run/cell" / sidepath)
        require(side == cp, "Sidecar mismatch")
        prepared.append((destination / sidepath, public(side, side_hash, sidepath.as_posix())))
    result = public(raw, raw_hash, "result.json")
    result["section81_execution"] = {
        "job_exit": sanitize(job | {"command": subprocess.list2cmdline(job["command"])}), "job_exit_sha256": job_hash,
        "attempt": sanitize(attempt | {"command": subprocess.list2cmdline(attempt["command"])}), "attempt_sha256": attempt_hash,
        "cost_scope": "Parent, child and stage costs are nested, not additive. Local GPU includes desktop activity.",
        "single_seed_descriptive_only": True, "five_seed_inference_performed": False,
    }
    prepared.append((destination / "result.json", result))
    require(all(not list(private_fields(v)) for _, v in prepared), "Private output metadata")
    for path, value in prepared:
        write_derived(path, value)
    print(json.dumps({"exported_json_files": len(prepared), "method": identity["method"],
                      "dataset": identity["dataset"], "seed": identity["seed"]}))


if __name__ == "__main__":
    main()
