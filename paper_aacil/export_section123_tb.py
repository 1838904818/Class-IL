"""Publish one completed TB cell, retaining schedules and both decision views."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous-attempt", type=Path)
    a = p.parse_args()
    job = read(a.source / "job_exit.json")
    summary = read(a.source / "run/summary.json")
    raw_path = a.source / "run/cell/result.json"
    raw = read(raw_path)
    assert job["status"] == "complete" and job["exit_code"] == 0
    assert summary["completed"] and summary["scientific_result"] and not summary["smoke"]
    assert raw["completed"] and raw["scientific_result"]
    identity = raw["identity"]
    assert identity["method"] in ("ar_es", "replay_bs_es")
    assert identity["seed"] in (1, 2, 3, 4, 42)
    assert len(raw["checkpoints"]) == len(raw["tasks"])
    for i, cp in enumerate(raw["checkpoints"]):
        assert cp["task_index"] == i
        checkpoint = a.source / "run/cell" / cp["checkpoint_file"]
        assert sha(checkpoint) == cp["checkpoint_sha256"]
        assert read(checkpoint.with_suffix(".json")) == cp
        assert set(cp["operating_points"]) >= {"0.01", "0.02"}
        assert cp["default_decision"]
    config = dict(identity["effective_config"])
    config["problem_scout_data_root"] = "<prepared-data-root>"
    public_identity = {k: identity[k] for k in ("git_commit", "dataset", "method", "seed", "source_hashes")}
    public_identity["effective_config"] = config
    result = {k: v for k, v in raw.items() if k != "identity"}
    result["identity"] = public_identity
    result["publication"] = dict(raw_result_sha256=sha(raw_path), summary=summary,
        completed_attempt_elapsed_seconds=job["elapsed_seconds"],
        scope="Single TB cell; no schedule-selection rule or five-seed test applied.",
        forgetting_definition="Both views use equal-weight mean over all old classes including Normal of best pre-final recall minus final recall, signed and unclipped; values are fractions (multiply by 100 for percentage points). Default uses uncalibrated decisions; operating points use calibrated fixed-FPR decisions, so values differ.",
        cost_scope="Stage and completed-attempt times are nested; prior failed attempts are separate costs.")
    if a.previous_attempt:
        prior = read(a.previous_attempt / "job_exit.json")
        assert prior["status"] == "failed"
        assert all(prior[k] == job[k] for k in ("dataset", "method", "seed"))
        result["publication"]["previous_failed_attempt"] = dict(
            elapsed_seconds=prior["elapsed_seconds"], error=prior["error"],
            job_exit_sha256=sha(a.previous_attempt / "job_exit.json"))
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Retain provenance hashes, never row identities; use an unambiguous public key.
    text = text.replace('"row_ids_sha256":', '"row_set_sha256":')
    for forbidden in ("/home/", "/scr/", "C:\\", "row_ids", "leon12138"):
        assert forbidden not in text, forbidden
    assert not a.output.exists()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("Exported one completed TB cell")


if __name__ == "__main__":
    main()
