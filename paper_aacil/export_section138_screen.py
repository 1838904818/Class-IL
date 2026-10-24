"""Export one completed screening cell, never sealed confirmation outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_section81 import public


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(confirmation=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell-root", type=Path, required=True)
    p.add_argument("--attempt-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    summary = read(a.cell_root / "summary.json")
    # Check the role before opening the result; sealed outputs are never read.
    role = "confirmation" if confirmation else "screen"
    if confirmation:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ncd.section147_selection import SELECTED, SELECTION_COMMIT
        assert summary["method"] in SELECTED
        assert summary["seed"] in range(10, 15)
    else:
        assert summary["seed"] in (1, 2)
    assert summary["output_role"] == role
    assert summary["completed"] and summary["scientific_result"]
    raw_path = a.cell_root / "cell/result.json"
    raw = read(raw_path)
    identity = raw["identity"]
    attempt = read(a.attempt_root / "exit.json")
    assert raw["completed"] and raw["scientific_result"] and not identity["smoke"]
    assert identity["output_role"] == role and identity["seed"] == summary["seed"]
    if confirmation:
        assert identity["selection_commit"] == SELECTION_COMMIT
    assert identity["dataset"] == summary["dataset"]
    assert identity["dataset"] == attempt["dataset"]
    assert identity["method"] == summary["method"] == attempt["configuration"]
    assert identity["git_commit"] == summary["source_commit"] == attempt["git_commit"]
    assert attempt["seed"] == identity["seed"] and attempt["exit_code"] == 0
    assert attempt["guard_reason"] is None
    assert len(raw["checkpoints"]) == len(raw["tasks"]) == summary["checkpoints"]
    for index, checkpoint in enumerate(raw["checkpoints"]):
        assert checkpoint["task_index"] == index
        rel = f"checkpoints/checkpoint_{index:03d}.pt"
        assert checkpoint["checkpoint_file"] == rel
        assert sha(a.cell_root / "cell" / rel) == checkpoint["checkpoint_sha256"]
    destination = a.output_root / summary["dataset"].lower() / f"{summary['method']}_seed{summary['seed']}.json"
    assert not destination.exists(), "Already exported; do not re-export"
    body = public(dict(raw, execution_summary=summary), sha(raw_path), "result.json")
    body["publication_derivation"]["original_json"] = f"section138/{role}/cell/result.json"
    body["execution_cost"] = {k: attempt[k] for k in (
        "elapsed_seconds", "sampled_peak_rss_bytes", "exit_code", "guard_reason")}
    body["execution_cost"]["attempt_sha256"] = sha(a.attempt_root / "exit.json")
    body["execution_cost"]["scope"] = "Completed parent attempt; stage timings are nested, not additive"
    text = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for forbidden in ("C:\\", "/scr/", "/home/", "leon12138"):
        assert forbidden not in text
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(destination.name)


if __name__ == "__main__":
    main()
