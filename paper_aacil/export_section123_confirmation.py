"""Export completed fixed confirmation cells without rerunning inference."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix-root", type=Path, required=True)
    p.add_argument("--single-root", type=Path, action="append", default=[])
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    terminal = read(a.matrix_root / "job_exit.json")
    aggregate = read(a.matrix_root / "summary.json")
    assert terminal["completed"] and terminal["error"] is None and aggregate["completed"]
    roots = [(a.matrix_root / f"{c['dataset'].lower()}_w128_{c['method']}_seed{c['seed']}", c["elapsed_seconds"])
             for c in aggregate["cells"]]
    for root in a.single_root:
        job = read(root / "job_exit.json")
        assert job["status"] == "complete" and job["exit_code"] == 0
        roots.append((root / "run", job["elapsed_seconds"]))
    prepared, seen = [], set()
    for root, elapsed in roots:
        summary = read(root / "summary.json")
        assert summary["completed"] and summary["scientific_result"] and not summary["smoke"]
        key = (summary["dataset"], summary["method"], summary["seed"])
        assert key not in seen
        seen.add(key)
        raw_path = root / "cell/result.json"
        raw = read(raw_path)
        assert raw["completed"] and raw["identity"]["seed"] == key[2]
        assert raw["identity"]["dataset"] == key[0]
        assert raw["identity"]["git_commit"] == summary["source_commit"]
        assert len(raw["checkpoints"]) == summary["checkpoint_count"]
        for cp in raw["checkpoints"]:
            assert sha(root / "cell" / cp["checkpoint_file"]) == cp["checkpoint_sha256"]
        body = dict(raw, execution_summary=summary, completed_attempt_elapsed_seconds=elapsed)
        for name in ("default_stages", "calibrated_stages"):
            path = root / (name + ".json")
            if path.exists():
                body[name] = read(path)
                assert len(body[name]) == len(raw["checkpoints"])
        if key[1] != "ar_joint":
            assert "default_stages" in body
        result = public(body, sha(raw_path), "cell/result.json")
        result["publication_derivation"]["original_json"] = "section123/confirmation/cell/result.json"
        result["publication_derivation"]["additional_json_sha256"] = {
            f.name: sha(f) for f in (root / "summary.json", root / "default_stages.json", root / "calibrated_stages.json") if f.exists()}
        result["publication_derivation"]["scope"] = "Single confirmation cell; no cross-method analysis. Attempt time includes training/evaluation; stage times are nested."
        dest = a.output_root / key[0].lower() / f"{key[1]}_seed{key[2]}.json"
        assert not dest.exists(), dest
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        for forbidden in ("C:\\", "/scr/", "/home/", "leon12138"):
            assert forbidden not in text, forbidden
        prepared.append((dest, text))
    assert len(seen) == 50
    for dest, text in prepared:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print("Exported 50 unique completed confirmation cells; no recomputation")


if __name__ == "__main__":
    main()
