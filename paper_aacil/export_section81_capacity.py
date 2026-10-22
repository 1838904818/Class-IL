"""Export completed capacity cells without private paths or row identifiers."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    master = read(args.input / "summary.json")
    assert master["completed"] and not master["smoke"]
    assert master["job_id"] in ("490920", "491011")
    remaining = master["job_id"] == "491011"
    matrix = [("UNSW-NB15", 1024), ("CIC-IDS-2017", 512), ("CIC-IDS-2017", 1024)] if remaining else [("UNSW-NB15", 512)]
    expected = {(d, w, m, s) for d, w in matrix for m in ("ar", "replay_bs", "freq", "wacb") for s in (1, 2, 3, 4, 42)}
    assert len(master["cells"]) == master["cell_count"] == len(expected)
    assert {(c.get("dataset", master["dataset"]), c.get("width", master["width"]), c["method"], c["seed"]) for c in master["cells"]} == expected
    outputs = {}
    for cell in master["cells"]:
        name = f'{cell["method"]}_seed{cell["seed"]}'
        dataset, width = cell.get("dataset", master["dataset"]), cell.get("width", master["width"])
        if remaining:
            name = f'{dataset.lower()}_w{width}_{name}'
        root = args.input / name
        raw = root / "cell/result.json"
        result = read(raw)
        summary = read(root / "summary.json")
        assert cell["exit_code"] == 0 and summary == cell["summary"]
        assert result["completed"] and summary["scientific_result"]
        assert not summary["profile"] and not summary["smoke"]
        assert len(result["checkpoints"]) == len(result["tasks"])
        assert summary["dataset"] == dataset and summary["width"] == width
        assert summary["job_id"] == master["job_id"]
        if remaining:
            assert summary["source_commit"] == "c339d49a7014cc264c3b2990a24e9d4ee75b00d5"
        external = cell["method"] in ("freq", "wacb")
        stage_path = root / "calibrated_stages.json" if external else raw
        stages = read(stage_path) if external else result["checkpoints"]
        config = dict(result["identity"]["effective_config"])
        config["problem_scout_data_root"] = "<prepared-data-root>"
        exported = {
            "schema_version": "section81_capacity_public_v1",
            "summary": summary,
            "raw_result_sha256": digest(raw),
            "raw_summary_sha256": digest(root / "summary.json"),
            "stage_source_sha256": digest(stage_path),
            "effective_config": config,
            "tasks": result["tasks"], "class_order": result["class_order"],
            "stages": [{"task_index": cp["task_index"],
                        "checkpoint_sha256": result["checkpoints"][i]["checkpoint_sha256"],
                        "operating_points": cp["operating_points"]}
                       for i, cp in enumerate(stages)],
            "scope": f"{dataset} three-hidden-layer width-{width} capacity sensitivity; fixed-FPR view, not default decision.",
        }
        text = json.dumps(exported, indent=2, sort_keys=True, allow_nan=False) + "\n"
        for forbidden in ("/home/", "/scr/", "C:\\", "row_ids", "leon12138"):
            assert forbidden not in text, forbidden
        outputs[name + ".json"] = text
    assert not args.output.exists(), "Refusing to overwrite an existing export"
    args.output.mkdir(parents=True)
    for name, text in outputs.items():
        with (args.output / name).open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(f"Exported {len(outputs)} cells; no training or inference")


if __name__ == "__main__":
    main()
