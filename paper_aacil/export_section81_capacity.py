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
    assert master["completed"] and master["cell_count"] == 20
    assert master["job_id"] == "490920" and not master["smoke"]
    expected = {(m, s) for m in ("ar", "replay_bs", "freq", "wacb") for s in (1, 2, 3, 4, 42)}
    assert {(c["method"], c["seed"]) for c in master["cells"]} == expected
    outputs = {}
    for cell in master["cells"]:
        name = f'{cell["method"]}_seed{cell["seed"]}'
        root = args.input / name
        raw = root / "cell/result.json"
        result = read(raw)
        summary = read(root / "summary.json")
        assert cell["exit_code"] == 0 and summary == cell["summary"]
        assert result["completed"] and summary["scientific_result"]
        assert not summary["profile"] and not summary["smoke"]
        assert len(result["checkpoints"]) == 5
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
            "scope": "UNSW-NB15 three-hidden-layer width-512 capacity sensitivity; fixed-FPR view, not default decision.",
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
