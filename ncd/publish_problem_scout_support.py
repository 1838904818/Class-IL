"""Create portable, hash-linked copies of Section21 attempt-cost evidence."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("scout_merge", Path(__file__).with_name("merge_problem_scout.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def publish(evidence, section, output):
    source = evidence / "section21_attempt_cost_accounting.json"
    costs = m.read_json(source)
    merged = m.read_json(section / "merged_v1/result.json")
    by_raw_sha = {e["original_result_sha256"]: e for e in merged["cells"]}
    files = {}

    def save_copy(path, relative):
        relative = m.safe_relative(relative).as_posix()
        raw = path.read_bytes()
        data = m.redact_log(raw.decode("utf-8")).encode()
        m.scan_portable(data.decode())
        m.require(relative not in files or files[relative] == data, "Conflicting evidence destination")
        files[relative] = data
        return {"published_file": relative, "published_sha256": m.sha_bytes(data), "original_sha256": m.sha_bytes(raw)}

    def walk(value):
        if isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, dict):
            if "logical_path" in value and "sha256" in value:
                logical = value["logical_path"]
                if logical.startswith("protected_raw/"):
                    relative = logical.removeprefix("protected_raw/")
                    path = evidence / m.safe_relative(relative)
                    m.require(m.sha_file(path) == value["sha256"], "Evidence original hash mismatch")
                    value.update(save_copy(path, "evidence/" + relative))
                elif logical.startswith("section21/"):
                    relative = logical.removeprefix("section21/")
                    path = section / m.safe_relative(relative)
                    m.require(m.sha_file(path) == value["sha256"], "Local cost original hash mismatch")
                    if value["sha256"] in by_raw_sha:
                        entry = by_raw_sha[value["sha256"]]
                        value.update(published_file="../merged_v1/" + entry["result"], published_sha256=entry["result_sha256"])
                    else:
                        value.update(published_file="../" + relative, published_sha256=value["sha256"])
                else:
                    raise ValueError("Unknown cost source prefix")
            for child in list(value.values()):
                walk(child)

    walk(costs)
    receipts = []
    for filename in ("section21_source_data_validation_20261015.log", "independent_g1_g7_20261015.log", "section21_merged_validation_20261015.log"):
        receipts.append(save_copy(evidence / filename, "validation/" + filename))
    costs["publication"] = {"original_accounting_sha256": m.sha_file(source), "publisher_script_sha256": m.sha_file(Path(__file__)),
                            "merged_index_sha256": m.sha_file(section / "merged_v1/result.json"),
                            "redaction": "User-specific operational roots/account name in evidence copies only; original hashes retained", "validation_receipts": receipts}
    files["attempt_costs.json"] = m.json_bytes(costs)
    for relative, data in files.items():
        m.scan_portable(data.decode())
        destination = output / relative
        m.require(not destination.exists() or destination.read_bytes() == data, "Refusing differing existing support artifact")
    if output.exists():
        m.require({p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} <= set(files), "Unexpected files in support output")
    for relative, data in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
    m.require(all(m.sha_file(output / p) == m.sha_bytes(data) for p, data in files.items()), "Support file hash mismatch")
    return {"status": "PASS", "files": len(files), "bytes": sum(map(len, files.values())), "accounting_sha256": m.sha_file(output / "attempt_costs.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--section-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.evidence_root, args.section_root, args.output), indent=2))
