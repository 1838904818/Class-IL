"""Publish the fixed first-valid Section21 cells without weights or raw data.

Inputs are immutable execution artifacts. Selection is fixed by package identity,
never by scores. The output records original hashes and only redacts operational
command/log locations; checkpoint records and scientific identities are unchanged.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

SEEDS = (1, 2, 3, 4, 42)
METHODS = ("joint", "replay", "icarl", "replay_lwf", "rcl_tab", "derpp")
DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018")
EFFECTIVE_SHA = "8c4753a3605cdd7536fa7e03e71e82fc0af9668244e080ed182a4ce34c7e1f1f"
SOURCES = {"ab33ca61747b0e499c4006b02e4eeda29589d7f3", "1f35f6587bb63aba85c43b0654d15709d46fc3d8"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    return value


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def safe_relative(name):
    path = Path(name)
    require(not path.anchor and ".." not in path.parts and ":" not in name, "Unsafe relative artifact path")
    return path


def row_key(row):
    return (row["dataset"], row["method"], int(row["seed"]), row["split"], int(row["class_id"]))


def check_csv(path, rows):
    with path.open(newline="", encoding="utf-8") as stream:
        actual = list(csv.DictReader(stream))
    require(len(actual) == len(rows), "CSV row count mismatch")
    require(len({row_key(r) for r in actual}) == len(rows), "Duplicate CSV row")
    actual = {row_key(r): r for r in actual}
    for expected in rows:
        observed = actual[row_key(expected)]
        require(set(observed) == set(expected), "CSV field mismatch")
        for key, value in expected.items():
            if isinstance(value, (int, float)):
                require(math.isclose(float(observed[key]), value, rel_tol=1e-12, abs_tol=1e-12), "CSV number mismatch")
            else:
                require(observed[key] == ("" if value is None else value), "CSV value mismatch")


def redact_log(text):
    """Redact roots and optional comma-separated NCD_PUBLICATION_REDACT_USERS."""
    text = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", "<LOCAL_USER_HOME>", text, flags=re.I)
    text = re.sub(r"/(home|scr)/user/[^/\s\"']+", lambda m: "<HPC_HOME>" if m[1] == "home" else "<HPC_SCRATCH>", text)
    # Some commands store a relative path to scratch without a leading slash.
    text = re.sub(r"(?<![A-Za-z])(?:home|scr)/user/[^/\s\"']+", "<HPC_USER_STORAGE>", text)
    for user in filter(None, os.environ.get("NCD_PUBLICATION_REDACT_USERS", "").split(",")):
        text = re.sub(r"\b" + re.escape(user.strip()) + r"\b", "<HPC_USER>", text)
    return text


def scan_portable(text):
    require(not re.search(r"[A-Za-z]:[\\/]+Users[\\/]|/(?:home|scr)/user/", text, re.I), "Personal path in publication")
    for user in filter(None, os.environ.get("NCD_PUBLICATION_REDACT_USERS", "").split(",")):
        require(not re.search(r"\b" + re.escape(user.strip()) + r"\b", text), "User identifier in publication")
    require(not re.search(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}", text), "Secret-like material in publication")


def selected_packages(local, hpc):
    packages = [("local_tabular", local / "formal_v1", False, {"NSL-KDD", "UNSW-NB15"}, set(SEEDS))]
    for short, dataset, remote_seeds, local_seeds in (
        ("cic17", "CIC-IDS-2017", (1, 2, 3), (4, 42)),
        ("cic18", "CSE-CIC-IDS2018", (1, 2, 3, 4), (42,)),
    ):
        for seed in remote_seeds:
            packages.append((f"hpc_{short}_seed{seed}", hpc / short / f"seed_{seed}" / "all", True, {dataset}, {seed}))
        for seed in local_seeds:
            packages.append((f"local_{short}_seed{seed}", local / "local_tail_v1" / short / f"seed_{seed}" / "all", False, {dataset}, {seed}))
    return packages


def remote_weight_evidence(evidence_root):
    found = {}
    for path in sorted(evidence_root.glob("*.log")):
        content = path.read_text(encoding="utf-8", errors="replace")
        for digest, remote_path in re.findall(r"([0-9a-f]{64})\s+(/[^\r\n]+)", content):
            if "/results/formal/" not in remote_path or not remote_path.endswith(".pt"):
                continue
            key = remote_path.split("/results/formal/", 1)[1]
            found.setdefault((key, digest), []).append((path.name, sha_file(path)))
    return found


def merge(local, hpc, evidence_root, output):
    files, entries, all_rows, proofs, packages = {}, [], [], [], []
    identities = set()
    remote_proofs = remote_weight_evidence(evidence_root)
    evidence_coverage = Counter(name for values in remote_proofs.values() for name, _ in values)
    published_evidence = {}

    def put(relative, data):
        relative = safe_relative(relative).as_posix()
        require(relative not in files, f"Duplicate output path: {relative}")
        scan_portable(data.decode("utf-8"))
        files[relative] = data

    def add_log(source, destination):
        data = source.read_bytes()
        sanitized = redact_log(data.decode("utf-8", errors="strict")).encode()
        put(destination, sanitized)
        return {"file": destination, "original_sha256": sha_bytes(data), "published_sha256": sha_bytes(sanitized),
                "redaction": "User-specific storage roots and account name only; all lines retained"}

    for name, root, is_remote, allowed_data, allowed_seeds in selected_packages(local, hpc):
        index = read_json(root / "result.json")
        root_sha = sha_file(root / "result.json")
        if name != "local_tabular":
            require(index["execution_status"] == "complete", f"Incomplete package: {name}")
        package_rows, count = [], 0
        for entry in index["cells"]:
            require(entry["dataset"] in allowed_data and entry["seed"] in allowed_seeds, "Unexpected selected cell")
            require(entry["method"] in METHODS, "Unexpected method")
            path = root / safe_relative(entry["result"])
            require(sha_file(path) == entry["result_sha256"], "Original cell hash mismatch")
            cell = read_json(path)
            identity = cell["identity"]
            key = tuple(identity[k] for k in ("dataset", "method", "seed"))
            require(key not in identities, "Duplicate scientific identity")
            identities.add(key)
            require(key == tuple(entry[k] for k in ("dataset", "method", "seed")), "Index identity mismatch")
            require(cell["completed"] is True and identity["git_commit"] in SOURCES, "Incomplete or unknown source")
            require(canonical(identity["effective_config"]) == identity["effective_config_sha256"] == EFFECTIVE_SHA, "Effective config mismatch")
            require(identity["effective_config"]["epochs_per_task"] == 8 and not identity["effective_config"]["smoke"], "Nonformal execution")
            require(len(cell["checkpoints"]) == entry["checkpoints"] == len(cell["tasks"]), "Checkpoint count mismatch")
            cell_dir = safe_relative(entry["result"]).parent
            for task, checkpoint in enumerate(cell["checkpoints"]):
                require(checkpoint["task_index"] == task and checkpoint["identity_hash"] == canonical(identity), "Checkpoint identity mismatch")
                cp_name = f"checkpoints/checkpoint_{task:03d}.json"
                cp_path = path.parent / cp_name
                require(read_json(cp_path) == checkpoint, "Standalone checkpoint differs")
                weight = safe_relative(checkpoint["checkpoint_file"])
                digest = checkpoint["checkpoint_sha256"]
                if is_remote:
                    short = "cic17" if identity["dataset"] == "CIC-IDS-2017" else "cic18"
                    remote_key = f"{short}/seed_{identity['seed']}/all/" + (cell_dir / weight).as_posix()
                    evidence = remote_proofs.get((remote_key, digest), [])
                    require(bool(evidence), f"Missing actual remote weight hash evidence: {remote_key}")
                    selected_evidence = min(evidence, key=lambda item: (-evidence_coverage[item[0]], item[0]))
                    evidence_name, evidence_sha = selected_evidence
                    if evidence_name not in published_evidence:
                        published_evidence[evidence_name] = add_log(evidence_root / evidence_name, f"evidence/{evidence_name}")
                    proof = {"verification": "actual_remote_weight_sha256_saved_observation",
                             "evidence_file": published_evidence[evidence_name]["file"],
                             "evidence_file_sha256": evidence_sha,
                             "published_evidence_sha256": published_evidence[evidence_name]["published_sha256"]}
                else:
                    require(sha_file(path.parent / weight) == digest, "Actual local weight hash mismatch")
                    proof = {"verification": "actual_local_weight_sha256_at_merge"}
                cp_output = (cell_dir / cp_name).as_posix()
                put(cp_output, cp_path.read_bytes())
                proofs.append({"checkpoint_record": cp_output, "record_sha256": sha_file(cp_path), "weight_sha256": digest, **proof})
            final = cell["checkpoints"][-1]
            for split in ("id", "shift"):
                if final[split] is not None:
                    package_rows.extend(dict(dataset=key[0], method=key[1], seed=key[2], split=split, **row) for row in final[split]["per_class"])
            clean = copy.deepcopy(cell)
            for field in ("command", "log_path"):
                if field in clean:
                    clean[field] = "[Operational location redacted; original immutable artifact is SHA256-bound below]"
            clean["publication"] = {"source_package": name, "original_result_sha256": entry["result_sha256"],
                                    "original_matrix_sha256": root_sha, "redacted_fields": [k for k in ("command", "log_path") if k in cell],
                                    "identity_and_checkpoints_unchanged": True, "weights_included": False}
            data = json_bytes(clean)
            put(entry["result"], data)
            entries.append({**entry, "result_sha256": sha_bytes(data), "original_result_sha256": entry["result_sha256"], "source_package": name})
            count += 1
        require(package_rows == index["final_rows"], "Final rows do not reconstruct original matrix")
        check_csv(root / "final_rows.csv", package_rows)
        log_records = []
        for source in sorted(root.rglob("*")):
            if source.is_file() and source.suffix in (".log", ".jsonl") and not source.name.endswith(".public.log"):
                log_records.append(add_log(source, f"logs/{name}/" + source.relative_to(root).as_posix()))
            elif source.is_file() and source.name in ("job_start.json", "job_exit.json"):
                log_records.append(add_log(source, f"logs/{name}/" + source.relative_to(root).as_posix()))
        if not is_remote:
            prefix = "formal_v1" if name == "local_tabular" else f"local_tail_{name.removeprefix('local_')}_v1"
            for suffix in ("console.log", "stderr.log"):
                source = evidence_root / f"{prefix}.{suffix}"
                require(source.is_file(), f"Missing complete local log: {source.name}")
                log_records.append(add_log(source, f"logs/{name}/{suffix}"))
        packages.append({"package": name, "cells": count, "original_matrix_sha256": root_sha,
                         "original_final_rows_csv_sha256": sha_file(root / "final_rows.csv"), "logs": log_records})
        all_rows.extend(package_rows)
    expected = {(d, m, s) for d in DATASETS for m in METHODS for s in SEEDS}
    require(identities == expected and len(entries) == 120, "Not the complete fixed 120-cell matrix")
    require(len(proofs) == 510 and len(all_rows) == len({row_key(r) for r in all_rows}) == 1140, "Merged count mismatch")
    unsupported = [r for r in all_rows if r["recall"] is None]
    require(len(unsupported) == 30 and all(r["dataset"] == "CIC-IDS-2017" and r["split"] == "shift" and r["class_name"] == "PortScan" and r["support"] == 0 for r in unsupported), "Unsupported row mismatch")
    all_rows.sort(key=row_key)
    entries.sort(key=lambda e: (e["dataset"], e["method"], e["seed"]))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
    put("final_rows.csv", stream.getvalue().encode())
    result = {"schema_version": "section21_portable_first_valid_v1", "execution_status": "complete", "smoke": False,
              "expected": {"datasets": list(DATASETS), "methods": list(METHODS), "seeds": list(SEEDS)}, "expected_cells": 120,
              "cells": entries, "checkpoint_count": 510, "final_rows": all_rows, "final_row_count": 1140,
              "unsupported_shift_rows": 30, "training_seconds_total": sum(e["training_seconds_total"] for e in entries),
              "selection_policy": "Fixed first-valid package identities established before aggregate analysis; never outcome-based. Completed HPC tabular duplicates and aborted partial attempts are excluded from scientific rows.",
              "source_packages": packages, "checkpoint_verification": proofs,
              "remote_weight_evidence": list(published_evidence.values()),
              "merge_provenance": {"script_sha256": sha_file(Path(__file__)), "scientific_source_commits": sorted(SOURCES),
                                   "originals_unchanged": True, "weights_or_datasets_included": False,
                                   "scope": "Complete training artifacts; G1-G7 validation, literature and report qualification are separate gates"}}
    put("result.json", json_bytes(result))
    # Validate everything before writing. Existing differing bytes fail closed.
    for relative, data in files.items():
        destination = output / relative
        require(not destination.exists() or destination.read_bytes() == data, f"Refusing to replace differing artifact: {relative}")
    if output.exists():
        require({p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} <= set(files), "Unexpected files in output")
    for relative, data in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
    require(all(sha_file(output / p) == sha_bytes(b) for p, b in files.items()), "Published file hash mismatch")
    return {"status": "PASS", "cells": 120, "checkpoints": 510, "final_rows": 1140, "unsupported_shift_rows": 30,
            "local_actual_weights": sum(p["verification"].startswith("actual_local") for p in proofs),
            "remote_actual_weight_observations": sum(p["verification"].startswith("actual_remote") for p in proofs),
            "files": len(files), "bytes": sum(map(len, files.values())), "root_sha256": sha_file(output / "result.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--hpc-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.local_root, args.hpc_root, args.evidence_root, args.output), indent=2))


if __name__ == "__main__":
    main()
