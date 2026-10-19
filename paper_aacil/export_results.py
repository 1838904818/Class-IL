"""Export small, path-sanitized publication copies, never weights.

Reads completed non-smoke physical cells from the four explicit experiment
roots; --section29 additionally includes formal Section29/local_v1 cells.
Original JSON, checkpoints, logs, matrices and score caches are immutable.
The paper table generator can consume --input publication_v1 without weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "ncd/results/problem_scout/section28"
SECTION29 = ROOT / "ncd/results/problem_scout/section29"
INPUT_NAMES = ("replay_bs_local_v1", "cic_local_queue_v1",
               "cic_aacil_fallback_v1", "hpc_cic_section28")
METHODS = {"replay_bs", "aacil", "ts_mem", "aa_noattmem"}
SECTION29_METHODS = {"aacil_kd", "aacil_nogkd", "aacil_iso"}
FORMAL_SEEDS = {1, 2, 3, 4, 42}
PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[/\\]|/(?:home|scr)/)")
URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
PATH_KEYS = {"command", "command_line", "log_path", "path", "cwd", "output_dir",
             "output_path", "data_root", "data_path", "config_path", "python_executable"}


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def private_text(value):
    return isinstance(value, str) and not URL.match(value) and bool(PRIVATE_PATH.search(value))


def private_fields(value, trail=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from private_fields(item, (*trail, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from private_fields(item, (*trail, str(index)))
    elif private_text(value):
        yield ".".join(trail)


def normalize_path(value):
    text = value.replace("\\", "/")
    # Longest execution root first: retain a runnable repository-relative command.
    replacements = ((ROOT.as_posix(), ""), (ROOT.parents[1].as_posix(), "<WORKSPACE>"),
                    (Path.home().as_posix(), "<USER_HOME>"))
    for prefix, replacement in replacements:
        text = re.sub(re.escape(prefix) + r"(?=/|$)", lambda _: replacement, text, flags=re.IGNORECASE)
    # Removing the repository root can leave a leading slash after an option.
    text = re.sub(r"(?:(?<=\s)|^)/(ncd|paper_aacil|work)(?=/)", r"\1", text)
    text = re.sub(r'/(home|scr)/(?:user/)?[^/\s\"\']+',
                  lambda m: "<HPC_HOME>" if m.group(1) == "home" else "<HPC_SCRATCH>", text)
    if private_text(text):
        raise ValueError("Unrecognized private absolute path in metadata; no generic redaction applied")
    return text


def sanitize(value, trail=(), changed=None):
    if changed is None:
        changed = []
    if isinstance(value, dict):
        return {key: sanitize(item, (*trail, key), changed) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, (*trail, str(index)), changed) for index, item in enumerate(value)]
    if not private_text(value):
        return value
    if "identity" in trail or (trail and trail[-1] == "checkpoint_file"):
        raise ValueError("Protected identity/checkpoint reference contains a private path: " + ".".join(trail))
    key = trail[-1] if trail else ""
    if key not in PATH_KEYS and not key.endswith(("_path", "_dir", "_root")):
        raise ValueError("Private path outside permitted command/path metadata: " + ".".join(trail))
    cleaned = normalize_path(value)
    if cleaned != value:
        changed.append(".".join(trail))
    return cleaned


def derived_copy(source, source_hash, source_path, result_path, result_hash):
    changed = []
    exported = sanitize(source, changed=changed)
    # Identity and every non-string value are carried through, not recomputed.
    if exported.get("identity") != source.get("identity"):
        raise AssertionError("Publication export must not alter execution identity")
    exported["publication_derivation"] = {
        "kind": "derived_small_json_with_path_metadata_sanitization",
        "original_result": relative(result_path), "original_result_sha256": result_hash,
        "original_json": relative(source_path), "original_json_sha256": source_hash,
        "redacted_fields": changed,
        "changes": "Only private absolute roots in command/path metadata are normalized; original identity and all original numeric values are unchanged.",
        "not_byte_identical_to_original": True,
        "checkpoint_reference_base": "checkpoint_file is unchanged and resolves against the original result directory, not this publication directory; weights are not exported.",
    }
    if list(private_fields(exported)):
        raise ValueError("Private path remains in derived publication JSON")
    return exported, changed


def write_derived(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() == payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    for attempt in range(40):
        try:
            temporary.replace(path)
            return True
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=INPUT_NAMES, action="append",
                        help="Explicit Section28 experiment root name; defaults to all four")
    parser.add_argument("--section29", action="store_true",
                        help="Also export completed formal KD/noGKD/ISO cells from Section29/local_v1; never tuning or smoke")
    parser.add_argument("--output-dir", type=Path, default=SECTION / "publication_v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    roots = [SECTION / name for name in (args.input or INPUT_NAMES)]
    if args.section29:
        roots.append(SECTION29 / "local_v1")
    if not output.is_relative_to(SECTION) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in roots):
        parser.error("Publication output must be a separate child of Section28, never an original experiment root")
    selected = {}
    for root in roots:
        allowed_methods = SECTION29_METHODS if root == SECTION29 / "local_v1" else METHODS
        for path in sorted(root.rglob("result.json")):
            raw, source_hash = read(path)
            identity = raw.get("identity", {})
            config = identity.get("effective_config", {})
            if (not raw.get("completed") or not raw.get("checkpoints") or
                    identity.get("method") not in allowed_methods or raw.get("smoke") or
                    config.get("smoke") or identity.get("seed") not in FORMAL_SEEDS or
                    raw.get("selection_only") or config.get("selection_only")):
                continue  # Matrices, incomplete cells, smoke and tuning are never formal publication evidence.
            if allowed_methods is SECTION29_METHODS and (
                    raw.get("test_evaluation_performed") is not True or
                    len(raw["checkpoints"]) != len(raw.get("tasks", [])) or
                    any(checkpoint.get("test_evaluation_performed") is not True
                        for checkpoint in raw["checkpoints"])):
                continue  # Selection-only/partial Section29 records cannot masquerade as formal cells.
            key = tuple(identity[k] for k in ("dataset", "method", "seed"))
            completed = datetime.fromisoformat(raw["completed_at"])
            if completed.tzinfo is None:
                raise ValueError("Completed-cell timestamp must carry a timezone: " + relative(path))
            candidate = (completed, relative(path))
            if key not in selected or candidate < selected[key][0]:
                selected[key] = (candidate, path, raw, source_hash)
    files, rewritten, redactions, groups = 0, 0, 0, Counter()
    for (dataset, method, seed), (_, result_path, result, result_hash) in sorted(selected.items()):
        destination = output / dataset.lower().replace("-", "_") / f"{method}_seed{seed}"
        prepared = []
        derived, changed = derived_copy(result, result_hash, result_path, result_path, result_hash)
        prepared.append((destination / "result.json", derived, changed))
        for checkpoint in result["checkpoints"]:
            # Resolve only the small JSON sidecar; never open/read/copy the PT payload.
            checkpoint_relative = Path(checkpoint["checkpoint_file"])
            if checkpoint_relative.is_absolute() or ".." in checkpoint_relative.parts:
                raise ValueError("Expected an unchanged cell-relative checkpoint_file")
            sidecar_relative = checkpoint_relative.with_suffix(".json")
            if not re.fullmatch(r"checkpoint_\d+\.json", sidecar_relative.name):
                raise ValueError("Unexpected checkpoint sidecar name")
            sidecar_path = result_path.parent / sidecar_relative
            sidecar, sidecar_hash = read(sidecar_path)
            derived, changed = derived_copy(sidecar, sidecar_hash, sidecar_path, result_path, result_hash)
            prepared.append((destination / sidecar_relative, derived, changed))
        for path, derived, changed in prepared:
            rewritten += int(write_derived(path, derived))
            files += 1
            redactions += len(changed)
        groups[f"{dataset}/{method}"] += 1
    # Curves already use repository-relative binding paths; inspect rather than
    # generically copying/re-writing their immutable caches or global summary.
    curve_private = []
    for path in (SECTION / "paper_curves_v1").rglob("*.json"):
        raw, _ = read(path)
        fields = list(private_fields(raw))
        if fields:
            curve_private.append({"path": relative(path), "fields": fields})
    print(json.dumps({"completed_cells_exported": len(selected), "small_json_files": files,
        "files_written": rewritten, "path_fields_redacted": redactions, "groups": dict(groups),
        "publication_root": relative(output), "curve_private_path_fields": curve_private,
        "weights_logs_matrices_copied": False}, ensure_ascii=False))
    if curve_private:
        raise ValueError("Curve metadata has private paths; preserve originals and export only these affected files separately")


if __name__ == "__main__":
    main()
