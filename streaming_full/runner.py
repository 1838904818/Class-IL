from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .validation import RunConfig, run_manifest


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration JSON must contain one object")
    allowed = {field.name for field in fields(RunConfig)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown RunConfig keys: {', '.join(unknown)}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run shard-backed, full-data OFRA validation without loading a full dataset into RAM."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--evaluation-view",
        action="append",
        default=[],
        metavar="NAME=MANIFEST",
        help=(
            "Repeatable test-only mask view. NAME must match the embedded view name; "
            "official is reserved."
        ),
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="JSON object whose keys are fields of streaming_full.validation.RunConfig.",
    )
    parser.add_argument("--device", help="Override the config device, for example cpu or cuda:0.")
    parser.add_argument("--quiet", action="store_true", help="Suppress task progress messages.")
    parser.add_argument(
        "--skip-shard-hash-verification",
        action="store_true",
        help="Allow unverified shards. Do not use this for reportable experiments.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = _load_config(args.config_json)
    if args.device:
        values["device"] = args.device
    if args.quiet:
        values["verbose"] = False
    if args.skip_shard_hash_verification:
        values["verify_shard_hashes"] = False
    config = RunConfig(**values)
    evaluation_paths = []
    names = set()
    for value in args.evaluation_view:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--evaluation-view must use NAME=MANIFEST")
        if name == "official" or name in names:
            raise ValueError("evaluation-view names must be unique and non-reserved")
        manifest_value = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if manifest_value.get("name") != name:
            raise ValueError("CLI evaluation-view name disagrees with its manifest")
        names.add(name)
        evaluation_paths.append(Path(raw_path))
    output = run_manifest(
        args.manifest,
        seeds=args.seeds,
        output_dir=args.output_dir,
        config=config,
        evaluation_view_paths=evaluation_paths,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "protocol_sha256": output["summary"]["protocol_sha256"],
                "deterministic_result_sha256": output["summary"][
                    "deterministic_result_sha256"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
