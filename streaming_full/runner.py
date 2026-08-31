from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument(
        "--training-calibration-audit",
        type=Path,
        help=(
            "Hash-verified training-only calibration audit. Required only for "
            "training_only_calibration_macro_f1 checkpoint selection."
        ),
    )
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
    parser.add_argument(
        "--wandb-project",
        help="Enable one online W&B run per seed in this new or dedicated project.",
    )
    parser.add_argument("--wandb-entity", help="Optional W&B user or team entity.")
    parser.add_argument("--wandb-group", help="Optional W&B run group.")
    parser.add_argument(
        "--wandb-run-name-prefix",
        default="ofra",
        help="Prefix for the per-seed W&B run names.",
    )
    parser.add_argument(
        "--wandb-tag",
        action="append",
        default=[],
        help="Repeatable W&B tag. No credentials are accepted as CLI arguments.",
    )
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="Enable hash-bound epoch/task-boundary recovery for one seed.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        help="Cleanly pause before this process runtime budget is exhausted.",
    )
    parser.add_argument(
        "--recovery-stop-margin-seconds",
        type=float,
        default=1800.0,
        help="Reserved time for a clean checkpoint before the runtime deadline.",
    )
    parser.add_argument(
        "--recovery-minimum-next-unit-seconds",
        type=float,
        default=0.0,
        help="Conservative lower bound for the next epoch/task unit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_runtime_seconds is not None and not args.recovery:
        raise ValueError("--max-runtime-seconds requires --recovery")
    if args.recovery and len(args.seeds) != 1:
        raise ValueError("--recovery requires exactly one seed")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        raise ValueError("--max-runtime-seconds must be positive")
    if args.recovery_stop_margin_seconds < 0:
        raise ValueError("--recovery-stop-margin-seconds cannot be negative")
    if args.recovery_minimum_next_unit_seconds < 0:
        raise ValueError("--recovery-minimum-next-unit-seconds cannot be negative")
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
    tracker = None
    if args.wandb_project:
        from .wandb_tracking import WandbSeedTracker

        tracker = WandbSeedTracker(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            run_name_prefix=args.wandb_run_name_prefix,
            tags=args.wandb_tag,
            output_dir=args.output_dir,
        )
    try:
        output = run_manifest(
            args.manifest,
            seeds=args.seeds,
            output_dir=args.output_dir,
            config=config,
            evaluation_view_paths=evaluation_paths,
            training_calibration_audit_path=args.training_calibration_audit,
            event_sink=tracker,
            recovery_enabled=args.recovery,
            recovery_deadline_unix=(
                time.time() + args.max_runtime_seconds
                if args.max_runtime_seconds is not None
                else None
            ),
            recovery_stop_margin_seconds=args.recovery_stop_margin_seconds,
            recovery_minimum_next_unit_seconds=(
                args.recovery_minimum_next_unit_seconds
            ),
        )
    finally:
        if tracker is not None:
            tracker.close()
    if output.get("paused") is not None:
        print(
            json.dumps(
                {
                    "status": "paused",
                    "output_dir": str(args.output_dir.resolve()),
                    "protocol_sha256": output["protocol"]["protocol_sha256"],
                    "recovery": output["paused"],
                },
                indent=2,
            )
        )
        return 0
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
