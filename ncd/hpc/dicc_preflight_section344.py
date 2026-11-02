#!/usr/bin/env python3
"""Fail-closed static checks for DICC Slurm job scripts.

This project guard complements, but does not replace, live DICC rules. Run it
locally and again on the login node before submission.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


QOS_MAX_SECONDS = {
    "limited": 60 * 60,
    "short": 60 * 60,
    "normal": 8 * 60 * 60,
    "long": 3 * 24 * 60 * 60,
}

LIMITED_PARTITIONS = {
    "cpu-epyc",
    "gpu-k40c",
    "gpu-titan",
    "gpu-v100s",
}

REQUIRED_DIRECTIVES = (
    "job-name",
    "partition",
    "qos",
    "time",
    "nodes",
    "ntasks",
    "cpus-per-task",
    "output",
    "error",
)

ALLOWED_DIRECTIVES = set(REQUIRED_DIRECTIVES) | {
    "dependency",
    "mem",
    "mem-per-cpu",
    "gpus",
    "gpus-per-node",
    "gpus-per-task",
    "gres",
    "mail-type",
    "mail-user",
    "signal",
    "chdir",
    # Retained only so the validator can issue a specific hard failure.
    "array",
    "exclusive",
    "oversubscribe",
}

SHORT_DIRECTIVES = {
    "-J": "job-name",
    "-p": "partition",
    "-q": "qos",
    "-t": "time",
    "-N": "nodes",
    "-n": "ntasks",
    "-c": "cpus-per-task",
    "-o": "output",
    "-e": "error",
}

SECRET_NAMES = (
    "WANDB_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "PASSWORD",
    "PASSWD",
)


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str
    line: int | None = None


def parse_directives(text: str) -> tuple[dict[str, list[tuple[str, int]]], list[Issue]]:
    directives: dict[str, list[tuple[str, int]]] = {}
    issues: list[Issue] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^\s*#SBATCH\s+(.+?)\s*$", line)
        if not match:
            continue
        try:
            tokens = shlex.split(match.group(1), comments=False, posix=True)
        except ValueError as exc:
            issues.append(Issue("ERROR", "SBATCH_PARSE", str(exc), line_number))
            continue
        if not tokens:
            issues.append(Issue("ERROR", "SBATCH_EMPTY", "Empty #SBATCH directive.", line_number))
            continue

        token = tokens[0]
        key: str | None = None
        value: str | None = None
        if token.startswith("--"):
            option = token[2:]
            if "=" in option:
                key, value = option.split("=", 1)
                if len(tokens) != 1:
                    issues.append(
                        Issue(
                            "ERROR",
                            "SBATCH_EXTRA",
                            "Unexpected text after a #SBATCH --key=value directive.",
                            line_number,
                        )
                    )
            else:
                key = option
                if len(tokens) == 2:
                    value = tokens[1]
                elif len(tokens) > 2:
                    issues.append(
                        Issue(
                            "ERROR",
                            "SBATCH_EXTRA",
                            "Use one unambiguous value per #SBATCH directive.",
                            line_number,
                        )
                    )
        elif token in SHORT_DIRECTIVES:
            key = SHORT_DIRECTIVES[token]
            if len(tokens) == 2:
                value = tokens[1]
            elif len(tokens) > 2:
                issues.append(
                    Issue(
                        "ERROR",
                        "SBATCH_EXTRA",
                        "Use one unambiguous value per #SBATCH directive.",
                        line_number,
                    )
                )
        else:
            issues.append(
                Issue(
                    "ERROR",
                    "SBATCH_UNKNOWN",
                    f"Unsupported or ambiguous directive syntax: {token}",
                    line_number,
                )
            )
            continue

        if not key:
            continue
        if value is None:
            # Flag-like directives are retained so prohibited flags can be found.
            value = "true"
        directives.setdefault(key, []).append((value, line_number))

    return directives, issues


def one(
    directives: dict[str, list[tuple[str, int]]],
    key: str,
    issues: list[Issue],
) -> tuple[str | None, int | None]:
    values = directives.get(key, [])
    if not values:
        return None, None
    if len(values) > 1:
        issues.append(
            Issue(
                "ERROR",
                "SBATCH_DUPLICATE",
                f"Directive --{key} appears more than once.",
                values[-1][1],
            )
        )
    return values[-1]


def parse_positive_int(
    value: str | None,
    name: str,
    line: int | None,
    issues: list[Issue],
) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        issues.append(Issue("ERROR", "RESOURCE_INTEGER", f"--{name} must be an integer.", line))
        return None
    if parsed <= 0:
        issues.append(Issue("ERROR", "RESOURCE_POSITIVE", f"--{name} must be positive.", line))
        return None
    return parsed


def parse_time_seconds(value: str, line: int | None, issues: list[Issue]) -> int | None:
    try:
        days = 0
        rest = value
        if "-" in rest:
            day_text, rest = rest.split("-", 1)
            days = int(day_text)
        parts = [int(part) for part in rest.split(":")]
        if len(parts) == 1:
            hours, minutes, seconds = 0, parts[0], 0
        elif len(parts) == 2:
            hours, minutes, seconds = 0, parts[0], parts[1]
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            raise ValueError
        if min(days, hours, minutes, seconds) < 0 or minutes >= 60 or seconds >= 60:
            raise ValueError
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    except (TypeError, ValueError):
        issues.append(
            Issue(
                "ERROR",
                "TIME_FORMAT",
                "--time must use Slurm syntax such as 00:30:00, 8:00:00, or 3-00:00:00.",
                line,
            )
        )
        return None


def parse_memory_gib(value: str, line: int | None, issues: list[Issue]) -> float | None:
    match = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)([kmgt]?)(?:i?b)?", value.strip())
    if not match:
        issues.append(
            Issue(
                "ERROR",
                "MEMORY_FORMAT",
                "Memory must use a Slurm value such as 32G or 64000M.",
                line,
            )
        )
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper() or "M"
    factors = {"K": 1 / (1024 * 1024), "M": 1 / 1024, "G": 1, "T": 1024}
    return amount * factors[unit]


def gpu_count(directives: dict[str, list[tuple[str, int]]], issues: list[Issue]) -> tuple[int | None, int | None]:
    candidates: list[tuple[int, int]] = []

    for key in ("gpus", "gpus-per-node", "gpus-per-task"):
        for value, line in directives.get(key, []):
            count_text = value.rsplit(":", 1)[-1]
            count = parse_positive_int(count_text, key, line, issues)
            if count is not None:
                candidates.append((count, line))

    for value, line in directives.get("gres", []):
        match = re.search(r"(?:^|,)gpu(?::[^:,]+)?:(\d+)(?:,|$)", value)
        if not match:
            issues.append(
                Issue(
                    "ERROR",
                    "GPU_GRES",
                    "Use --gres=gpu:COUNT or --gres=gpu:TYPE:COUNT.",
                    line,
                )
            )
            continue
        candidates.append((int(match.group(1)), line))

    if not candidates:
        return None, None
    if len(candidates) > 1:
        issues.append(
            Issue(
                "ERROR",
                "GPU_DUPLICATE",
                "Use exactly one GPU request directive.",
                candidates[-1][1],
            )
        )
    return candidates[-1]


def active_shell_lines(text: str) -> list[tuple[int, str]]:
    active: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        active.append((line_number, stripped))
    return active


def check_body(text: str, walltime: int | None, issues: list[Issue]) -> None:
    lines = active_shell_lines(text)
    body = "\n".join(line for _, line in lines)

    if not re.search(r"(?m)^\s*set\s+-[A-Za-z]*e[A-Za-z]*u[A-Za-z]*o\s+pipefail\b", body):
        issues.append(
            Issue(
                "ERROR",
                "SHELL_STRICT",
                "Add `set -euo pipefail` before job commands.",
            )
        )

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if not re.search(rf"(?m)^\s*(?:export\s+)?{variable}\s*=", body):
            issues.append(
                Issue(
                    "ERROR",
                    "THREAD_CAP",
                    f"Set {variable} from SLURM_CPUS_PER_TASK to prevent CPU oversubscription.",
                )
            )

    if not re.search(r"(?m)^\s*srun(?:\s|$)", body):
        issues.append(
            Issue(
                "ERROR",
                "SRUN_REQUIRED",
                "Launch the calculation with `srun` inside the allocation.",
            )
        )

    banned_patterns: list[tuple[str, str, str]] = [
        (
            r"\b(?:python(?:3)?\s+-m\s+http\.server|jupyter(?:-lab|-notebook|\s+lab|\s+notebook)?|"
            r"uvicorn|gunicorn|flask\s+run|streamlit\s+run|gradio|tensorboard|wandb\s+(?:server|local))\b",
            "SERVER",
            "Servers and notebook/dashboard listeners are prohibited on DICC nodes.",
        ),
        (
            r"\b(?:ngrok|cloudflared|frp[cs]?|socat|ncat\s+-l|nc\s+-l)\b|"
            r"\bssh\b[^\n]*(?:\s-[RLD]\s*|\s-[RLD][0-9])",
            "TUNNEL",
            "Reverse proxies, tunnels, port forwarding, and listeners are prohibited.",
        ),
        (
            r"\b(?:sudo|su\s+-|apt(?:-get)?|dnf|yum|pacman|systemctl|service)\b",
            "SYSTEM_ADMIN",
            "System administration and system package managers are prohibited.",
        ),
        (
            r"\b(?:pip(?:3)?|conda|mamba|micromamba|npm|yarn|pnpm)\s+(?:install|create|update|upgrade)\b",
            "INSTALL_IN_JOB",
            "Build the environment in an allocated setup session, not in a training job.",
        ),
        (
            r"\b(?:curl|wget|git\s+clone|scp|rsync|rclone)\b",
            "TRANSFER_IN_JOB",
            "Pre-stage code and data; do not perform network transfer inside a compute job.",
        ),
        (
            r"\b(?:sbatch|salloc|scancel)\b|\bsrun\b[^\n]*--pty\b",
            "NESTED_SCHEDULER",
            "A job must not submit/cancel jobs or start a nested interactive allocation.",
        ),
        (
            r"\bxargs\b[^\n]*\s-P(?:\s*\d+)?\b|\bparallel\b|\btorchrun\b|\bmpirun\b|\bmpiexec\b",
            "FANOUT",
            "Parallel fan-out and distributed launchers require a separately approved design.",
        ),
        (
            r"(?:^|[;&|]\s*)[^#\n]+&\s*(?:$|#)",
            "BACKGROUND",
            "Do not launch unmanaged background processes in a batch job.",
        ),
        (
            r"\bfind\b[^\n]*(?:-exec|-execdir)[^\n]*\btouch\b|"
            r"\bfind\b[^\n]*\|\s*xargs\b[^\n]*\btouch\b|"
            r"\btouch\s+-[acm]\b|\bos\.utime\s*\(",
            "TIMESTAMP_EVASION",
            "Timestamp manipulation that can evade scratch cleanup is prohibited.",
        ),
        (
            r"\brm\s+-[^\n]*r[^\n]*f|\bchmod\s+(?:-R\s+)?777\b",
            "DESTRUCTIVE",
            "Recursive forced deletion and world-writable permissions are blocked.",
        ),
        (
            r"(?:[A-Za-z]:[\\/])",
            "WINDOWS_PATH",
            "Windows paths cannot be used in a DICC job.",
        ),
        (
            r"/(?:home|scr)/user/(?!\$\{?USER\}?)(?!\$USER\b)[A-Za-z0-9._-]+",
            "OTHER_USER_PATH",
            "Use $HOME or /scr/user/$USER; hard-coded user paths are blocked.",
        ),
    ]

    for line_number, line in lines:
        for pattern, code, message in banned_patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                issues.append(Issue("ERROR", code, message, line_number))

        if re.match(r"^(?:python(?:3)?|Rscript|matlab)\b", line):
            issues.append(
                Issue(
                    "ERROR",
                    "DIRECT_LAUNCH",
                    "Launch calculations with `srun`, not directly from the batch shell.",
                    line_number,
                )
            )

        for secret in SECRET_NAMES:
            match = re.search(rf"\b{re.escape(secret)}\s*=\s*([^\s;]+)", line, flags=re.IGNORECASE)
            if not match:
                continue
            value = match.group(1).strip("\"'")
            if value and not value.startswith("$"):
                issues.append(
                    Issue(
                        "ERROR",
                        "EMBEDDED_SECRET",
                        f"Do not embed a literal value for {secret}; inject it securely at runtime.",
                        line_number,
                    )
                )

    if walltime is not None and walltime > 8 * 60 * 60:
        if not re.search(r"(?i)(checkpoint|resume(?:-from)?|--resume|DICC_CHECKPOINT)", body):
            issues.append(
                Issue(
                    "ERROR",
                    "CHECKPOINT",
                    "Jobs longer than eight hours require explicit checkpoint/resume handling.",
                )
            )


def validate(
    path: Path,
    text: str,
    account_status: str,
) -> tuple[list[Issue], dict[str, str]]:
    directives, issues = parse_directives(text)

    # Bounded section346 dependency: queue now, allocate only after the one
    # explicitly owned BiC14 continuation succeeds. No general dependency API.
    dependency, dependency_line = one(directives, "dependency", issues)
    if dependency != "afterok:493652":
        issues.append(Issue("ERROR", "OWNED_DEPENDENCY", "Section344 requires exactly afterok:493652; verify owner live before submission.", dependency_line))

    if not text.startswith("#!"):
        issues.append(Issue("ERROR", "SHEBANG", "The first line must be a shell shebang.", 1))

    for key, entries in directives.items():
        if key not in ALLOWED_DIRECTIVES:
            for _, line in entries:
                issues.append(
                    Issue(
                        "ERROR",
                        "SBATCH_NOT_ALLOWED",
                        f"--{key} is outside the project's reviewed directive allowlist.",
                        line,
                    )
                )

    values: dict[str, str] = {}
    locations: dict[str, int | None] = {}
    for key in set(REQUIRED_DIRECTIVES) | {
        "mem",
        "mem-per-cpu",
        "array",
        "exclusive",
        "oversubscribe",
        "chdir",
    }:
        value, line = one(directives, key, issues)
        if value is not None:
            values[key] = value
        locations[key] = line

    for key in REQUIRED_DIRECTIVES:
        if key not in values:
            issues.append(
                Issue(
                    "ERROR",
                    "SBATCH_REQUIRED",
                    f"Missing required #SBATCH --{key} directive.",
                )
            )

    if "mem" not in values and "mem-per-cpu" not in values:
        issues.append(
            Issue(
                "ERROR",
                "SBATCH_REQUIRED",
                "Specify exactly one of --mem or --mem-per-cpu.",
            )
        )
    if "mem" in values and "mem-per-cpu" in values:
        issues.append(
            Issue(
                "ERROR",
                "MEMORY_DUPLICATE",
                "Use --mem or --mem-per-cpu, not both.",
                locations["mem-per-cpu"],
            )
        )

    for prohibited in ("array", "exclusive", "oversubscribe"):
        if prohibited in values:
            issues.append(
                Issue(
                    "ERROR",
                    "SBATCH_PROHIBITED",
                    f"--{prohibited} is disabled by the project's conservative DICC policy.",
                    locations[prohibited],
                )
            )

    nodes = parse_positive_int(values.get("nodes"), "nodes", locations["nodes"], issues)
    tasks = parse_positive_int(values.get("ntasks"), "ntasks", locations["ntasks"], issues)
    cpus = parse_positive_int(
        values.get("cpus-per-task"),
        "cpus-per-task",
        locations["cpus-per-task"],
        issues,
    )
    if nodes is not None and nodes != 1:
        issues.append(Issue("ERROR", "ONE_NODE", "This project is restricted to one node.", locations["nodes"]))
    if tasks is not None and tasks != 1:
        issues.append(Issue("ERROR", "ONE_TASK", "This project is restricted to one task.", locations["ntasks"]))
    if cpus is not None and cpus > 16:
        issues.append(
            Issue(
                "ERROR",
                "CPU_CAP",
                "The project safety cap is 16 CPUs per task; benchmark before requesting more.",
                locations["cpus-per-task"],
            )
        )

    for memory_key in ("mem", "mem-per-cpu"):
        if memory_key in values:
            memory_gib = parse_memory_gib(values[memory_key], locations[memory_key], issues)
            if memory_gib is not None and memory_gib > 128:
                issues.append(
                    Issue(
                        "ERROR",
                        "MEMORY_CAP",
                        "The project safety cap is 128 GiB; benchmark and revise before requesting more.",
                        locations[memory_key],
                    )
                )

    partition = values.get("partition")
    partition_line = locations["partition"]
    gpu_partition = False
    if partition is not None:
        partitions = [item.strip().rstrip("*") for item in partition.split(",") if item.strip()]
        if len(partitions) != 1:
            issues.append(
                Issue(
                    "ERROR",
                    "ONE_PARTITION",
                    "Choose one explicit partition; do not provide a fallback list.",
                    partition_line,
                )
            )
        elif not re.fullmatch(r"(?:cpu|gpu)-[A-Za-z0-9._-]+", partitions[0]):
            issues.append(
                Issue(
                    "ERROR",
                    "PARTITION_NAME",
                    "Partition must be an explicit DICC cpu-* or gpu-* name.",
                    partition_line,
                )
            )
        else:
            partition = partitions[0]
            values["partition"] = partition
            gpu_partition = partition.startswith("gpu-")

        if account_status == "limited" and partition not in LIMITED_PARTITIONS:
            issues.append(
                Issue(
                    "ERROR",
                    "LIMITED_PARTITION",
                    "LIMITED accounts may use only the documented limited-access partitions.",
                    partition_line,
                )
            )

    requested_gpus, gpu_line = gpu_count(directives, issues)
    if gpu_partition:
        if requested_gpus is None:
            issues.append(
                Issue(
                    "ERROR",
                    "GPU_REQUIRED",
                    "A GPU partition requires an explicit GPU request.",
                    partition_line,
                )
            )
        elif requested_gpus > 2:
            issues.append(
                Issue(
                    "ERROR",
                    "GPU_CAP",
                    "Current DICC announcement limits a job to at most two GPUs.",
                    gpu_line,
                )
            )
    elif requested_gpus is not None:
        issues.append(
            Issue(
                "ERROR",
                "GPU_ON_CPU",
                "Do not request GPUs in a CPU partition.",
                gpu_line,
            )
        )

    qos = values.get("qos")
    qos_line = locations["qos"]
    if qos is not None:
        qos = qos.lower()
        values["qos"] = qos
        if qos not in QOS_MAX_SECONDS:
            issues.append(
                Issue(
                    "ERROR",
                    "QOS_UNKNOWN",
                    "Use one documented QoS: limited, short, normal, or long.",
                    qos_line,
                )
            )
        if account_status == "limited" and qos != "limited":
            issues.append(
                Issue(
                    "ERROR",
                    "LIMITED_QOS",
                    "A LIMITED account must use the limited QoS.",
                    qos_line,
                )
            )
        if account_status == "full" and qos == "limited":
            issues.append(
                Issue(
                    "ERROR",
                    "FULL_QOS",
                    "A full-access account should use short, normal, or long QoS.",
                    qos_line,
                )
            )

    walltime: int | None = None
    if "time" in values:
        walltime = parse_time_seconds(values["time"], locations["time"], issues)
        if walltime is not None and qos in QOS_MAX_SECONDS and walltime > QOS_MAX_SECONDS[qos]:
            issues.append(
                Issue(
                    "ERROR",
                    "QOS_WALLTIME",
                    f"Requested walltime exceeds the documented {qos} QoS limit.",
                    locations["time"],
                )
            )
        if account_status == "limited" and walltime is not None and walltime > 3600:
            issues.append(
                Issue(
                    "ERROR",
                    "LIMITED_WALLTIME",
                    "A LIMITED account is capped at one hour.",
                    locations["time"],
                )
            )

    for output_key in ("output", "error"):
        output_value = values.get(output_key)
        if output_value is not None and not re.search(r"%(?:j|A)", output_value):
            issues.append(
                Issue(
                    "ERROR",
                    "LOG_COLLISION",
                    f"--{output_key} must include %j or %A to keep job logs distinct.",
                    locations[output_key],
                )
            )

    job_name = values.get("job-name")
    if job_name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", job_name):
        issues.append(
            Issue(
                "ERROR",
                "JOB_NAME",
                "Use a short job name containing only letters, digits, dot, underscore, or hyphen.",
                locations["job-name"],
            )
        )

    chdir = values.get("chdir")
    if chdir is not None:
        if re.search(r"[A-Za-z]:[\\/]", chdir):
            issues.append(
                Issue(
                    "ERROR",
                    "CHDIR_WINDOWS",
                    "--chdir cannot contain a Windows path.",
                    locations["chdir"],
                )
            )
        if re.search(r"/(?:home|scr)/user/(?!\$\{?USER\}?)(?!\$USER\b)[A-Za-z0-9._-]+", chdir):
            issues.append(
                Issue(
                    "ERROR",
                    "CHDIR_OWNER",
                    "--chdir must not name another or hard-coded user directory.",
                    locations["chdir"],
                )
            )

    check_body(text, walltime, issues)
    return issues, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, help="Path to one sbatch script.")
    parser.add_argument(
        "--account-status",
        required=True,
        choices=("limited", "full"),
        help="Current status from account-info; never guess.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable result.")
    parser.add_argument(
        "--print-directive",
        metavar="NAME",
        help="After a successful validation, print one normalized directive value.",
    )
    args = parser.parse_args()

    try:
        text = args.job.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"DICC preflight: cannot read {args.job}: {exc}", file=sys.stderr)
        return 2

    issues, values = validate(args.job, text, args.account_status)
    errors = [issue for issue in issues if issue.severity == "ERROR"]

    if args.json:
        print(
            json.dumps(
                {
                    "job": str(args.job),
                    "account_status": args.account_status,
                    "ok": not errors,
                    "directives": values,
                    "issues": [asdict(issue) for issue in issues],
                },
                indent=2,
            )
        )
    elif args.print_directive and not errors:
        value = values.get(args.print_directive)
        if value is None:
            print(f"Directive not found: {args.print_directive}", file=sys.stderr)
            return 2
        print(value)
    else:
        status = "PASS" if not errors else "FAIL"
        print(f"DICC preflight: {status} - {args.job}")
        for issue in issues:
            location = f"line {issue.line}: " if issue.line else ""
            print(f"[{issue.severity}] {issue.code}: {location}{issue.message}")
        if not issues:
            print("No policy violations detected. Live DICC checks are still required.")

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
