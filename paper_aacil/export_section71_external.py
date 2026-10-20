"""Publish the complete ten-cell external 1% evaluation without new inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import ROOT, read, private_fields, normalize_path, write_derived

METHODS = ("freq", "wacb")
SEEDS = tuple(range(5, 10))
SOURCES = {
    "f4f55927b277582de179b0c9211c272ec3997de1337375d2a72fa8778e594548": "cpu",
    "ca8b7b9498f4d69b54c8cc0c7a17d7872d0c7f95ceb9579fa67c1507c984b9f6": "cuda",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def execution(path):
    value, digest = read(path)
    require(value["status"] in ("failed", "complete") and value["ended_at"], "Nonterminal execution")
    require(value["source_sha256"] in SOURCES, "Unknown evaluator source")
    if "command" in value:
        value["command"] = [normalize_path(x) for x in value["command"]]
    return dict(record=value, original_sha256=digest)


def diagnostic(path):
    raw = path.read_bytes()
    text = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    rows = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    terminal = rows[-1]
    require(terminal["exit_code"] == 0 and terminal["ended_at"], "Diagnostic not terminal")
    result = next((r for r in rows if r.get("diagnostic_only")), None)
    if result:
        require(result["default_cm_reproduced"] and result["difference"]["absolute_difference_sum"] == 0,
                "Diagnostic did not reproduce original CM")
    return dict(log_sha256=hashlib.sha256(raw).hexdigest(), terminal=terminal,
                diagnostic_result=result, result_output_missing=result is None,
                cost_scope="Child diagnostic time is nested in supervisor wall time.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--diagnostic-logs", type=Path, nargs=2, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    dest = ROOT / "ncd/results/problem_scout/section70/external_alpha1_v1"
    require(not dest.exists(), "Publication already exists; do not export twice")
    require(not (source / ".evaluation.lock").exists(), "Evaluation still locked")
    attempts = [execution(p) for p in sorted(source.glob("attempt_*.json"))]
    supervisors = [execution(p) for p in sorted(source.glob("cuda_supervisor_*.json"))]
    expected = {(m, s) for m in METHODS for s in SEEDS}
    require(any(a["record"]["status"] == "complete" and
                {(c["method"], c["seed"]) for c in a["record"]["completed_cells"]} == expected
                for a in attempts), "No complete ten-cell assignment")
    require(supervisors and all(s["record"]["status"] == "complete" for s in supervisors),
            "CUDA supervisor not complete")
    diagnostics = [diagnostic(p) for p in args.diagnostic_logs]
    require(sum(d["diagnostic_result"] is not None for d in diagnostics) == 1, "Diagnostic capture mismatch")
    prepared = []
    for method, seed in sorted(expected):
        folder = source / f"{method}_seed{seed}"
        result, digest = read(folder / "result.json")
        require(result["completed"] and result["method"] == method and result["seed"] == seed
                and result["dataset"] == "CSE-CIC-IDS2018" and result["alpha"] == .01
                and len(result["stages"]) == len(result["tasks"]) == 4, "Cell identity mismatch")
        release = ROOT / f"ncd/results/problem_scout/section64/interim_local_v1/seed_{seed}/result.json"
        _, release_sha = read(release)
        published = ROOT / result["binding"]["training_result"]
        prior, published_sha = read(published)
        require(result["binding"]["release_sha256"] == release_sha and
                result["binding"]["publication_sha256"] == published_sha and
                result["binding"]["training_result_sha256"] == prior["publication_derivation"]["original_result_sha256"],
                "Published training source differs")
        for t in range(4):
            stage, stage_sha = read(folder / f"task_{t:03d}.json")
            binding = stage["binding"]["source"]
            require(binding["evaluation_source_sha256"] in SOURCES and
                    binding["device"] == SOURCES[binding["evaluation_source_sha256"]] and
                    all(binding[k] == result["binding"][k] for k in binding
                        if k not in ("device", "evaluation_source_sha256")), "Stage source mismatch")
            require(stage["completed"] and stage["default_cm_reproduced"] and stage["task_index"] == t
                    and stage["metrics"] == result["stages"][t]["metrics"]
                    and stage["calibration"]["target_fpr"] == .01
                    and stage["binding"]["checkpoint_sha256"] == prior["checkpoints"][t]["checkpoint_sha256"],
                    "Stage reproduction or calibration mismatch")
            stage["publication"] = dict(original_sha256=stage_sha, inference_repeated=False)
            prepared.append((dest / folder.name / f"task_{t:03d}.json", stage))
        result["publication"] = dict(original_sha256=digest, evaluation_attempts=attempts,
            cuda_supervisors=supervisors, diagnostics=diagnostics, inference_repeated=False,
            cost_scope="All attempts retained. Stage and child times are nested in supervisor times; do not add twice.",
            missing_complete_external_groups=["replay", "derpp"], final_tests_executed=False)
        prepared.append((dest / folder.name / "result.json", result))
    for _, value in prepared:
        require(not list(private_fields(value)), "Private metadata in publication")
    for path, value in prepared:
        write_derived(path, value)
    print("Published ten external cells / forty stages as fifty JSON files; no new inference or tests.")


if __name__ == "__main__":
    main()
