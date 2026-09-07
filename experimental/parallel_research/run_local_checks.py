"""Run bounded local reference tests; never submit or execute an experiment."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TRACKS = ("fair_comparison", "prospective_sampling", "repair_controls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise SystemExit("Verification reports must be outside the source package")
    if output.exists():
        raise SystemExit("Refusing to overwrite a verification report")
    gates = json.loads((ROOT / "DEPENDENCY_GATES.json").read_text(encoding="utf-8"))
    assert gates["real_training_performed"] is False
    assert gates["historical_results_modified"] is False
    assert gates["new_hpc_submission_authorized"] is False
    assert not gates["experimental_reviews_closed_by_this_package"]
    assert tuple(row["id"] for row in gates["tracks"]) == TRACKS
    report = {"schema": "local-reference-checks-v1",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python_version": sys.version.split()[0],
              "scope": "synthetic tests and local contract checks only",
              "remote_actions": False, "real_data_integration_verified": False,
              "experimental_results": False, "tracks": [], "source_sha256": {}}
    for track in TRACKS:
        directory = ROOT / track
        test_directory = directory / "tests" if (directory / "tests").is_dir() else directory
        tests = sorted(test_directory.glob("test_*.py"))
        if not tests or not (directory / "README.md").is_file():
            report["tracks"].append({"id": track, "status": "BLOCKED_MISSING_FILES"})
            continue
        command = [sys.executable, "-B", "-m", "unittest", "discover", "-s",
                   str(test_directory), "-p", "test_*.py", "-v"]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=120)
            log = result.stdout + result.stderr
            count = re.search(r"Ran (\d+) tests? in", log)
            passed = result.returncode == 0 and count and int(count.group(1)) > 0
            report["tracks"].append({"id": track, "status": "PASS" if passed else "FAIL",
                                      "test_count": int(count.group(1)) if count else 0,
                                      "exit_code": result.returncode, "log": log})
        except subprocess.TimeoutExpired:
            report["tracks"].append({"id": track, "status": "TIMEOUT"})
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".md", ".json"}:
            report["source_sha256"][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["status"] = "PASS" if all(row["status"] == "PASS" for row in report["tracks"]) else "FAIL"
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "tracks": [
        {key: row[key] for key in ("id", "status", "test_count") if key in row}
        for row in report["tracks"]], "scope": report["scope"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
