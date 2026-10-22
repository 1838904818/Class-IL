"""Package tracked capacity code, including transitive src imports, as LF bytes."""
import argparse
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to replace an earlier package")
    paths = subprocess.check_output(["git", "ls-files", "ncd", "src"], cwd=ROOT, text=True).splitlines()
    selected = [path for path in paths if
        (path.endswith(".py") and not path.startswith("ncd/results/")) or path in
        ("ncd/configs/problem_scout_section21.json", "ncd/hpc/run_section81_capacity.sbatch", "ncd/hpc/submit_checked.sh")]
    required = {"src/methods/replay_dpmeans.py", "src/config.py", "src/methods/base.py", "src/metrics.py"}
    if not required.issubset(selected):
        raise RuntimeError("Missing transitive source dependencies")
    subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=tar",
        "--output=" + str(args.output.resolve()), "HEAD", "--", *selected], cwd=ROOT, check=True)
    with tarfile.open(args.output) as archive:
        for path in selected:
            blob = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT)
            if archive.extractfile(path).read() != blob:
                raise RuntimeError("Packaged bytes differ from committed source: " + path)
    print(f"Packaged and verified {len(selected)} source/config files")


if __name__ == "__main__":
    main()
