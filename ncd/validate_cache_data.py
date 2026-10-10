from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ncd.data import PreparedData, prepare_dataset


SUPPORTED_DATASETS = ("CIC-IDS-2017", "CSE-CIC-IDS2018")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_prepared(dataset: str, prepared: PreparedData) -> dict[str, object]:
    train_counts = np.bincount(prepared.y_train, minlength=len(prepared.class_order))
    test_counts = np.bincount(prepared.y_test, minlength=len(prepared.class_order))
    task0_mask = np.isin(prepared.y_train, prepared.tasks[0])
    if not np.all(train_counts > 0) or not np.all(test_counts > 0):
        raise RuntimeError(f"{dataset} contains an empty class")
    if not np.isfinite(prepared.x_train).all() or not np.isfinite(prepared.x_test).all():
        raise RuntimeError(f"{dataset} contains non-finite transformed values")
    task0_mean_max_abs = float(
        np.max(np.abs(np.mean(prepared.x_train[task0_mask], axis=0, dtype=np.float64)))
    )
    return {
        "dataset": dataset,
        "train_shape": list(prepared.x_train.shape),
        "test_shape": list(prepared.x_test.shape),
        "class_order": prepared.class_order,
        "train_class_counts": dict(zip(prepared.class_order, train_counts.astype(int).tolist())),
        "test_class_counts": dict(zip(prepared.class_order, test_counts.astype(int).tolist())),
        "tasks": prepared.tasks,
        "task0_train_rows": int(task0_mask.sum()),
        "task0_feature_mean_max_abs": task0_mean_max_abs,
        "feature_count": len(prepared.feature_names),
        "finite": True,
        "source_files": [
            {
                "path": str(Path(source).resolve(strict=True)),
                "sha256": _sha256(Path(source)),
            }
            for source in prepared.source_files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen CIC cache loading on a compute node")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries: list[dict[str, object]] = []
    for dataset in args.datasets:
        print(f"CACHE_VALIDATION_START dataset={dataset}", flush=True)
        prepared = prepare_dataset(dataset, args.data_root)
        summary = summarize_prepared(dataset, prepared)
        summaries.append(summary)
        print(
            "CACHE_VALIDATION_OK "
            f"dataset={dataset} train={summary['train_shape']} test={summary['test_shape']} "
            f"features={summary['feature_count']}",
            flush=True,
        )

    payload = {
        "status": "validated",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "source_commit": os.environ.get("NCD_SOURCE_COMMIT"),
        "datasets": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"CACHE_VALIDATION_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
