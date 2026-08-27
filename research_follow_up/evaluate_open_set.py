"""Evaluate OFRA unknown rejection from a hash-bound monitoring checkpoint.

Thresholds are fitted on training-only calibration rows from
``build_train_protocol.py``. The held-out class is used only for evaluation.
The script reports confidence-only, distance-only, conservative-AND and an
empirical joint anomaly score. It does not create an unlabeled semantic head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
ALGORITHM = "ofra_known_calibrated_open_set_gate_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _binary_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be one-dimensional and aligned")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("binary metrics require both classes")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    change = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(scores)]
    tp = [0]
    fp = [0]
    thresholds = [np.inf]
    cumulative_tp = 0
    cumulative_fp = 0
    for start, end in zip(starts, ends):
        group = sorted_labels[start:end]
        cumulative_tp += int(group.sum())
        cumulative_fp += int(len(group) - group.sum())
        tp.append(cumulative_tp)
        fp.append(cumulative_fp)
        thresholds.append(float(sorted_scores[start]))
    return (
        np.asarray(fp, dtype=np.float64) / negatives,
        np.asarray(tp, dtype=np.float64) / positives,
        np.asarray(thresholds, dtype=np.float64),
    )


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = _binary_curve(labels, scores)
    return float(np.trapezoid(tpr, fpr))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    fpr, recall, _ = _binary_curve(labels, scores)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    tp = recall * positives
    fp = fpr * negatives
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    return float(np.sum(np.diff(recall) * precision[1:]))


def empirical_anomaly_scores(
    calibration_p: np.ndarray,
    calibration_d: np.ndarray,
    p_values: np.ndarray,
    d_values: np.ndarray,
) -> np.ndarray:
    sorted_p = np.sort(np.asarray(calibration_p, dtype=np.float64))
    sorted_d = np.sort(np.asarray(calibration_d, dtype=np.float64))
    if not len(sorted_p) or not len(sorted_d):
        raise ValueError("calibration signals cannot be empty")
    low_confidence = 1.0 - (
        np.searchsorted(sorted_p, p_values, side="right") / len(sorted_p)
    )
    large_distance = np.searchsorted(sorted_d, d_values, side="right") / len(sorted_d)
    return 0.5 * (low_confidence + large_distance)


def rejection_metrics(known_reject: np.ndarray, unknown_reject: np.ndarray) -> dict:
    known_reject = np.asarray(known_reject, dtype=bool)
    unknown_reject = np.asarray(unknown_reject, dtype=bool)
    tp = int(unknown_reject.sum())
    fn = len(unknown_reject) - tp
    fp = int(known_reject.sum())
    tn = len(known_reject) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    known_accept = tn / (tn + fp) if tn + fp else 0.0
    return {
        "known_rows": len(known_reject),
        "unknown_rows": len(unknown_reject),
        "true_unknown": tp,
        "false_unknown_on_known": fp,
        "unknown_precision": precision,
        "unknown_recall": recall,
        "known_false_unknown_rate": fp / (tn + fp) if tn + fp else 0.0,
        "known_acceptance_rate": known_accept,
        "open_set_balanced_accuracy": 0.5 * (recall + known_accept),
    }


def oscr_like_auc(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_correct: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    thresholds = np.unique(
        np.quantile(
            np.concatenate([known_scores, unknown_scores]),
            np.linspace(0.0, 1.0, 101),
        )
    )
    points: list[dict[str, float]] = []
    for threshold in thresholds:
        known_accepted = known_scores <= threshold
        unknown_accepted = unknown_scores <= threshold
        ccr = float(np.mean(known_correct & known_accepted))
        fpr = float(np.mean(unknown_accepted))
        points.append(
            {
                "anomaly_acceptance_threshold": float(threshold),
                "known_correct_classification_rate": ccr,
                "unknown_false_acceptance_rate": fpr,
            }
        )
    points.sort(key=lambda item: item["unknown_false_acceptance_rate"])
    x = np.asarray([item["unknown_false_acceptance_rate"] for item in points])
    y = np.asarray([item["known_correct_classification_rate"] for item in points])
    return float(np.trapezoid(y, x)), points


def _score_array(checkpoint: object, raw: np.ndarray, *, batch_size: int) -> dict[str, np.ndarray]:
    import torch

    values = np.asarray(raw, dtype=np.float64)
    seen = [int(value) for value in checkpoint.metadata["seen_classes"]]
    axis = np.asarray(seen, dtype=np.int64)
    p_max_parts: list[np.ndarray] = []
    d_min_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        normalized = ((batch - checkpoint.mean) / checkpoint.scale).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(normalized)).to(checkpoint.device)
        with torch.no_grad():
            embedding = checkpoint.encoder(tensor).cpu().numpy().astype(np.float32)
        embedding_tensor = torch.from_numpy(np.ascontiguousarray(embedding)).to(
            checkpoint.device
        )
        head = np.empty((len(batch), len(seen)), dtype=np.float32)
        with torch.no_grad():
            for column, class_id in enumerate(seen):
                logits = checkpoint.heads[class_id](embedding_tensor)
                head[:, column] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        raw_router = checkpoint.router._distances(
            embedding, checkpoint.router.cap, seen
        )
        router_mean = raw_router.mean(axis=1, keepdims=True)
        router_std = raw_router.std(axis=1, keepdims=True) + 1e-8
        router_z = (raw_router - router_mean) / router_std
        joint = head + np.float32(0.5) * router_z
        p_max_parts.append(head.max(axis=1))
        d_min_parts.append(-raw_router.max(axis=1))
        predicted_parts.append(axis[joint.argmax(axis=1)])
    return {
        "p_max": np.concatenate(p_max_parts),
        "d_min": np.concatenate(d_min_parts),
        "predicted": np.concatenate(predicted_parts),
    }


def _resolve_records(base: Path, records: object) -> list[Path]:
    if not isinstance(records, list) or not records:
        raise ValueError("shard record must be a non-empty list")
    paths: list[Path] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise TypeError("invalid shard record")
        path = (base / record["path"]).resolve()
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"shard hash mismatch: {path}")
        paths.append(path)
    return paths


def _score_paths(
    checkpoint: object,
    items: Iterable[tuple[int, Path]],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    p_max: list[np.ndarray] = []
    d_min: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    for class_id, path in items:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            for start in range(0, len(values), batch_size):
                result = _score_array(
                    checkpoint,
                    np.asarray(values[start : start + batch_size]),
                    batch_size=batch_size,
                )
                p_max.append(result["p_max"])
                d_min.append(result["d_min"])
                predicted.append(result["predicted"])
                truth.append(np.full(len(result["p_max"]), class_id, dtype=np.int64))
        finally:
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
    return {
        "p_max": np.concatenate(p_max),
        "d_min": np.concatenate(d_min),
        "predicted": np.concatenate(predicted),
        "truth": np.concatenate(truth),
    }


def _balanced_calibration_items(
    audit: dict,
    root: Path,
    seen: list[int],
    *,
    cap_per_class: int,
    seed: int,
) -> list[tuple[int, np.ndarray]]:
    by_class = {int(item["id"]): item for item in audit["classes"]}
    output: list[tuple[int, np.ndarray]] = []
    for class_id in seen:
        item = by_class[class_id]["calibration"]
        path = root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"calibration hash mismatch: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            rows = min(len(values), cap_per_class)
            class_seed = int.from_bytes(
                hashlib.sha256(f"{seed}|calibration|{class_id}".encode()).digest()[:8],
                "little",
            )
            rng = np.random.default_rng(class_seed)
            indices = np.sort(rng.choice(len(values), size=rows, replace=False))
            output.append((class_id, np.asarray(values[indices]).copy()))
        finally:
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
    return output


def _post_update_metrics(
    checkpoint: object,
    known_items: list[tuple[int, Path]],
    unknown_items: list[tuple[int, Path]],
    *,
    batch_size: int,
) -> dict:
    old = _score_paths(checkpoint, known_items, batch_size=batch_size)
    new = _score_paths(checkpoint, unknown_items, batch_size=batch_size)
    old_accuracy = float(np.mean(old["predicted"] == old["truth"]))
    new_recall = float(np.mean(new["predicted"] == new["truth"]))
    all_correct = np.concatenate(
        [old["predicted"] == old["truth"], new["predicted"] == new["truth"]]
    )
    return {
        "old_class_accuracy": old_accuracy,
        "new_class_recall": new_recall,
        "overall_accuracy": float(all_correct.mean()),
        "old_rows": len(old["truth"]),
        "new_rows": len(new["truth"]),
    }


def evaluate(
    *,
    runtime_root: Path,
    checkpoint_manifest: Path,
    manifest_path: Path,
    sampling_audit_path: Path,
    unknown_class_id: int,
    output_path: Path,
    calibration_cap_per_class: int = 5_000,
    calibration_seed: int = 42,
    batch_size: int = 4096,
    post_update_checkpoint_manifest: Path | None = None,
    candidate_min_support: int = 50,
    candidate_evaluation_purity: float = 0.90,
) -> dict:
    runtime_root = runtime_root.resolve()
    sys.path.insert(0, str(runtime_root))
    from streaming_full.monitoring import load_checkpoint

    checkpoint_manifest = checkpoint_manifest.resolve()
    manifest_path = manifest_path.resolve()
    sampling_audit_path = sampling_audit_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(sampling_audit_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(checkpoint_manifest, device="cpu")
    seen = [int(value) for value in checkpoint.metadata["seen_classes"]]
    if unknown_class_id in seen:
        raise ValueError("unknown class is already present in the pre-update checkpoint")
    classes = {int(item["id"]): item for item in manifest["classes"]}
    if unknown_class_id not in classes:
        raise ValueError("unknown class is absent from the manifest")
    manifest_root = manifest_path.parent

    calibration_arrays = _balanced_calibration_items(
        audit,
        sampling_audit_path.parent,
        seen,
        cap_per_class=calibration_cap_per_class,
        seed=calibration_seed,
    )
    calibration_results: list[dict[str, np.ndarray]] = []
    for _, values in calibration_arrays:
        calibration_results.append(
            _score_array(checkpoint, values, batch_size=batch_size)
        )
    calibration_p = np.concatenate([item["p_max"] for item in calibration_results])
    calibration_d = np.concatenate([item["d_min"] for item in calibration_results])
    tau_p = float(np.quantile(calibration_p, 0.05))
    tau_d = float(np.quantile(calibration_d, 0.95))
    calibration_joint = empirical_anomaly_scores(
        calibration_p, calibration_d, calibration_p, calibration_d
    )
    tau_joint = float(np.quantile(calibration_joint, 0.95))

    known_items: list[tuple[int, Path]] = []
    for class_id in seen:
        for path in _resolve_records(manifest_root, classes[class_id]["test"]):
            known_items.append((class_id, path))
    unknown_items = [
        (unknown_class_id, path)
        for path in _resolve_records(manifest_root, classes[unknown_class_id]["test"])
    ]
    known = _score_paths(checkpoint, known_items, batch_size=batch_size)
    unknown = _score_paths(checkpoint, unknown_items, batch_size=batch_size)
    known_joint = empirical_anomaly_scores(
        calibration_p, calibration_d, known["p_max"], known["d_min"]
    )
    unknown_joint = empirical_anomaly_scores(
        calibration_p, calibration_d, unknown["p_max"], unknown["d_min"]
    )

    rules = {
        "confidence_only": (
            known["p_max"] < tau_p,
            unknown["p_max"] < tau_p,
        ),
        "distance_only": (
            known["d_min"] > tau_d,
            unknown["d_min"] > tau_d,
        ),
        "conservative_and": (
            (known["p_max"] < tau_p) & (known["d_min"] > tau_d),
            (unknown["p_max"] < tau_p) & (unknown["d_min"] > tau_d),
        ),
        "empirical_joint": (
            known_joint > tau_joint,
            unknown_joint > tau_joint,
        ),
    }
    rule_metrics = {
        name: rejection_metrics(known_reject, unknown_reject)
        for name, (known_reject, unknown_reject) in rules.items()
    }
    labels = np.concatenate(
        [np.zeros(len(known_joint), dtype=np.int8), np.ones(len(unknown_joint), dtype=np.int8)]
    )
    continuous = {
        "head_low_confidence": np.concatenate([1.0 - known["p_max"], 1.0 - unknown["p_max"]]),
        "centroid_distance": np.concatenate([known["d_min"], unknown["d_min"]]),
        "empirical_joint": np.concatenate([known_joint, unknown_joint]),
    }
    ranking_metrics = {
        name: {
            "auroc": roc_auc(labels, scores),
            "aupr_unknown": average_precision(labels, scores),
        }
        for name, scores in continuous.items()
    }
    closed_correct = known["predicted"] == known["truth"]
    oscr_auc, oscr_points = oscr_like_auc(known_joint, unknown_joint, closed_correct)

    primary_known_reject, primary_unknown_reject = rules["conservative_and"]
    candidate_rows = int(primary_known_reject.sum() + primary_unknown_reject.sum())
    candidate_unknown_rows = int(primary_unknown_reject.sum())
    candidate_unknown_fraction = (
        candidate_unknown_rows / candidate_rows if candidate_rows else 0.0
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "evidence": {
            "runtime_root": str(runtime_root),
            "checkpoint_manifest": {
                "path": str(checkpoint_manifest),
                "sha256": sha256_file(checkpoint_manifest),
            },
            "derived_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "sampling_audit": {
                "path": str(sampling_audit_path),
                "sha256": sha256_file(sampling_audit_path),
            },
        },
        "protocol": {
            "unknown_class_id": unknown_class_id,
            "unknown_class_name": classes[unknown_class_id]["name"],
            "seen_classes": seen,
            "calibration_source": "balanced per-class training-only holdout",
            "calibration_cap_per_class": calibration_cap_per_class,
            "calibration_rows": len(calibration_p),
            "thresholds": {
                "p_max_lower_05": tau_p,
                "d_min_upper_95": tau_d,
                "empirical_joint_upper_95": tau_joint,
            },
            "primary_rule": "p_max < tau_p AND d_min > tau_d",
            "test_used_for_threshold_selection": False,
        },
        "closed_set_reference": {
            "known_rows": len(known["truth"]),
            "known_classification_accuracy": float(closed_correct.mean()),
            "unknown_rows_forced_to_a_seen_class": len(unknown["truth"]),
        },
        "rejection_rules": rule_metrics,
        "continuous_unknown_ranking": ranking_metrics,
        "oscr_style": {
            "auc": oscr_auc,
            "definition": (
                "area of known correct-classification rate versus unknown false-acceptance rate"
            ),
            "points": oscr_points,
        },
        "candidate_buffer_evaluation_only": {
            "primary_rule": "conservative_and",
            "rows": candidate_rows,
            "unknown_rows": candidate_unknown_rows,
            "known_contamination_rows": int(primary_known_reject.sum()),
            "unknown_fraction_using_ground_truth": candidate_unknown_fraction,
            "minimum_support": candidate_min_support,
            "evaluation_purity_threshold": candidate_evaluation_purity,
            "eligible_for_analyst_review_under_evaluation_oracle": bool(
                candidate_rows >= candidate_min_support
                and candidate_unknown_fraction >= candidate_evaluation_purity
            ),
            "deployment_caveat": (
                "ground-truth purity is unavailable in deployment; this field evaluates "
                "the candidate rule and never authorizes autonomous semantic head creation"
            ),
        },
    }
    if post_update_checkpoint_manifest is not None:
        post_path = post_update_checkpoint_manifest.resolve()
        post_checkpoint = load_checkpoint(post_path, device="cpu")
        if unknown_class_id not in [int(v) for v in post_checkpoint.metadata["seen_classes"]]:
            raise ValueError("post-update checkpoint does not contain the new class")
        post = _post_update_metrics(
            post_checkpoint,
            known_items,
            unknown_items,
            batch_size=batch_size,
        )
        post["pre_update_old_class_accuracy"] = float(closed_correct.mean())
        post["old_class_accuracy_change"] = (
            post["old_class_accuracy"] - float(closed_correct.mean())
        )
        post["checkpoint_manifest"] = {
            "path": str(post_path),
            "sha256": sha256_file(post_path),
        }
        result["post_labelled_head_update"] = post

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, result)
    result_hash = sha256_file(output_path)
    print(json.dumps({"output": str(output_path), "sha256": result_hash}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--post-update-checkpoint-manifest", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sampling-audit", type=Path, required=True)
    parser.add_argument("--unknown-class-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-cap-per-class", type=int, default=5_000)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--candidate-min-support", type=int, default=50)
    parser.add_argument("--candidate-evaluation-purity", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        runtime_root=args.runtime_root,
        checkpoint_manifest=args.checkpoint_manifest,
        post_update_checkpoint_manifest=args.post_update_checkpoint_manifest,
        manifest_path=args.manifest,
        sampling_audit_path=args.sampling_audit,
        unknown_class_id=args.unknown_class_id,
        output_path=args.output,
        calibration_cap_per_class=args.calibration_cap_per_class,
        calibration_seed=args.calibration_seed,
        batch_size=args.batch_size,
        candidate_min_support=args.candidate_min_support,
        candidate_evaluation_purity=args.candidate_evaluation_purity,
    )


if __name__ == "__main__":
    main()
