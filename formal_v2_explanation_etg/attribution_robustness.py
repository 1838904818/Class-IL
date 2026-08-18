#!/usr/bin/env python3
"""Source-bound attribution robustness pilot.

Expected Gradients are read from the completed formal analysis. Feature
ablation and Gradient x Input are computed against the same routed
``joint_cap3000`` class margin, frozen official-test probes, checkpoints, and
ETG thresholds. Integrated Gradients are retained as a diagnostic only and
must pass their completeness check before they are interpreted. The script
never trains or changes OFRA.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from formal_v2_explanation_etg.analyze import (  # noqa: E402
    JointMarginModel,
    PRIMARY_ALLOWED_RECALL_DROP,
    PRIMARY_JACCARD_THRESHOLD,
    _probe_rows,
    _softmax,
    _topk,
    build_etg_ledger,
    jaccard,
)
from streaming_full.data import canonical_sha256, load_manifest, sha256_file  # noqa: E402
from streaming_full.monitoring import (  # noqa: E402
    load_checkpoint,
    validate_checkpoint_manifest,
    validate_monitoring_result,
)


METHODS = ("expected_gradients", "feature_ablation", "gradient_x_input")
DIAGNOSTIC_METHODS = ("integrated_gradients",)
ALL_METHODS = METHODS + DIAGNOSTIC_METHODS


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def without_timing(value: object) -> object:
    if isinstance(value, dict):
        return {
            k: without_timing(v)
            for k, v in value.items()
            if k not in {"timing", "deterministic_result_sha256"}
        }
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def explain_gradient_x_input(
    model: torch.nn.Module,
    raw: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(raw), batch_size):
        values = torch.from_numpy(np.ascontiguousarray(raw[start : start + batch_size])).to(device)
        values.requires_grad_(True)
        score = model(values)
        gradient = torch.autograd.grad(score.sum(), values, create_graph=False)[0]
        chunks.append((gradient * values).detach().cpu().numpy().astype(np.float64))
        del values, score, gradient
    return np.vstack(chunks)


def explain_feature_ablation(
    model: torch.nn.Module,
    raw: np.ndarray,
    baseline: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return per-row score changes after replacing one feature at a time."""
    raw = np.asarray(raw, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32).reshape(-1)
    if raw.shape[1] != len(baseline):
        raise ValueError("feature-ablation baseline dimension mismatch")
    with torch.no_grad():
        original_parts: list[np.ndarray] = []
        for start in range(0, len(raw), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(raw[start : start + batch_size])).to(device)
            original_parts.append(model(batch).squeeze(1).cpu().numpy())
        original = np.concatenate(original_parts).astype(np.float64)
    values = np.empty(raw.shape, dtype=np.float64)
    features_per_batch = max(1, batch_size // len(raw))
    for feature_start in range(0, raw.shape[1], features_per_batch):
        features = list(
            range(feature_start, min(feature_start + features_per_batch, raw.shape[1]))
        )
        ablated = np.repeat(raw[None, :, :], len(features), axis=0)
        for local_index, feature in enumerate(features):
            ablated[local_index, :, feature] = baseline[feature]
        flat = np.ascontiguousarray(ablated.reshape(-1, raw.shape[1]))
        score_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(flat), batch_size):
                batch = torch.from_numpy(flat[start : start + batch_size]).to(device)
                score_parts.append(model(batch).squeeze(1).cpu().numpy())
        scores = np.concatenate(score_parts).reshape(len(features), len(raw)).astype(np.float64)
        values[:, features] = (original[None, :] - scores).T
    return values


def explain_integrated_gradients(
    model: torch.nn.Module,
    raw: np.ndarray,
    baseline: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if steps < 2:
        raise ValueError("integrated-gradients steps must be at least two")
    baseline = np.asarray(baseline, dtype=np.float32).reshape(1, -1)
    # Gauss-Legendre excludes both endpoints.  This is important for the
    # routed z-score because the background-mean endpoint can make all family
    # distances nearly equal and produce an ill-conditioned local derivative.
    nodes_np, weights_np = np.polynomial.legendre.leggauss(steps)
    alphas = torch.tensor((nodes_np + 1.0) / 2.0, dtype=torch.float32, device=device)
    weights = torch.tensor(weights_np / 2.0, dtype=torch.float32, device=device)
    chunks: list[np.ndarray] = []
    completeness_errors: list[float] = []
    for start in range(0, len(raw), batch_size):
        batch_np = np.ascontiguousarray(raw[start : start + batch_size])
        batch = torch.from_numpy(batch_np).to(device)
        base = torch.from_numpy(np.repeat(baseline, len(batch_np), axis=0)).to(device)
        delta = batch - base
        gradient_sum = torch.zeros_like(batch)
        for alpha, weight in zip(alphas, weights):
            point = (base + alpha * delta).detach().requires_grad_(True)
            score = model(point)
            gradient = torch.autograd.grad(score.sum(), point, create_graph=False)[0]
            gradient_sum += weight * gradient
            del point, score, gradient
        attribution = delta * gradient_sum
        with torch.no_grad():
            score_delta = (model(batch) - model(base)).squeeze(1)
            attr_sum = attribution.sum(dim=1)
            completeness_errors.extend((attr_sum - score_delta).abs().cpu().tolist())
        chunks.append(attribution.detach().cpu().numpy().astype(np.float64))
        del batch, base, delta, gradient_sum, attribution, score_delta, attr_sum
    errors = np.asarray(completeness_errors, dtype=np.float64)
    return np.vstack(chunks), {
        "mean_abs_completeness_error": float(errors.mean()),
        "max_abs_completeness_error": float(errors.max()),
    }


def selected_rationale_mass(checkpoint, raw: np.ndarray, class_id: int, features: np.ndarray) -> float:
    seen = [int(value) for value in checkpoint.metadata["seen_classes"]]
    column = seen.index(int(class_id))
    original = checkpoint.score(raw)["joint_scores"]
    q_original = _softmax(original)[:, column]
    masked = raw.copy()
    masked[:, features] = checkpoint.mean[features]
    q_masked = _softmax(checkpoint.score(masked)["joint_scores"])[:, column]
    return float(np.mean(q_original - q_masked))


def add_transitions(rows: list[dict[str, Any]], result: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = {(int(row["checkpoint"]), int(row["class_id"])): row for row in rows}
    transitions: list[dict[str, Any]] = []
    for checkpoint in range(1, len(result["checkpoints"])):
        previous = checkpoint - 1
        common = sorted(
            set(result["checkpoints"][previous]["seen_classes"])
            & set(result["checkpoints"][checkpoint]["seen_classes"])
        )
        for class_id in common:
            left = by_key[(previous, int(class_id))]
            right = by_key[(checkpoint, int(class_id))]
            overlap = jaccard(left["top15_indices"], right["top15_indices"])
            delta_recall = float(right["recall"] - left["recall"])
            eligible = delta_recall > -PRIMARY_ALLOWED_RECALL_DROP
            transitions.append(
                {
                    "from_checkpoint": previous,
                    "to_checkpoint": checkpoint,
                    "class_id": int(class_id),
                    "class_name": right["class_name"],
                    "recall_before": float(left["recall"]),
                    "recall_after": float(right["recall"]),
                    "delta_recall": delta_recall,
                    "jaccard_top15": float(overlap),
                    "primary_eligible": bool(eligible),
                    "primary_event": bool(eligible and overlap < PRIMARY_JACCARD_THRESHOLD),
                }
            )
    return transitions


def attach_ledger(rows: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger = build_etg_ledger(rows, transitions)
    ledger_map = {(int(x["checkpoint"]), int(x["class_id"])): x for x in ledger}
    for row in rows:
        item = ledger_map[(int(row["checkpoint"]), int(row["class_id"]))]
        row["etg_state"] = item["state_after"]
        row["etg_action"] = item["action"]
    return ledger


def agreement(method_rows: dict[str, list[dict[str, Any]]], method_transitions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    row_maps = {
        method: {(int(x["checkpoint"]), int(x["class_id"])): x for x in rows}
        for method, rows in method_rows.items()
    }
    transition_maps = {
        method: {
            (int(x["from_checkpoint"]), int(x["to_checkpoint"]), int(x["class_id"])): x
            for x in rows
        }
        for method, rows in method_transitions.items()
    }
    row_keys = sorted(set.intersection(*(set(value) for value in row_maps.values())))
    transition_keys = sorted(set.intersection(*(set(value) for value in transition_maps.values())))
    pairs: dict[str, Any] = {}
    for left, right in combinations(METHODS, 2):
        overlaps = [
            jaccard(row_maps[left][key]["top15_indices"], row_maps[right][key]["top15_indices"])
            for key in row_keys
        ]
        pairs[f"{left}__vs__{right}"] = {
            "row_count": len(row_keys),
            "top15_jaccard_mean": float(np.mean(overlaps)),
            "top15_jaccard_median": float(np.median(overlaps)),
            "top15_jaccard_min": float(np.min(overlaps)),
            "admission_decision_agreement": float(
                np.mean([row_maps[left][key]["admitted"] == row_maps[right][key]["admitted"] for key in row_keys])
            ),
            "etg_state_agreement": float(
                np.mean([row_maps[left][key]["etg_state"] == row_maps[right][key]["etg_state"] for key in row_keys])
            ),
            "silent_drift_event_agreement": float(
                np.mean([
                    transition_maps[left][key]["primary_event"] == transition_maps[right][key]["primary_event"]
                    for key in transition_keys
                ])
            ),
        }
    return {
        "common_checkpoint_class_rows": len(row_keys),
        "common_adjacent_class_transitions": len(transition_keys),
        "pairwise": pairs,
        "all_method_admission_agreement": float(
            np.mean([
                len({row_maps[m][key]["admitted"] for m in METHODS}) == 1 for key in row_keys
            ])
        ),
        "all_method_etg_state_agreement": float(
            np.mean([
                len({row_maps[m][key]["etg_state"] for m in METHODS}) == 1 for key in row_keys
            ])
        ),
        "all_method_silent_drift_conclusion_agreement": float(
            np.mean([
                len({transition_maps[m][key]["primary_event"] for m in METHODS}) == 1
                for key in transition_keys
            ])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-analysis", type=Path, required=True)
    parser.add_argument("--expected-attributions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ablation-batch-size", type=int, default=256)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument(
        "--reuse-integrated-diagnostic",
        type=Path,
        help="Reuse source-bound Integrated-Gradients rows from an earlier canonical run.",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = load_json(result_dir / "result_seed_1.json")
    protocol = load_json(result_dir / "protocol.json")
    protocol_without_hash = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    if protocol.get("protocol_sha256") != canonical_sha256(protocol_without_hash):
        raise RuntimeError("training protocol self-hash mismatch")
    if result.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise RuntimeError("training result/protocol mismatch")
    if result.get("deterministic_result_sha256") != canonical_sha256(without_timing(result)):
        raise RuntimeError("training result deterministic hash mismatch")
    validate_monitoring_result(result, output_base=result_dir, expected_protocol=protocol["monitoring"])

    manifest_path = cache_dir / "streaming_manifest.json"
    manifest = load_manifest(manifest_path, verify_hashes=True)
    if manifest.manifest_sha256 != protocol.get("manifest_sha256"):
        raise RuntimeError("local cache is not the hash-bound training cache")
    feature_schema = load_json(cache_dir / "feature_schema.json")
    feature_names = list(feature_schema["feature_columns"])
    probe_record = result["monitoring"]["probe_manifest"]
    probe_path = result_dir / str(probe_record["relative_path"])
    probe = load_json(probe_path)
    if sha256_file(probe_path) != probe_record["file_sha256"]:
        raise RuntimeError("probe manifest hash mismatch")
    official = _probe_rows(manifest, list(probe["official_test"]["samples"]), split="official_test")
    background = _probe_rows(
        manifest,
        list(probe["task0_train_background"]["samples"]),
        split="task0_train_background",
    )
    baseline = background.values.mean(axis=0, dtype=np.float64).astype(np.float32)

    expected_analysis = load_json(args.expected_analysis.resolve())
    if expected_analysis.get("canonical_sha256") != canonical_sha256(
        {k: v for k, v in expected_analysis.items() if k != "canonical_sha256"}
    ):
        raise RuntimeError("expected-gradients analysis canonical hash mismatch")
    expected_rows = [dict(row) for row in expected_analysis["checkpoint_rows"]]
    expected_archive = np.load(args.expected_attributions.resolve(), allow_pickle=False)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    method_rows: dict[str, list[dict[str, Any]]] = {"expected_gradients": expected_rows}
    attribution_arrays: dict[str, np.ndarray] = {}
    completeness: list[dict[str, Any]] = []
    reused_integrated: dict[str, Any] | None = None
    if args.reuse_integrated_diagnostic:
        diagnostic_path = args.reuse_integrated_diagnostic.resolve()
        diagnostic = load_json(diagnostic_path)
        if diagnostic.get("canonical_sha256") != canonical_sha256(
            {k: v for k, v in diagnostic.items() if k != "canonical_sha256"}
        ):
            raise RuntimeError("reused Integrated-Gradients diagnostic hash mismatch")
        if diagnostic.get("dataset") != manifest.dataset or int(diagnostic.get("seed", -1)) != 1:
            raise RuntimeError("reused Integrated-Gradients diagnostic scope mismatch")
        method_rows["integrated_gradients"] = [
            dict(row) for row in diagnostic["checkpoint_rows"]["integrated_gradients"]
        ]
        reused_integrated = {
            **dict(diagnostic["methods"]["integrated_gradients"]),
            "reused_source_file_sha256": sha256_file(diagnostic_path),
            "reused_source_canonical_sha256": diagnostic["canonical_sha256"],
        }
    expected_map = {(int(x["checkpoint"]), int(x["class_id"])): x for x in expected_rows}
    scheduled_methods = ["feature_ablation"]
    if reused_integrated is None:
        scheduled_methods.append("integrated_gradients")
    scheduled_methods.append("gradient_x_input")
    for method in scheduled_methods:
        rows: list[dict[str, Any]] = []
        for checkpoint_entry, result_checkpoint in zip(
            result["monitoring"]["checkpoints"], result["checkpoints"]
        ):
            checkpoint_index = int(checkpoint_entry["checkpoint"])
            checkpoint_manifest = result_dir / str(checkpoint_entry["checkpoint_manifest_relative_path"])
            metadata = validate_checkpoint_manifest(checkpoint_manifest)
            checkpoint = load_checkpoint(checkpoint_manifest, device=str(device))
            seen = [int(value) for value in metadata["seen_classes"]]
            seen_set = set(seen)
            mask = np.asarray(
                [int(record["class_id"]) in seen_set for record in official.records], dtype=bool
            )
            raw_seen = official.values[mask]
            record_seen = [r for keep, r in zip(mask.tolist(), official.records) if keep]
            recall = {
                int(row["class_id"]): float(row["recall"])
                for row in result_checkpoint["views"]["official"]["arms"]["joint_cap3000"]["per_class"]
            }
            for class_id in seen:
                class_mask = np.asarray(
                    [int(record["class_id"]) == class_id for record in record_seen], dtype=bool
                )
                class_raw = raw_seen[class_mask]
                model = JointMarginModel(checkpoint, class_id).to(device).eval()
                if method == "feature_ablation":
                    values = explain_feature_ablation(
                        model,
                        class_raw,
                        checkpoint.mean,
                        device=device,
                        batch_size=args.ablation_batch_size,
                    )
                elif method == "integrated_gradients":
                    values, error = explain_integrated_gradients(
                        model,
                        class_raw,
                        baseline,
                        device=device,
                        batch_size=args.batch_size,
                        steps=args.ig_steps,
                    )
                    completeness.append(
                        {"checkpoint": checkpoint_index, "class_id": class_id, **error}
                    )
                else:
                    values = explain_gradient_x_input(
                        model, class_raw, device=device, batch_size=args.batch_size
                    )
                mean_abs = np.mean(np.abs(values), axis=0)
                mean_signed = np.mean(values, axis=0)
                prefix = f"{method}__checkpoint_{checkpoint_index:03d}_class_{class_id:03d}"
                attribution_arrays[f"{prefix}_mean_abs"] = mean_abs
                attribution_arrays[f"{prefix}_mean_signed"] = mean_signed
                top15 = _topk(mean_abs, 15)
                expected_row = expected_map[(checkpoint_index, class_id)]
                mass = selected_rationale_mass(checkpoint, class_raw, class_id, top15)
                null = float(expected_row["random_null_95"])
                rows.append(
                    {
                        "dataset": manifest.dataset,
                        "seed": 1,
                        "checkpoint": checkpoint_index,
                        "class_id": class_id,
                        "class_name": expected_row["class_name"],
                        "probe_rows": len(class_raw),
                        "background_rows": len(background.values),
                        "recall": recall[class_id],
                        "rationale_mass": mass,
                        "random_null_95": null,
                        "mass_margin": mass - null,
                        "admitted": bool(mass > null),
                        "top15_indices": top15.tolist(),
                        "top15_features": [feature_names[index] for index in top15],
                    }
                )
                del model, values
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    json.dumps(
                        {
                            "event": "method_row_complete",
                            "method": method,
                            "checkpoint": checkpoint_index,
                            "class_id": class_id,
                        }
                    ),
                    flush=True,
                )
            del checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        method_rows[method] = rows

    method_transitions: dict[str, list[dict[str, Any]]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for method in ALL_METHODS:
        transitions = add_transitions(method_rows[method], result)
        ledger = attach_ledger(method_rows[method], transitions)
        method_transitions[method] = transitions
        ledgers[method] = ledger

    np.savez_compressed(output_dir / "attribution_robustness_mean_attributions.npz", **attribution_arrays)
    summary = {
        "schema_version": "ofra_attribution_robustness_v1",
        "status": "completed_source_bound_single_seed_pilot",
        "dataset": manifest.dataset,
        "seed": 1,
        "source_training_job": 388991,
        "source_etg_job": 389896,
        "score_target": "joint_cap3000 class margin",
        "methods": {
            "expected_gradients": {
                "source": "completed formal analysis",
                "analysis_file_sha256": sha256_file(args.expected_analysis.resolve()),
                "attributions_file_sha256": sha256_file(args.expected_attributions.resolve()),
            },
            "integrated_gradients": reused_integrated
            or {
                "baseline": "mean of the frozen Task-0 training background",
                "integration_rule": "Gauss-Legendre on [0,1]; endpoints excluded",
                "steps": args.ig_steps,
                "completeness": {
                    "row_count": len(completeness),
                    "mean_of_mean_abs_error": float(
                        np.mean([x["mean_abs_completeness_error"] for x in completeness])
                    ),
                    "maximum_abs_error": float(
                        np.max([x["max_abs_completeness_error"] for x in completeness])
                    ),
                },
            },
            "feature_ablation": {
                "definition": "single-feature replacement by the frozen checkpoint mean; attribution is the routed class-margin decrease"
            },
            "gradient_x_input": {
                "definition": "raw cached feature value multiplied by local score gradient"
            },
        },
        "thresholds": {
            "top_k": 15,
            "silent_drift_jaccard": PRIMARY_JACCARD_THRESHOLD,
            "allowed_recall_drop": PRIMARY_ALLOWED_RECALL_DROP,
            "admission": "selected-feature deletion mass exceeds the same fixed random-control 95th percentile used by Expected Gradients",
        },
        "agreement": agreement(method_rows, method_transitions),
        "method_summaries": {
            method: {
                "checkpoint_class_rows": len(method_rows[method]),
                "admitted_rows": sum(bool(x["admitted"]) for x in method_rows[method]),
                "silent_drift_events": sum(bool(x["primary_event"]) for x in method_transitions[method]),
                "eligible_transitions": sum(bool(x["primary_eligible"]) for x in method_transitions[method]),
                "certified_admissions": sum(x["action"] == "admission_certified" for x in ledgers[method]),
                "refused_admissions": sum(x["action"].startswith("admission_refused") for x in ledgers[method]),
                "escalations": sum(x["action"] == "human_review_escalation" for x in ledgers[method]),
                "strict_recertifications": sum(x["action"] == "strict_recertified" for x in ledgers[method]),
                "strict_recertification_failures": sum(
                    x["action"].startswith("strict_recertification_failed") for x in ledgers[method]
                ),
            }
            for method in ALL_METHODS
        },
        "checkpoint_rows": method_rows,
        "transition_rows": method_transitions,
        "guardrails": [
            "This is a MalayaNetwork_GT seed-1 robustness pilot, not a multi-seed estimate.",
            "The experiment measures method dependence; it does not identify a uniquely correct explainer.",
            "Integrated Gradients are a diagnostic only because the routed score failed the reported completeness check; they are excluded from the primary three-method agreement statistics.",
            "ETG remains an offline post-hoc ledger and does not change OFRA training or routing.",
        ],
        "source_bindings": {
            "training_result_file_sha256": sha256_file(result_dir / "result_seed_1.json"),
            "training_protocol_file_sha256": sha256_file(result_dir / "protocol.json"),
            "streaming_manifest_file_sha256": sha256_file(manifest_path),
            "feature_schema_file_sha256": sha256_file(cache_dir / "feature_schema.json"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    summary["canonical_sha256"] = canonical_sha256(summary)
    (output_dir / "attribution_robustness.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "canonical_sha256": summary["canonical_sha256"]}))


if __name__ == "__main__":
    main()
