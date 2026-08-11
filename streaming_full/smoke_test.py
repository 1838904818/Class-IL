from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .data import (
    BlockShuffleSampler,
    ClassShards,
    MatrixReservoir,
    canonical_sha256,
    load_manifest,
    sha256_file,
)
from .routers import bounded_dpmeans
from .summarize import summarize_campaign
from .validation import ARMS, RunConfig, run_manifest


def _write_shard(path: Path, values: np.ndarray) -> dict:
    np.save(path, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return {"path": path.name, "rows": int(len(values)), "sha256": sha256_file(path)}


def _make_synthetic_manifest(root: Path) -> Path:
    rng = np.random.default_rng(20260714)
    feature_dim = 6
    class_names = ["Normal", "DoS", "Probe", "Botnet"]
    classes = []
    for class_id, name in enumerate(class_names):
        center = np.zeros(feature_dim, dtype=np.float32)
        center[class_id] = 3.0
        center[-1] = class_id * 0.75
        train = rng.normal(center, 0.55, size=(48, feature_dim)).astype(np.float32)
        test = rng.normal(center, 0.55, size=(20, feature_dim)).astype(np.float32)
        train_shards = [
            _write_shard(root / f"class_{class_id}_train_{part}.npy", block)
            for part, block in enumerate(np.array_split(train, 2))
        ]
        test_shards = [
            _write_shard(root / f"class_{class_id}_test_{part}.npy", block)
            for part, block in enumerate(np.array_split(test, 2))
        ]
        classes.append(
            {
                "id": class_id,
                "name": name,
                "train": train_shards,
                "test": test_shards,
            }
        )
    overlap = {
        "dataset": "synthetic_streaming_smoke_only",
        "exact_cross_split_overlap_unique_feature_rows": 0,
    }
    overlap["canonical_report_sha256"] = canonical_sha256(overlap)
    overlap_path = root / "split_overlap_audit.json"
    overlap_path.write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    overlap_sha256 = sha256_file(overlap_path)
    builder_source = {"algorithm": "synthetic_smoke_builder_source_v1", "files": []}
    builder_source["canonical_source_sha256"] = canonical_sha256(builder_source)
    feature_schema_sha256 = "a" * 64
    manifest = {
        "schema_version": 1,
        "dataset": "synthetic_streaming_smoke_only",
        "feature_dim": feature_dim,
        "normal_class_id": 0,
        "source": {
            "builder": "ofra-fullcache",
            "builder_version": "smoke-v1",
            "feature_schema_sha256": feature_schema_sha256,
            "class_cap": None,
            "builder_source_sha256": builder_source["canonical_source_sha256"],
            "split_overlap_audit_sha256": overlap_sha256,
        },
        "tasks": [[0, 1], [2, 3]],
        "classes": classes,
    }
    path = root / "streaming_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fullcache = {
        "format_version": 1,
        "tool": "ofra-fullcache",
        "tool_version": "smoke-v1",
        "dataset": "synthetic_streaming_smoke_only",
        "uncapped": True,
        "builder_source": builder_source,
        "feature_schema_sha256": feature_schema_sha256,
        "sidecars": {
            "streaming_manifest.json": sha256_file(path),
            "split_overlap_audit.json": overlap_sha256,
        },
        "streaming_runner": {
            "compatible": True,
            "relative_manifest": "streaming_manifest.json",
            "schema_version": 1,
            "normal_class_id": 0,
            "tasks": [[0, 1], [2, 3]],
        },
    }
    fullcache["canonical_manifest"] = {
        "algorithm": "sha256-canonical-json-excluding-this-field-v1",
        "sha256": canonical_sha256(fullcache),
    }
    (root / "manifest.json").write_text(json.dumps(fullcache, indent=2), encoding="utf-8")
    return path


def _assert_run(output: dict) -> None:
    result = output["results"][0]
    assert output["protocol"]["normal_class"] == {
        "class_id": 0,
        "class_name": "Normal",
        "source": "explicit manifest normal_class_id",
    }
    provenance = output["protocol"]["input_source_provenance"]
    assert provenance["builder"] == "ofra-fullcache"
    assert Path(provenance["fullcache_manifest"]["path"]).name == "manifest.json"
    assert len(provenance["fullcache_manifest"]["sha256"]) == 64
    assert result["normalization"]["count"] == 96
    assert result["normalization"]["source_classes"] == [0, 1]
    pretrain_sampler = result["pretrain_history"][0]["sampler"]
    assert pretrain_sampler["algorithm"] == (
        "proportional_cumulative_quota_over_per_class_block_samplers_v1"
    )
    assert pretrain_sampler["mixed_batches"] == 6
    assert pretrain_sampler["single_class_batches"] == 0
    assert pretrain_sampler["class_rows_seen"] == {"0": 48, "1": 48}
    assert sum(
        record["quota_rows"] for record in pretrain_sampler["class_samplers"].values()
    ) == 96
    for class_history in result["training_history"].values():
        epoch = class_history[0]
        assert epoch["positive_sampler"]["sample_rows"] == epoch["positive_rows"]
        assert epoch["negative_sampler"]["sample_rows"] == epoch["negative_rows"]
        assert epoch["negative_rows"] <= 4 * epoch["positive_rows"]
    assert len(result["checkpoints"]) == 2
    expected_rows = [40, 80]
    for checkpoint, total_rows in zip(result["checkpoints"], expected_rows):
        official = checkpoint["views"]["official"]
        assert set(official["arms"]) == set(ARMS)
        for metrics in official["arms"].values():
            confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            assert int(confusion.sum()) == total_rows
            binary = metrics["binary_detection"]
            assert binary["normal_class_id"] == 0
            assert binary["normal_class_name"] == "Normal"
            assert binary["benign_support"] == 20
            assert binary["attack_support"] == total_rows - 20
            expected_fpr = (confusion[0].sum() - confusion[0, 0]) / confusion[0].sum()
            expected_recall = confusion[1:, 1:].sum() / confusion[1:].sum()
            assert np.isclose(binary["benign_false_positive_rate"], expected_fpr)
            assert np.isclose(binary["attack_detection_recall"], expected_recall)
    for arm in ARMS:
        matrix = result["summary"]["views"]["official"][arm][
            "task_accuracy_matrix"
        ]
        assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)
        assert matrix[0][0] is not None and matrix[0][1] is None
        assert matrix[1][0] is not None and matrix[1][1] is not None
    for class_id in range(4):
        router = result["router_records"][str(class_id)]
        assert router["cap3000"]["sample_total"] == 48
        assert router["cap3000"]["sample_used"] == 20
        assert router["cap3000"]["discovered_count_sum"] == 20
        assert router["cap3000"]["retained_centroid_count"] <= 4
        assert router["cap3000"]["retained_count_sum"] <= 20
        assert router["cap3000"]["count_sum"] == 20
        assert router["cap3000"]["selected_index_count"] == 20
        assert router["uncapped"]["sample_total"] == 48
        assert router["uncapped"]["sample_used"] == 48
        assert router["uncapped"]["count_sum"] == 48
        assert router["uncapped"]["base_cap_rows"] == 20
        assert router["uncapped"]["added_rows"] == 28
        assert router["uncapped"]["selected_rows_skipped"] == 20
        assert router["uncapped"]["complement_rows_expected"] == 28
        assert router["uncapped"]["complement_refinement_passes"] == 1
        assert router["uncapped"]["router_construction_passes_over_class_shards"] == 2
        assert (
            router["uncapped"]["shared_initial_centroid_sha256"]
            == router["cap3000"]["centroid_sha256"]
        )
        assert (
            router["uncapped"]["shared_initial_count_sha256"]
            == router["cap3000"]["count_sha256"]
        )
        assert router["uncapped"]["shared_initial_lambda"] == router["cap3000"]["lambda"]
        assert (
            router["uncapped"]["cap_selected_index_sha256"]
            == router["cap3000"]["selected_index_sha256"]
        )
        exemplar = result["exemplar_records"][str(class_id)]
        assert exemplar["candidate_reservoir"]["seen"] == 48
        assert exemplar["candidate_reservoir"]["retained"] == 24
        assert exemplar["selected_rows"] == 4
        assert len(exemplar["selected_normalized_sha256"]) == 64


def _assert_cap_cover_identity(output: dict) -> None:
    result = output["results"][0]
    for router in result["router_records"].values():
        cap = router["cap3000"]
        uncapped = router["uncapped"]
        assert cap["sample_total"] == cap["sample_used"] == 48
        assert cap["selected_index_count"] == 48
        assert uncapped["base_cap_rows"] == 48
        assert uncapped["added_rows"] == 0
        assert uncapped["selected_rows_skipped"] == 48
        assert uncapped["complement_rows_expected"] == 0
        assert uncapped["state_identical_to_cap"] is True
        assert uncapped["centroid_sha256"] == cap["centroid_sha256"]
        assert uncapped["count_sha256"] == cap["count_sha256"]
        assert uncapped["count_sum"] == cap["count_sum"] == 48
    for checkpoint in result["checkpoints"]:
        arms = checkpoint["views"]["official"]["arms"]
        assert (
            arms["router_only_cap3000"]["confusion_matrix"]
            == arms["router_only_uncapped"]["confusion_matrix"]
        )
        assert (
            arms["joint_cap3000"]["confusion_matrix"]
            == arms["joint_uncapped"]["confusion_matrix"]
        )


def _assert_class_shard_io(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path, verify_hashes=True)
    record = manifest.classes[0]
    expected = np.vstack([np.load(shard.path, allow_pickle=False) for shard in record.train])
    view = ClassShards(record.train, manifest.feature_dim, max_open_memmaps=1)
    indices = np.asarray([47, 0, 24, 24, 23, 25, 1], dtype=np.int64)
    assert np.array_equal(view.take(indices), expected[indices])
    assert view.cached_shard_count <= 1
    streamed = np.vstack(list(view.batches(7)))
    assert np.array_equal(streamed, expected)
    assert view.cached_shard_count <= 1
    blocks, _ = view.index_blocks(7)
    first = BlockShuffleSampler(
        blocks,
        population_rows=len(view),
        sample_rows=len(view),
        seed=101,
        block_rows=7,
    )
    second = BlockShuffleSampler(
        blocks,
        population_rows=len(view),
        sample_rows=len(view),
        seed=101,
        block_rows=7,
    )
    first_indices = np.concatenate(list(first.iter_chunks(5)))
    second_indices = np.concatenate(list(second.iter_chunks(5)))
    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(np.sort(first_indices), np.arange(len(view)))
    subset = BlockShuffleSampler(
        blocks,
        population_rows=len(view),
        sample_rows=17,
        seed=202,
        block_rows=7,
    )
    subset_indices = np.concatenate(list(subset.iter_chunks(5)))
    assert len(subset_indices) == len(np.unique(subset_indices)) == 17
    view.close()
    assert view.cached_shard_count == 0


def _assert_dpmeans_topk() -> None:
    centers = np.arange(40, dtype=np.float32)[:, None] * 10.0
    values = np.vstack([centers, centers + 0.01]).astype(np.float32)
    centroids, counts, _, stats = bounded_dpmeans(
        values,
        quantile=0.001,
        max_centroids=32,
        max_iter=4,
        rng=np.random.default_rng(71),
    )
    assert stats["discovered_centroid_count"] > 32
    assert stats["retained_centroid_count"] == 32
    assert stats["discovered_count_sum"] == len(values)
    assert stats["retained_count_sum"] == int(counts.sum())
    assert len(centroids) == len(counts) == 32


def _scalar_algorithm_r(
    values: np.ndarray, capacity: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    retained = np.empty((capacity, values.shape[1]), dtype=np.float32)
    retained_indices = np.empty(capacity, dtype=np.int64)
    for seen, row in enumerate(values, start=1):
        if seen <= capacity:
            retained[seen - 1] = row
            retained_indices[seen - 1] = seen - 1
        else:
            slot = int(rng.integers(0, seen))
            if slot < capacity:
                retained[slot] = row
                retained_indices[slot] = seen - 1
    return retained, retained_indices


def _assert_vectorized_reservoir() -> None:
    values = np.arange(400, dtype=np.float32).reshape(200, 2)
    expected, expected_indices = _scalar_algorithm_r(values, 11, 29)
    first = MatrixReservoir(11, 2, np.random.default_rng(29))
    first.update(values)
    second = MatrixReservoir(11, 2, np.random.default_rng(29))
    for block in np.array_split(values, [17, 83, 151]):
        second.update(block)
    assert first.seen == second.seen == len(values)
    assert first.retained == second.retained == 11
    assert np.array_equal(first.array(), expected)
    assert np.array_equal(second.array(), expected)
    assert np.array_equal(first.indices(), expected_indices)
    assert np.array_equal(second.indices(), expected_indices)
    assert len(np.unique(first.indices())) == first.retained
    assert first.record()["algorithm"] == "reservoir_algorithm_r_vectorized_variable_high_v2"

    inclusion = np.zeros(20, dtype=np.int64)
    population = np.arange(20, dtype=np.float32)[:, None]
    trials = 1000
    for seed in range(trials):
        reservoir = MatrixReservoir(5, 1, np.random.default_rng(seed))
        reservoir.update(population)
        inclusion[reservoir.array().astype(np.int64).ravel()] += 1
    expected_count = trials * 5 / len(population)
    assert np.max(np.abs(inclusion - expected_count)) < 6.0 * np.sqrt(
        trials * 0.25 * 0.75
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ofra_streaming_smoke_") as temporary:
        root = Path(temporary)
        manifest = _make_synthetic_manifest(root)
        _assert_class_shard_io(manifest)
        _assert_dpmeans_topk()
        _assert_vectorized_reservoir()
        config = RunConfig(
            pretrain_epochs=1,
            epochs_per_task=1,
            batch_size=16,
            eval_batch_size=16,
            learning_rate=2e-3,
            d_model=12,
            n_layers=2,
            lora_rank=2,
            lora_alpha=4.0,
            minority_threshold=1000,
            negative_ratio=4,
            exemplar_capacity=4,
            exemplar_candidate_capacity=24,
            router_cap_samples=20,
            router_lambda_quantile=0.30,
            router_max_centroids=4,
            device="cpu",
            deterministic=True,
            verify_shard_hashes=True,
            verbose=False,
        )
        first = run_manifest(manifest, seeds=[17], output_dir=root / "first", config=config)
        second = run_manifest(manifest, seeds=[17], output_dir=root / "second", config=config)
        _assert_run(first)
        _assert_run(second)
        first_hash = first["results"][0]["deterministic_result_sha256"]
        second_hash = second["results"][0]["deterministic_result_sha256"]
        assert first["summary"]["protocol_sha256"] == second["summary"]["protocol_sha256"]
        assert first_hash == second_hash
        result_path = root / "first" / "result_seed_17.json"
        modification_time = result_path.stat().st_mtime_ns
        resumed = run_manifest(manifest, seeds=[17], output_dir=root / "first", config=config)
        assert result_path.stat().st_mtime_ns == modification_time
        assert resumed["results"][0]["deterministic_result_sha256"] == first_hash
        assert not list((root / "first").glob("*.tmp"))
        mismatched = RunConfig(**{**config.__dict__, "epochs_per_task": 2})
        try:
            run_manifest(manifest, seeds=[17], output_dir=root / "first", config=mismatched)
        except RuntimeError as error:
            assert "protocol differs" in str(error)
        else:
            raise AssertionError("resume must fail closed when the protocol changes")
        cap_cover_config = RunConfig(
            **{**config.__dict__, "router_cap_samples": 64}
        )
        cap_cover = run_manifest(
            manifest,
            seeds=[19],
            output_dir=root / "cap_cover",
            config=cap_cover_config,
        )
        _assert_cap_cover_identity(cap_cover)
        campaign_seeds = [11, 12, 13, 14, 15]
        campaign_root = root / "campaign"
        run_manifest(
            manifest,
            seeds=campaign_seeds,
            output_dir=campaign_root / "synthetic",
            config=config,
        )
        paired = summarize_campaign(
            campaign_root,
            campaign_root / "paired_report.json",
            expected_seeds=campaign_seeds,
            expected_datasets=["synthetic_streaming_smoke_only"],
        )
        dataset_report = paired["datasets"]["synthetic_streaming_smoke_only"]
        comparison = dataset_report["views"]["official"][
            "primary_arm_comparisons"
        ][
            "joint_uncapped_minus_joint_cap3000"
        ]
        metric = comparison["metrics"]["final_macro_f1"]
        assert metric["n_pairs"] == 5 and len(metric["pairs"]) == 5
        assert "p_value_holm" in metric["paired_t"]
        assert "p_value_holm" in metric["wilcoxon_exact"]
        assert Path(metric["pairs"][0]["source_json"]).is_file()
        assert len(paired["campaign_report_sha256"]) == 64
        print(
            json.dumps(
                {
                    "status": "ok",
                    "dataset": "synthetic_streaming_smoke_only",
                    "protocol_sha256": first["summary"]["protocol_sha256"],
                    "deterministic_result_sha256": first_hash,
                    "real_data_used": False,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
