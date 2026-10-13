"""Read-only independent recomputation of Exp H Section 17.5/17.6.

No project statistics helpers are imported. Rank residuals use a scalar
centered-rank projection, independent of the production least-squares path.
All output is stdout; input artifacts and historical results are not changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits

METHODS = ("replay", "icarl", "replay_lwf", "kd_perturbed", "rcl_tab",
           "global_ir", "oldpair_kd", "crr")
SEEDS = (1, 2, 3, 4, 42)
METRICS = ("old_pair_pi_drift", "self_margin_pi_drift")
RECALLS = ("shift_recall_before", "shift_recall_after",
           "id_recall_before", "id_recall_after")


def identity(row, retention=False):
    fields = ("setting", "method", "seed", "class_id") if retention else (
        "setting", "method", "seed", "transition", "class_id")
    return tuple(str(row[key]) for key in fields)


def indexed(rows, retention=False):
    result = {identity(row, retention): row for row in rows}
    assert len(result) == len(rows), "Duplicate row identity"
    return result


def compare_rows(left, right, retention=False):
    left, right = indexed(left, retention), indexed(right, retention)
    assert left.keys() == right.keys(), "Row identity sets differ"
    largest = 0.0
    for key, row in left.items():
        other = right[key]
        assert row.keys() == other.keys(), ("Different row fields", key)
        for field, value in row.items():
            candidate = other[field]
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                a, b = float(value), float(candidate)
                assert math.isfinite(a) and math.isfinite(b), (key, field)
                largest = max(largest, abs(a-b))
                assert abs(a-b) <= 1e-12, (key, field, a, b)
            elif field == "pi_vectors_file":
                # Publication may rewrite a directory prefix, not artifact identity.
                a = str(value).replace("\\", "/").rsplit("/", 1)[-1]
                b = str(candidate).replace("\\", "/").rsplit("/", 1)[-1]
                assert a == b and a, (key, field, value, candidate)
            else:
                assert str(value) == str(candidate), (key, field, value, candidate)
    return largest



def resolve_shard(base, cell):
    """Resolve relocated/redacted prefixes by the exact setting/method/seed suffix.

    No paths outside the selected input directory are read. A missing canonical
    shard is an error even if the old published or local reference still exists.
    """
    setting, method, seed = cell["setting"], cell["method"], cell["seed"]
    assert setting in ("S1", "S2", "S3") and method in METHODS and seed in SEEDS
    relative = Path(setting) / "shards" / f"{method}_seed{seed}" / "exp_h_result.json"
    reference = str(cell["result"]).replace("\\", "/").rstrip("/")
    suffix = relative.as_posix()
    assert reference == suffix or reference.endswith("/" + suffix), (
        "Shard reference must retain its canonical identity suffix", reference, suffix)
    path = (base / relative).resolve()
    assert base == path or base in path.parents, "Shard escaped selected input directory"
    if not path.is_file():
        raise FileNotFoundError(f"Required shard missing: {relative.as_posix()}")
    return path


def partial_rank(x, decline, z):
    """Average ties; remove intercept by centering, then project onto rank Z."""
    rx, ry, rz = (rankdata(a, method="average") for a in (x, decline, z))
    rx, ry, rz = (a - a.mean() for a in (rx, ry, rz))
    zz = float(rz @ rz)
    if zz:
        rx = rx - rz * float(rx @ rz) / zz
        ry = ry - rz * float(ry @ rz) / zz
    denominator = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry) / denominator if denominator > 1e-12 else None


def summarize_units(rows, metric, draws, random_seed):
    x = np.array([r[metric] for r in rows], dtype=float)
    decline = np.array([r["shift_recall_before"]-r["shift_recall_after"] for r in rows])
    z = np.array([r["id_recall_after"]-r["id_recall_before"] for r in rows])
    labels = np.array([r["seed"] for r in rows])
    assert set(labels) == set(SEEDS)
    clusters = [np.where(labels == seed)[0] for seed in SEEDS]
    generator = np.random.default_rng(random_seed)
    estimates = []
    sampled_sizes = []
    for _ in range(draws):
        # Five cluster indices; every selected seed's rows repeated as a block.
        selected = generator.integers(0, 5, size=5)
        ids = np.concatenate([clusters[i] for i in selected])
        sampled_sizes.append(len(ids))
        estimates.append(partial_rank(x[ids], decline[ids], z[ids]))
    valid = [v for v in estimates if v is not None and math.isfinite(v)]
    assert len(valid) == draws, "Undefined bootstrap draws; no formal decision"
    return {
        "rho": partial_rank(x, decline, z),
        "ci95": np.percentile(valid, [2.5, 97.5], method="linear").tolist(),
        "bootstrap_valid": len(valid), "bootstrap_seed": random_seed,
        "bootstrap_size_range": [min(sampled_sizes), max(sampled_sizes)],
    }


def method_summary(rows, retention):
    """Matched old-class scope exactly stated in the existing report.

    An eligible (method, seed, class) endpoint appears once irrespective of
    transition count. Class ratios are averaged within seed, then across seeds.
    """
    lookup = indexed(retention, retention=True)
    result = []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        means = {metric: float(np.mean([r[metric] for r in selected])) for metric in METRICS}
        seed_values, class_seed_count = [], 0
        for seed in SEEDS:
            class_ids = {r["class_id"] for r in selected if r["seed"] == seed}
            assert class_ids
            ratios = []
            for class_id in sorted(class_ids):
                endpoint = lookup[(rows[0]["setting"], method, str(seed), str(class_id))]
                final, best = endpoint["shift_recall_final"], endpoint["shift_recall_best"]
                assert 0 <= final <= best <= 1 and best > 0
                ratios.append(final / best)
            seed_values.append(float(np.mean(ratios)))
            class_seed_count += len(ratios)
        result.append({"method": method,
                       **{"mean_"+metric: means[metric] for metric in METRICS},
                       "mean_shift_retention": float(np.mean(seed_values)),
                       "retention_class_seed_units": class_seed_count,
                       "drift_units": len(selected)})
    rhos = {metric: float(spearmanr([r["mean_"+metric] for r in result],
                                   [r["mean_shift_retention"] for r in result]).statistic)
            for metric in METRICS}
    return result, rhos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path(__file__).resolve().parent / "results/exp_h/section17/registered",
        help="Published registered directory, or another complete run such as local_v1",
    )
    args = parser.parse_args()
    base = args.input_dir.resolve()
    paths = [base / name for name in ("exp_h_result.json", "exp_h_analysis.json",
                                     "exp_h_rows.csv", "exp_h_retention_rows.csv")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    raw = json.loads(paths[0].read_bytes())
    generated = json.loads(paths[1].read_bytes())
    assert hashes["exp_h_result.json"] == generated["input_sha256"], (
        "Analysis input hash does not match the selected raw JSON; path redaction "
        "requires an explicitly rebound published input hash, not a hash bypass")
    assert raw["git_commit"] == generated["provenance"]["git_commit"]
    assert not raw["smoke"]
    assert set(raw["methods"]) == set(METHODS) and set(raw["seeds"]) == set(SEEDS)
    expected_cells = {(s, m, n) for s in ("S1", "S2", "S3") for m in METHODS for n in SEEDS}
    actual_cells = {(c["setting"], c["method"], c["seed"]) for c in raw["cells"]}
    assert len(raw["cells"]) == 120 and actual_cells == expected_cells
    with paths[2].open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    with paths[3].open(encoding="utf-8-sig", newline="") as handle:
        csv_retention = list(csv.DictReader(handle))
    row_delta = compare_rows(raw["rows"], csv_rows)
    ret_delta = compare_rows(raw["retention_rows"], csv_retention, True)
    shard_rows, shard_retention, shard_hashes = [], [], {}
    for cell in raw["cells"]:
        shard_path = resolve_shard(base, cell)
        shard_bytes = shard_path.read_bytes()
        shard_hashes[shard_path.relative_to(base).as_posix()] = hashlib.sha256(shard_bytes).hexdigest()
        shard = json.loads(shard_bytes)
        assert shard["git_commit"] == raw["git_commit"], "Shard source commit differs"
        assert all((r["setting"], r["method"], r["seed"]) ==
                   (cell["setting"], cell["method"], cell["seed"]) for r in
                   shard["rows"] + shard["retention_rows"]), "Shard identity differs"
        shard_rows.extend(shard["rows"])
        shard_retention.extend(shard["retention_rows"])
    assert len(shard_hashes) == 120, "All 120 canonical shards are required"
    shard_delta = compare_rows(raw["rows"], shard_rows)
    shard_ret_delta = compare_rows(raw["retention_rows"], shard_retention, True)
    config = raw["effective_config"]
    draws, bootstrap_seed = config["bootstrap_replicates"], config["bootstrap_seed"]
    assert draws >= 2000 and config["floor_recall"] == .2
    findings = {}
    max_stat_discrepancy = 0.0
    for position, setting in enumerate(("S1", "S2", "S3")):
        all_rows = [r for r in raw["rows"] if r["setting"] == setting]
        for row in all_rows:
            assert all(math.isfinite(row[k]) for k in METRICS+RECALLS)
            assert all(0 <= row[k] <= 1 for k in RECALLS)
        cells = defaultdict(set)
        for row in all_rows:
            cells[(row["method"],row["seed"])].add((row["transition"],row["class_id"]))
        assert len(cells) == 40
        unit_sets = list(cells.values())
        assert all(u == unit_sets[0] for u in unit_sets)
        assert len(unit_sets[0]) == raw["setting_metadata"][setting]["expected_units_per_cell"]
        retained = [r for r in all_rows if r["shift_recall_before"] >= .2]
        floor = [r for r in all_rows if r["shift_recall_before"] < .2]
        published = generated["settings"][setting]
        counts = {"raw_units":len(all_rows), "floor_excluded_units":len(floor),
                  "included_units":len(retained), "invalid_numeric_units":0}
        assert all(published["sample_size"][k] == v for k,v in counts.items())
        assert published["sample_size"]["included_by_seed"] == {
            str(k): v for k, v in Counter(r["seed"] for r in retained).items()}
        assert published["sample_size"]["floor_excluded_by_method"] == dict(
            Counter(r["method"] for r in floor))
        assert published["sample_size"]["secondary_invalid_numeric_units"] == 0
        units = {}
        for metric in METRICS:
            unit = summarize_units(retained, metric, draws, bootstrap_seed + position)
            reference = published["unit_level"][metric]
            differences = [abs(unit["rho"]-reference["rho"])] + [abs(a-b) for a,b in zip(unit["ci95"],reference["ci95"])]
            max_stat_discrepancy = max(max_stat_discrepancy, *differences)
            assert max(differences) < 1e-10, (setting,metric,differences)
            assert unit["bootstrap_valid"] == reference["bootstrap_valid"]
            units[metric] = unit
        method_rows, method_rhos = method_summary(retained, raw["retention_rows"])
        for row in method_rows:
            reference = next(r for r in published["method_level"]["rows"] if r["method"] == row["method"])
            for key,value in row.items():
                if key != "method":
                    assert abs(value-reference[key]) < 1e-12, (setting,key)
        for metric,value in method_rhos.items():
            assert abs(value-published["method_level"]["rho_drift_vs_retention"][metric]) < 1e-12
        findings[setting] = {**counts, "included_by_seed":dict(Counter(r["seed"] for r in retained)),
                             "unit_level":units, "method_rhos":method_rhos,
                             "method_rows":method_rows}
    primary = [v["unit_level"][METRICS[0]] for v in findings.values()]
    harm_settings = [s for s,v in findings.items() if v["unit_level"][METRICS[0]]["rho"] >= .2
                     and v["unit_level"][METRICS[0]]["ci95"][0] > 0]
    decision = ("harm_supported" if len(harm_settings) >= 2 else
                "harm_not_supported" if all(p["rho"] < .1 for p in primary) or
                all(p["ci95"][0] <= 0 <= p["ci95"][1] for p in primary) else "intermediate")
    assert decision == generated["verdict"]["decision"]
    assert harm_settings == generated["verdict"]["harm_settings"]
    assert hashes == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert all(hashlib.sha256((base / rel).read_bytes()).hexdigest() == digest
               for rel, digest in shard_hashes.items()), "A shard changed during validation"
    print(json.dumps({"source_hashes":hashes,"cells":120,"raw_rows":len(raw["rows"]),
                      "retention_rows":len(raw["retention_rows"]), "shards_verified":len(shard_hashes),
                      "json_csv_max_abs_difference":max(row_delta,ret_delta),
                      "aggregate_shards_max_abs_difference":max(shard_delta,shard_ret_delta),
                      "max_statistic_difference":max_stat_discrepancy,
                      "decision":decision,"harm_settings":harm_settings,"settings":findings,
                      "caveats":["Only five seed clusters; percentile CI stability is limited.",
                                 "S2 Malaya is pre-Section18-cleanup application classification, not attack-family NIDS.",
                                 "S3 KDDTest-21 is nested within KDDTest+, not an independent domain.",
                                 "Association is not causation and does not reverse CRR rejection."]},
                     ensure_ascii=False,indent=2))


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Validation must run without Python -O; assertions are required")
    with threadpool_limits(limits=1):
        main()
