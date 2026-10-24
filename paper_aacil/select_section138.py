"""Recompute section147 selection from the complete published screening grid only."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.section138_config import configurations
from ncd.analyze_stagewise_forgetting import stage_metrics, aggregate_trajectory


def main():
    source = ROOT / "paper_aacil/results/section138_screen"
    rows, hashes = [], {}
    expected = {f"{d}/{c.identifier}_seed{s}.json" for c in configurations()
                for d in ("unsw-nb15", "cic-ids-2017") for s in (1, 2)}
    assert {p.relative_to(source).as_posix() for p in source.rglob("*.json")} == expected
    for cfg in configurations():
        cells = []
        for dataset in ("UNSW-NB15", "CIC-IDS-2017"):
            for seed in (1, 2):
                path = source / dataset.lower() / f"{cfg.identifier}_seed{seed}.json"
                payload = path.read_bytes()
                hashes[path.relative_to(source).as_posix()] = hashlib.sha256(payload).hexdigest()
                r = json.loads(payload)
                i = r["identity"]
                assert r["completed"] and r["scientific_result"] and not i["smoke"]
                assert (i["dataset"], i["seed"], i["method"], i["output_role"]) == (dataset, seed, cfg.identifier, "screen")
                stages = []
                for cp in r["checkpoints"]:
                    metrics = stage_metrics(cp["operating_points"]["0.01"]["confusion_matrix"], cp["seen_classes"], r["class_order"])
                    stages.append(dict(task_index=cp["task_index"], metrics=metrics))
                trajectory = aggregate_trajectory(stages, r["tasks"], r["class_order"])
                last = r["checkpoints"][-1]
                default = stage_metrics(last["default_decision"]["confusion_matrix"], last["seen_classes"], r["class_order"])
                cells.append(dict(dataset=dataset, seed=seed, default_bacc=default["balanced_accuracy"],
                    fpr01_bacc=trajectory["final_balanced_accuracy"], forgetting=trajectory["forgetting_all_old_classes"]))
        rows.append(dict(configuration=cfg.identifier, reference=cfg.is_reference, cells=cells,
            score=statistics.mean(v for c in cells for v in (c["default_bacc"], c["fpr01_bacc"])),
            mean_forgetting=statistics.mean(c["forgetting"] for c in cells)))
    reference = next(r for r in rows if r["reference"])
    for row in rows:
        row["eligible"] = not row["reference"] and row["mean_forgetting"] <= reference["mean_forgetting"] + .05
    eligible = sorted((r for r in rows if r["eligible"]), key=lambda r: -r["score"])
    assert len(eligible) >= 3
    # Fail closed if a parameter-count tie-break is needed; never substitute lexical order.
    assert all(a["score"] != b["score"] for a, b in zip(eligible[:3], eligible[1:4])), "Parameter-count tie-break required"
    result = dict(protocol="Sections 138.2 and 147", scope="Exploratory screening; not superiority evidence",
        excluded="T1/T2 registered but not run (time), per section147 before scoring",
        units="Fractions; eligibility margin 0.05 equals five percentage points",
        forgetting="Signed mean best pre-final recall minus final recall over old classes including Normal",
        selected=[r["configuration"] for r in eligible[:3]], rows=sorted(rows, key=lambda r: -r["score"]),
        screening_sha256=hashes)
    output = ROOT / "paper_aacil/results/section138_selection.json"
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.exists():
        assert output.read_text() == text, "Existing selection differs"
    else:
        output.write_text(text)
    print(json.dumps(result["selected"]))


if __name__ == "__main__":
    main()
