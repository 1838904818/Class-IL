"""Registered CIC fixed200 S3 only; consumes published cells, never trains."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_rel


def main():
    repo = Path(__file__).resolve().parents[1]
    base = repo / "ncd/results/problem_scout/section81/publication_v1/cic_ids_2017/fixed200"
    rows = []
    sources = []
    for seed in (1, 2, 3, 4, 42):
        pair = {}
        identities = []
        for method in ("ar_tabt", "er_bal_tabt"):
            path = base / f"{method}_seed{seed}" / "result.json"
            raw = path.read_bytes()
            result = json.loads(raw)
            identity = result["identity"]
            execution = result["section81_execution"]
            assert result["completed"] and identity["seed"] == seed
            assert identity["dataset"] == "CIC-IDS-2017"
            assert identity["effective_config"]["memory_setting"] == "fixed200"
            assert not identity["effective_config"]["smoke"]
            assert execution["job_exit"]["exit_code"] == 0
            assert len(result["checkpoints"]) == 4
            point = result["checkpoints"][-1]["operating_points"]["0.01"]
            value = float(point["balanced_accuracy"])
            cm = np.asarray(point["confusion_matrix"], dtype=float)
            independently_derived = np.mean(np.diag(cm) / cm.sum(axis=1))
            assert np.isclose(value, independently_derived, rtol=0, atol=1e-12)
            assert np.isfinite(value) and 0 <= value <= 1
            pair[method] = value * 100
            identities.append((identity["data_binding"], identity["tasks"], identity["class_order"]))
            sources.append({"path": path.relative_to(repo).as_posix(),
                            "sha256": hashlib.sha256(raw).hexdigest()})
        assert identities[0] == identities[1]
        rows.append({"seed": seed, **pair,
                     "difference_pp": pair["ar_tabt"] - pair["er_bal_tabt"]})
    ar = np.array([r["ar_tabt"] for r in rows])
    er = np.array([r["er_bal_tabt"] for r in rows])
    differences = ar - er
    test = ttest_rel(ar, er, alternative="greater")
    manual_t = differences.mean() / (differences.std(ddof=1) / np.sqrt(5))
    assert np.isclose(test.statistic, manual_t)
    assert np.isclose(test.pvalue, t.sf(manual_t, df=4))
    flips = [np.mean(differences * signs)
             for signs in itertools.product((-1, 1), repeat=5)]
    output = {
        "scope": "Section 81 B1 CIC-IDS-2017 fixed200 alpha=0.01 S3 only",
        "hypothesis": "AR-TabT final BAcc > ER-Bal-TabT",
        "units": "percent for BAcc; percentage points for paired differences",
        "per_seed": rows,
        "ar_mean": float(ar.mean()), "er_mean": float(er.mean()),
        "mean_difference_pp": float(differences.mean()),
        "paired_t": {"alternative": "greater", "df": 4,
                     "statistic": float(test.statistic), "p": float(test.pvalue)},
        "exact_sign_flip": {"assignments": 32, "alternative": "greater",
                            "p": sum(x >= differences.mean() - 1e-12 for x in flips) / 32},
        "decision": "PASS" if test.pvalue < 0.05 else "FAIL",
        "sources": sources,
        "verification": "Each saved BAcc agrees with its saved confusion matrix; paired data bindings, class order and tasks match.",
        "limitations": ["Five paired seeds; no superiority is established by failure to reject.",
                         "S1/S2 are already reported by the lead in Section 113 and are not recomputed.",
                         "No alpha=0.02 test, training, inference, recalibration or cell re-export.",
                         "Bounded RQ2 comparison; RQ1, RQ3 and substantive innovation remain unresolved."],
        "reproduce": "python -B paper_aacil/compare_section81_cic_s3.py",
    }
    target = repo / "paper_aacil/results/section81_cic_fixed200_s3.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in output.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
