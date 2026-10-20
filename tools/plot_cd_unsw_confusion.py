"""Render saved final alpha=2% confusion matrices; no inference or new tests."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "ncd/results/problem_scout"
ROOT = BASE / "section28/forgetting_v1/unsw_nb15"
SEEDS = [1, 2, 3, 4, 42]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def summed(method):
    total, names = None, None
    for s in SEEDS:
        d = read(ROOT / f"{method}_seed{s}/result.json")
        stage = d["stages"][-1]
        binding = stage["binding"]
        if not (d["completed"] and d["dataset"] == "UNSW-NB15"
                and d["method"] == method and d["seed"] == s
                and d["protocol"]["alpha"] == binding["protocol"]["alpha"] == .02
                and stage["completed"] and stage["task_index"] == len(d["tasks"]) - 1
                and binding["seen_classes"] == list(range(len(d["class_order"])))):
            raise ValueError(f"Not a complete final alpha=2% input: {method}/{s}")
        if method == "aacil_kd_wa":
            raw = read(BASE / "section28/publication_v1/unsw_nb15" /
                       f"{method}_seed{s}/result.json")
            cp = raw["checkpoints"][-1]
            point = cp["operating_points"]["0.02"]
            if not (raw["completed"] and raw["identity"]["method"] == method
                    and raw["identity"]["seed"] == s
                    and cp["checkpoint_sha256"] == binding["checkpoint_sha256_recorded"]
                    and raw["identity"]["git_commit"] == binding["training_source_commit"]
                    and canonical(point) == binding["compact_saved_alpha2_sha256"]
                    and stage["mode"] == "reused_saved_stage_alpha2_no_forward"
                    and stage["calibration"]["target_fpr"] == .02):
                raise ValueError(f"Saved compact alpha=2% binding differs: {s}")
            saved_cm = point["confusion_matrix"]
        else:
            final = read(BASE / "section27/baseline_calibration_v2/unsw_nb15" /
                         f"{method}_seed{s}/result.json")
            point = final["normal_holdout_calibration"]["operating_points"]["0.02"]
            if not (final["completed"] and final["method"] == method and final["seed"] == s
                    and final["verification"]["all_requested_checks_passed"]
                    and final["binding"]["source_result"] == binding["source_result"]
                    and final["binding"]["checkpoint_sha256"] == binding["checkpoint_sha256_recorded"]
                    and final["binding"]["training_source_commit"] == binding["training_source_commit"]
                    and point["status"] == "available" and point["target_fpr"] == .02
                    and stage["mode"] == "reused_valid_final_calibration_no_forward"):
                raise ValueError(f"Saved final calibration binding differs: {s}")
            saved_cm = point["test"]["confusion_matrix"]
        m = np.asarray(stage["metrics"]["confusion_matrix"])
        if not (np.array_equal(m, saved_cm) and m.shape == (10, 10)
                and np.all(m >= 0) and np.all(m == np.floor(m))
                and int(m.sum()) == 30248 and np.all(m.sum(axis=1) > 0)):
            raise ValueError(f"Invalid or mismatched saved confusion matrix: {method}/{s}")
        if names is not None and names != d["class_order"]:
            raise ValueError("Class order differs between seeds")
        total = m if total is None else total + m
        names = d["class_order"]
    return total, names


def main():
    # Sum counts before row normalization; repeated seeds are not independent rows.
    matrices = [summed(m) for m in ("freq", "aacil_kd_wa")]
    if not (matrices[0][1] == matrices[1][1]
            and np.array_equal(matrices[0][0].sum(axis=1), matrices[1][0].sum(axis=1))):
        raise ValueError("Compared populations differ")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.1), layout="constrained")
    for ax, (cm, names), title in zip(axes, matrices,
            ("Freq", "AA-CIL-KD+WA (fixed candidate)")):
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        heat = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(names)):
            for j in range(len(names)):
                v = pct[i, j]
                if v >= 1:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10,
                            color="white" if v > 55 else "#14213D")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(len(names)), names, fontsize=11)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title, fontsize=14)
    fig.colorbar(heat, ax=axes, shrink=.72, label="Row share (%)")
    fig.suptitle("UNSW-NB15 final task | calibration target 2% | five seeds pooled",
                 fontsize=14)
    target = REPO / "docs/figures/unsw_confusion_fpr2_freq_vs_aacil_kdwa.png"
    fig.savefig(target, dpi=200)
    plt.close(fig)
    print("Verified 10 saved final alpha=2% matrices; rendered only; no inference/tests.")


if __name__ == "__main__":
    main()
