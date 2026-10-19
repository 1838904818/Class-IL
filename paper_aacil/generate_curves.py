"""Post-hoc paper curves from final checkpoints; CPU only, never training.

Existing Section26 curve inference is reused. AA-CIL/ablations each need one
test G/F pass and one permanent-Normal-holdout G pass, cached for later plots.
Replay-BS needs one cached test-logit pass; its fixed tau sweep is post hoc.
NSL test-Normal calibration is deliberately labelled a non-deployable sensitivity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ncd.model import MLP
from ncd.analyze_farq_operating_curves import cm_metrics
from ncd.run_problem_scout import save_json

DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017")
METHODS = ("aacil", "ts_mem", "aa_noattmem")
ALPHAS = [0., .0001, .0005, .001, .002, .005, .01, .02, .03, .05, .075,
          .1, .15, .2, .3, .5, .75, .9, .95, .99, .999]
BS_TAUS = np.linspace(-1., 3., 41).tolist()
LABELS = {"replay": "Replay / LA", "wacb": "WACB", "freq": "Freq", "derpp": "DER++",
          "aacil": "AA-CIL", "ts_mem": "TS-mem", "aa_noattmem": "AA-noAttMem", "replay_bs": "Replay-BS"}
STYLE = {"replay": ("#0072B2", "--"), "wacb": ("#E69F00", "--"),
         "freq": ("#009E73", "--"), "derpp": ("#CC79A7", "--"),
         "aacil": ("#000000", "-"), "ts_mem": ("#56B4E9", ":"),
         "aa_noattmem": ("#D55E00", "-."), "replay_bs": ("#666666", ":")}


def read(p):
    return json.loads(p.read_bytes())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def slug(dataset):
    return dataset.lower().replace("-", "_")


def metrics(margin, family, threshold, y, names, tail=None):
    normal = names.index("Normal")
    pred = np.full(len(y), normal, np.int64) if tail == "normal" else family if tail == "attack" else np.where(margin > threshold, family, normal)
    n = len(names)
    return cm_metrics(np.bincount(y*n+pred, minlength=n*n).reshape(n, n), names)


def threshold(scores, alpha):
    limit = int(Fraction(str(alpha))*len(scores))
    return float(np.sort(scores)[max(0, len(scores)-limit-1)])


def scores_for(state, identity, data, batch):
    dim, hidden = data["x_test"].shape[1], int(identity["effective_config"]["hidden"])
    gate = MLP(dim, hidden, 2)
    family = MLP(dim, hidden, len(identity["class_order"])-1)
    gate.load_state_dict(state["gate_model_state"])
    family.load_state_dict(state["family_model_state"])
    gate.eval(); family.eval()
    margins, predictions, benign = [], [], []
    attack_ids = np.asarray(state["attack_class_ids"])
    with torch.no_grad():
        for start in range(0, len(data["x_test"]), batch):
            x = torch.from_numpy(np.array(data["x_test"][start:start+batch], copy=True))
            g, f = gate(x), family(x)
            margins.append((g[:, 1]-g[:, 0]).numpy())
            predictions.append(attack_ids[f.argmax(1).numpy()])
        ids = state["calibration_row_ids"].numpy()
        for start in range(0, len(ids), batch):
            g = gate(torch.from_numpy(np.array(data["x_train"][ids[start:start+batch]], copy=True)))
            benign.append((g[:, 1]-g[:, 0]).numpy())
    return np.concatenate(margins), np.concatenate(predictions), np.concatenate(benign)


def load_data(identity, data_cache, data_root=None):
    dataset = identity["dataset"]
    if dataset not in data_cache:
        base = (data_root or ROOT/identity["effective_config"]["problem_scout_data_root"])/dataset.lower()
        if not (base/"test_x.npy").exists() and data_root is None:
            # HPC absolute paths are execution provenance, not local mount paths.
            local_config = read(ROOT/"ncd/configs/problem_scout_section21.json")
            base = ROOT/local_config["problem_scout_data_root"]/dataset.lower()
        data_cache[dataset] = {k: np.load(base/f, mmap_mode="r") for k, f in
                              (("x_train", "train_x.npy"), ("x_test", "test_x.npy"), ("y_test", "test_y.npy"))}
    return data_cache[dataset]


def curve_for(path, output, score_root, data_cache, batch, data_root=None):
    source = read(path)
    ident, final = source["identity"], source["checkpoints"][-1]
    binding = {"source_result": path.relative_to(ROOT).as_posix(), "source_result_sha256": digest(path),
               "checkpoint_sha256": final["checkpoint_sha256"], "checkpoint_file": final["checkpoint_file"],
               "training_source_commit": ident["git_commit"], "protocol": "aa_normal_quantile_curves_v1",
               "calibration_targets": ALPHAS, "device": "cpu", "cpu_threads": 2}
    if output.exists():
        old = read(output)
        if old["binding"] != binding or not old.get("completed"):
            raise ValueError("Existing curve provenance differs; preserve it and choose a new output root")
        return old, False
    start = time.perf_counter()
    dataset, method, seed = ident["dataset"], ident["method"], ident["seed"]
    data = load_data(ident, data_cache, data_root)
    cached = score_root/slug(dataset)/f"{method}_seed{seed}.npz"
    forward = False
    if cached.exists():
        with np.load(cached, allow_pickle=False) as z:
            if str(z["source_result_sha256"]) != binding["source_result_sha256"] or str(z["checkpoint_sha256"]) != final["checkpoint_sha256"]:
                raise ValueError("Local score cache binding differs; do not overwrite")
            margin, family, benign = z["margin"], z["family"], z["benign"]
    else:
        state = torch.load(path.parent/final["checkpoint_file"], map_location="cpu", weights_only=False)
        margin, family, benign = scores_for(state, ident, data, batch)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cached, margin=margin, family=family, benign=benign,
                            source_result_sha256=binding["source_result_sha256"], checkpoint_sha256=final["checkpoint_sha256"])
        forward = True
    inference_elapsed = time.perf_counter()-start
    names, y = ident["class_order"], data["y_test"]
    normal = names.index("Normal")
    points = [{"calibration_target": None, "tail": "all_Normal", **metrics(margin, family, None, y, names, "normal")}]
    for alpha in ALPHAS:
        t = threshold(benign, alpha)
        points.append({"calibration_target": alpha, "threshold": t,
                       "calibration_fpr": float(np.mean(benign > t)), **metrics(margin, family, t, y, names)})
    points.append({"calibration_target": None, "tail": "all_attack", **metrics(margin, family, None, y, names, "attack")})
    sensitivity = []
    if dataset == "NSL-KDD":
        test_benign = margin[y == normal]
        for alpha in (.01, .02, .05):
            for role, pool in (("permanent_TRAIN_Normal", benign), ("TEST_Normal_post_hoc_non_deployable", test_benign)):
                t = threshold(pool, alpha)
                sensitivity.append({"alpha": alpha, "calibration_source": role, "calibration_rows": len(pool),
                                    "threshold": t, "calibration_fpr": float(np.mean(pool > t)),
                                    "test_labels_used_for_threshold": role.startswith("TEST"),
                                    **metrics(margin, family, t, y, names)})
    row = {"completed": True, "completed_at": datetime.now().astimezone().isoformat(), "post_hoc": True,
           "dataset": dataset, "method": method, "seed": seed, "binding": binding,
           "points": points, "nsl_calibration_sensitivity": sensitivity,
           "score_cache_local_only": True, "cost_seconds": {"load_and_forward_or_score_cache": inference_elapsed,
               "cell_total": time.perf_counter()-start}, "new_forward_performed": forward,
           "limitations": ["Descriptive test curves, not a replacement for Section27.8 registered tests.",
               "NSL TEST-Normal calibration uses test labels and is non-deployable; it is only a post-hoc shift sensitivity.",
               "CPU-derived curve scores may differ numerically from original GPU evaluation; registered result JSON remains unchanged."]}
    save_json(output, row)
    return row, True


def bs_curve_for(path, output, score_root, data_cache, batch, data_root=None):
    """One CPU forward, fixed 41-point sweep, and separate calibrated markers."""
    source = read(path)
    ident, final = source["identity"], source["checkpoints"][-1]
    calibrated = final["normal_holdout_calibration"]
    names, seen = ident["class_order"], final["seen_classes"]
    if sorted(seen) != list(range(len(names))):
        raise ValueError("Replay-BS curves require the completed final all-class checkpoint")
    counts = np.asarray(calibrated["original_train_counts"], dtype=np.float64)
    if np.any(counts <= 0) or len(counts) != len(names):
        raise ValueError("Replay-BS original TRAIN prior is incomplete")
    log_prior = np.log(counts/counts.sum()).astype(np.float32)[seen]
    selected_taus = {a: calibrated["operating_points"][a]["total_tau"] for a in ("0.01", "0.02", "0.05")}
    binding = {"source_result": path.relative_to(ROOT).as_posix(), "source_result_sha256": digest(path),
               "checkpoint_sha256": final["checkpoint_sha256"], "checkpoint_file": final["checkpoint_file"],
               "training_source_commit": ident["git_commit"], "protocol": "replay_bs_fixed_tau_curves_v1",
               "tau_grid": BS_TAUS, "selected_normal_only_taus": selected_taus,
               "original_train_counts": counts.astype(np.int64).tolist(), "seen_class_columns": seen,
               "device": "cpu", "cpu_threads": 2}
    if output.exists():
        old = read(output)
        if old["binding"] != binding or not old.get("completed"):
            raise ValueError("Existing Replay-BS curve differs; preserve it and choose a new output root")
        return old, False
    start = time.perf_counter()
    dataset, seed = ident["dataset"], ident["seed"]
    data = load_data(ident, data_cache, data_root)
    cached = score_root/slug(dataset)/f"replay_bs_seed{seed}.npz"
    forward = False
    if cached.exists():
        with np.load(cached, allow_pickle=False) as z:
            if (str(z["source_result_sha256"]) != binding["source_result_sha256"] or
                    str(z["checkpoint_sha256"]) != binding["checkpoint_sha256"]):
                raise ValueError("Replay-BS local logits cache differs; do not overwrite")
            logits = z["logits"]
    else:
        state = torch.load(path.parent/final["checkpoint_file"], map_location="cpu", weights_only=False)
        model = MLP(data["x_test"].shape[1], int(ident["effective_config"]["hidden"]), len(names))
        model.load_state_dict(state["model_state"])
        model.eval()
        parts = []
        with torch.no_grad():
            for begin in range(0, len(data["x_test"]), batch):
                x = torch.from_numpy(np.array(data["x_test"][begin:begin+batch], copy=True))
                parts.append(model(x)[:, seen].numpy())
        logits = np.concatenate(parts)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cached, logits=logits, source_result_sha256=binding["source_result_sha256"],
                            checkpoint_sha256=binding["checkpoint_sha256"])
        forward = True
    if logits.shape != (len(data["y_test"]), len(seen)) or not np.isfinite(logits).all():
        raise ValueError("Invalid Replay-BS cached logits shape or values")
    inference_elapsed = time.perf_counter()-start
    values, prior = torch.from_numpy(logits), torch.from_numpy(log_prior)
    y, n = data["y_test"], len(names)

    def point(tau):
        pred = np.asarray(seen)[(values+float(tau)*prior).argmax(1).numpy()]
        cm = np.bincount(y*n+pred, minlength=n*n).reshape(n, n)
        return {"tau": float(tau), **cm_metrics(cm, names)}

    points = [point(tau) for tau in BS_TAUS]
    markers = [{"alpha": float(a), "selection_role": "permanent_TRAIN_Normal_only",
                "calibration_fpr": calibrated["operating_points"][a]["calibration_fpr"],
                "calibration_false_positives": calibrated["operating_points"][a]["calibration_false_positives"],
                "outside_fixed_tau_grid": not BS_TAUS[0] <= tau <= BS_TAUS[-1],
                **point(tau)} for a, tau in selected_taus.items()]
    fprs = [p["normal_fpr"] for p in points]
    row = {"completed": True, "completed_at": datetime.now().astimezone().isoformat(), "post_hoc": True,
           "dataset": dataset, "method": "replay_bs", "seed": seed, "binding": binding,
           "points": points, "calibrated_points": markers,
           "fixed_tau_sweep_test_fpr_range": [min(fprs), max(fprs)],
           "score_cache_local_only": True, "new_forward_performed": forward,
           "cost_seconds": {"load_and_forward_or_score_cache": inference_elapsed, "cell_total": time.perf_counter()-start},
           "limitations": ["Post-hoc Replay-BS fixed 41-point tau sweep; it does not modify Section26.10 grids or registered decisions.",
               "Calibrated markers use existing Normal-only selected total tau; markers outside the sweep are not joined to the curve.",
               "Missing low-FPR coverage remains a gap: no extrapolation, tau-range extension, or test-guided point selection.",
               "CPU-derived scores may differ numerically from original GPU evaluation; original result JSON remains unchanged."]}
    save_json(output, row)
    return row, True


def render(rows, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.7))
        for method, (color, style) in STYLE.items():
            group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if not group:
                continue
            arr = np.asarray([[[p["normal_fpr"], p["attack_detection"], p["macro_f1"]] for p in r["points"]] for r in group])*100
            mean = arr.mean(0)
            sd = arr.std(0, ddof=1) if len(group)>1 else np.zeros_like(mean)
            for ax, yi in zip(axes, (1, 1, 2)):
                ax.plot(mean[:, 0], mean[:, yi], color=color, linestyle=style, linewidth=1.4,
                        label=f"{LABELS[method]} (n={len(group)})")
                if method == "replay_bs":
                    markers = np.asarray([[[p["normal_fpr"], p["attack_detection"], p["macro_f1"]]
                                           for p in r["calibrated_points"]] for r in group])*100
                    marker_mean = markers.mean(0)
                    ax.scatter(marker_mean[:, 0], marker_mean[:, yi], facecolors="none", edgecolors=color,
                               s=20, linewidths=1, zorder=4)
                if method == "aacil" and len(group)>1:
                    ax.fill_between(mean[:, 0], np.clip(mean[:, yi]-sd[:, yi], 0, 100),
                                    np.clip(mean[:, yi]+sd[:, yi], 0, 100), color=color, alpha=.08)
        for ax, title in zip(axes, ("Full range", "Low-FPR range", "Low-FPR Macro-F1")):
            ax.set(title=title, xlabel="Test benign FPR (%)", ylim=(0, 100))
            ax.grid(True, color="#dddddd", linewidth=.5)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set(xlim=(0, 100), ylabel="Attack detection (%)")
        axes[1].set(xlim=(0, 10), ylabel="Attack detection (%)")
        axes[2].set(xlim=(0, 10), ylabel="Macro-F1 (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.suptitle(f"{dataset}: descriptive operating curves (post hoc)", y=.99)
        fig.tight_layout(rect=(0, .16, 1, .94))
        fig.savefig(figure_dir/f"{slug(dataset)}_operating_curves.pdf", bbox_inches="tight")
        fig.savefig(figure_dir/f"{slug(dataset)}_operating_curves.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aa-root", type=Path, action="append")
    p.add_argument("--bs-root", type=Path, action="append")
    p.add_argument("--fallback-root", type=Path, action="append", default=[], help="Additional AA/Replay-BS completed-cell root")
    p.add_argument("--data-root", type=Path, help="Local prepared data root for downloaded HPC checkpoints")
    p.add_argument("--output-dir", type=Path, default=ROOT/"ncd/results/problem_scout/section28/paper_curves_v1")
    p.add_argument("--score-cache-root", type=Path, default=ROOT.parent/"exp_h_section21_data/section28_curve_scores_local_only")
    p.add_argument("--figure-dir", type=Path, default=ROOT/"paper_aacil/figures")
    p.add_argument("--batch-size", type=int, default=2048)
    args = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    start = time.perf_counter(); started_at = datetime.now().astimezone().isoformat()
    rows, data_cache, new, reused, new_bs, reused_bs = [], {}, 0, 0, 0, 0
    prior_summary = read(args.output_dir/"summary.json") if (args.output_dir/"summary.json").exists() else {}
    common_roots = [ROOT/"ncd/results/problem_scout/section28/cic_local_queue_v1",
                    ROOT/"ncd/results/problem_scout/section28/hpc_cic_section28"]
    roots = (args.aa_root or [ROOT/"ncd/results/problem_scout/section27/local_v1",
                             ROOT/"ncd/results/problem_scout/section28/aacil_cic_local_v1", *common_roots])+args.fallback_root
    chosen = {}
    for root in roots:
        for path in root.resolve().rglob("result.json"):
            r = read(path)
            if r.get("completed") and r.get("identity", {}).get("method") in METHODS:
                key = tuple(r["identity"][k] for k in ("dataset", "method", "seed"))
                completed = datetime.fromisoformat(r["completed_at"])
                if key not in chosen or completed < chosen[key][0]:
                    chosen[key] = (completed, path)
    for key, (_, path) in sorted(chosen.items()):
        output = args.output_dir/slug(key[0])/f"{key[1]}_seed{key[2]}"/"result.json"
        row, created = curve_for(path, output, args.score_cache_root, data_cache, args.batch_size, args.data_root)
        rows.append(row); new += int(created); reused += int(not created)
        print(f"CURVE {'NEW' if created else 'REUSED'} {key}", flush=True)
    baseline = ROOT/"ncd/results/problem_scout/section26/operating_curves_v1"
    for path in baseline.glob("*/*/result.json"):
        r = read(path)
        if r.get("completed") and r.get("method") in ("replay", "wacb", "freq", "derpp"):
            rows.append({k: r[k] for k in ("dataset", "method", "seed", "points")})
    bs_chosen = {}
    for root in (args.bs_root or [ROOT/"ncd/results/problem_scout/section28/replay_bs_local_v1", *common_roots])+args.fallback_root:
        for path in root.resolve().rglob("result.json"):
            r = read(path)
            if r.get("completed") and r.get("identity", {}).get("method") == "replay_bs":
                key = tuple(r["identity"][k] for k in ("dataset", "method", "seed"))
                completed = datetime.fromisoformat(r["completed_at"])
                if key not in bs_chosen or completed < bs_chosen[key][0]:
                    bs_chosen[key] = (completed, path)
    for key, (_, path) in sorted(bs_chosen.items()):
        output = args.output_dir/slug(key[0])/f"replay_bs_seed{key[2]}"/"result.json"
        row, created = bs_curve_for(path, output, args.score_cache_root, data_cache, args.batch_size, args.data_root)
        rows.append(row); new_bs += int(created); reused_bs += int(not created)
        print(f"BS_CURVE {'NEW' if created else 'REUSED'} {key}", flush=True)
    render(rows, args.figure_dir)
    ended_at = datetime.now().astimezone().isoformat()
    elapsed = time.perf_counter()-start
    history = prior_summary.get("execution_attempts", [])
    if prior_summary and not history:
        history = [{"completed_at": prior_summary["completed_at"], "seconds": prior_summary["run_seconds"],
                    "new_AA_curve_cells": prior_summary["new_AA_curve_cells"], "reused_AA_curve_cells": prior_summary["reused_AA_curve_cells"]}]
    history.append({"started_at": started_at, "completed_at": ended_at, "seconds": elapsed,
                    "new_AA_curve_cells": new, "reused_AA_curve_cells": reused,
                    "new_BS_curve_cells": new_bs, "reused_BS_curve_cells": reused_bs})
    summary = {"post_hoc": True, "completed_at": ended_at, "execution_attempts": history,
               "new_AA_curve_cells": new, "reused_AA_curve_cells": reused, "run_seconds": elapsed,
               "new_BS_curve_cells": new_bs, "reused_BS_curve_cells": reused_bs,
               "plot_rule": "Seed means at common Normal calibration quantiles for AA variants and common fixed tau for cached baselines and Replay-BS. Replay-BS open circles are separate Normal-only calibrated points (1/2/5% targets), never joined to bridge gaps outside its fixed [-1,3] sweep. No extrapolation or mean-curve interpolation used for registered tests.",
               "uncertainty": "AA-CIL shading is +/- one sample SD across available seeds, not a confidence interval.",
               "nsl_calibration_sensitivity": [{"method": r["method"], "seed": r["seed"], "points": r["nsl_calibration_sensitivity"]} for r in rows if r.get("nsl_calibration_sensitivity")],
               "groups": [{"dataset": d, "method": m, "n": sum(r["dataset"]==d and r["method"]==m for r in rows),
                           "seeds": sorted(r["seed"] for r in rows if r["dataset"]==d and r["method"]==m)} for d in DATASETS for m in STYLE]}
    save_json(args.output_dir/"summary.json", summary)
    print(f"CURVES_COMPLETE AAnew={new} AAreused={reused} BSnew={new_bs} BSreused={reused_bs} seconds={summary['run_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
