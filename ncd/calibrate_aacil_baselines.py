"""Section27.9 Normal-only tau calibration with authorized bracket extension.

Uses saved Section26.9 weights and completed, immutable Section26.10 curves.
No training, GPU, test-threshold selection, or rewrite of the fixed test curve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from ncd import analyze_farq_operating_curves as old

DATASETS, SEEDS = old.DATASETS, old.SEEDS
METHODS = ("replay", "wacb", "freq", "derpp", "farq", "joint")
ALPHAS = (.01, .02, .05)
CALIBRATION_PROTOCOL = {
    "section": "27.9", "version": 2, "targets": list(ALPHAS),
    "initial_tau_range": [-1., 3.],
    "tau_range_policy": "normal_holdout_only_bidirectional_bracket_expansion",
    "initial_expansion_step": 4., "expansion_multiplier": 2.,
    "maximum_expansion_steps": 32, "bisection_iterations": 50,
    "pool": "Same seed-specific permanent Normal TRAIN holdout for every method; never optimized",
    "rule": "First adjacent initial-grid bracket when present; otherwise expand toward the missing infeasible lower or feasible upper endpoint; retain feasible upper endpoint after 50 bisections",
    "finite_sample_limit": "FP <= floor(alpha * calibration_count); strict Normal prior maximum and nonincreasing observed FP required",
    "test_role": "Evaluate selected total tau only; test metrics never select tau or the expansion range",
    "replay_la": "Same Replay weights and total tau; baseline tau=1 plus delta, no extra training or inference",
    "section26_10_unchanged": True,
    "limitation": "The permanent pool was available to allocation validation; it is not a fresh independent calibration sample",
}
FROZEN_UTILITY_SHA = "93bc5b919860c2784ea0ee9363769b2f8c9e0f975906d3bd323a629988786fbf"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def shared_read(path):
    if os.name != "nt":
        return old.read(path)
    import ctypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                       ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
    create.restype = ctypes.c_void_p
    handle = create(str(path.resolve()), 0x80000000, 7, None, 3, 128, None)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), "Cannot read JSON", str(path))
    with os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY), "r", encoding="utf-8") as stream:
        return json.load(stream)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


def inspect_input(path, expected, source_hashes, config_hash):
    curve = shared_read(path)
    if not curve.get("completed"):
        return None
    require(tuple(curve[k] for k in ("dataset", "method", "seed")) == expected
            and curve["verification"]["all_requested_checks_passed"]
            and curve["protocol"] == old.PROTOCOL, "Input is not a completed fixed Section26.10 curve")
    b = curve["binding"]
    source = (ROOT / b["source_result"]).resolve()
    require(source.exists() and old.digest(source) == b["source_result_sha256"], "Exact original result is unavailable")
    cell = old.verify_source(source, expected, source_hashes, config_hash)
    require(cell is not None, "Original training source is incomplete")
    ident, final = cell["identity"], cell["final"]
    require(b["identity_sha256"] == old.canonical(ident)
            and b["checkpoint_sha256"] == final["checkpoint_sha256"]
            and b["checkpoint_file"] == final["checkpoint_file"]
            and b["source_completed_at"] == cell["result"]["completed_at"]
            and b["training_source_commit"] == ident["git_commit"]
            and b["source_hashes"] == ident["source_hashes"]
            and b["config_sha256"] == ident["config_sha256"]
            and b["data_binding"] == ident["data_binding"]
            and b["protocol_sha256"] == old.canonical(old.PROTOCOL), "Input curve provenance differs")
    old.verify_curve(curve, final, ident["class_order"])
    if "normal_holdout_calibration" in curve:
        old.verify_calibration(curve, ident)
    completed = datetime.fromisoformat(curve["completed_at"].replace("Z", "+00:00"))
    require(completed.tzinfo is not None, "Input curve completion has no timezone")
    return cell | {"input_curve": curve, "input_curve_path": path,
                   "input_curve_sha256": old.digest(path), "curve_completed": completed}


def normal_pool(identity, data):
    old.verify_holdout(identity, data["y_train"])
    normal = identity["class_order"].index("Normal")
    rows = np.flatnonzero(data["y_train"] == normal)
    count = max(5, math.ceil(.1 * len(rows)))
    indices = np.random.default_rng(identity["seed"] * 100003 + normal * 1009 + 17).permutation(rows)[:count]
    sha = hashlib.sha256(indices.tobytes()).hexdigest()
    require(sha == identity["section26_holdout"][str(normal)]["validation_indices_sha256"], "Calibration pool changed")
    features = count * data["x_train"].shape[1] * data["x_train"].dtype.itemsize
    labels = count * data["y_train"].dtype.itemsize
    pool = {"count": count, "indices_sha256": sha, "feature_bytes": features,
            "label_bytes": labels, "index_bytes": indices.nbytes,
            "feature_and_label_storage_bytes": features + labels,
            "outside_attack_replay_memory_budget": True}
    return normal, indices, pool


def select_taus(logits, prior, normal):
    """No test arrays or test metrics enter this function."""
    attack = [c for c in range(len(prior)) if c != normal]
    gaps = prior[normal] - prior[attack]
    require(bool(torch.all(gaps > 0)), "Normal float32 log-prior must be strictly largest for reachable monotone calibration")
    count, probes = len(logits), {}

    def fp(tau):
        tau = float(tau)
        require(math.isfinite(tau), "Nonfinite calibration tau")
        if tau not in probes:
            shifted = logits + tau * prior
            require(bool(torch.isfinite(shifted).all()), "Nonfinite shifted calibration logits")
            probes[tau] = int((shifted.argmax(1) != normal).sum())
            ordered = sorted(probes.items())
            require(all(a[1] >= b[1] for a, b in zip(ordered, ordered[1:])), "Observed Normal FP violates tau monotonicity")
        return probes[tau]

    grid = [{"tau": t, "false_positives": fp(t), "normal_count": count, "normal_fpr": fp(t)/count} for t in old.TAUS]
    selected = {}
    for alpha in ALPHAS:
        limit = int(Fraction(str(alpha))*count)
        expansion, direction, initial = [], "none", None
        feasible = [i for i, p in enumerate(grid) if p["false_positives"] <= limit]
        if feasible and feasible[0] > 0:
            j = feasible[0]
            lower, upper = old.TAUS[j-1], old.TAUS[j]
            initial = [lower, upper]
        else:
            step = CALIBRATION_PROTOCOL["initial_expansion_step"]
            if feasible:
                direction, upper = "negative", old.TAUS[0]
                lower = upper
                for _ in range(CALIBRATION_PROTOCOL["maximum_expansion_steps"]):
                    lower -= step
                    value = fp(lower)
                    expansion.append({"tau": lower, "false_positives": value})
                    if value > limit:
                        break
                    upper, step = lower, step*2.
            else:
                direction, lower = "positive", old.TAUS[-1]
                upper = lower
                for _ in range(CALIBRATION_PROTOCOL["maximum_expansion_steps"]):
                    upper += step
                    value = fp(upper)
                    expansion.append({"tau": upper, "false_positives": value})
                    if value <= limit:
                        break
                    lower, step = upper, step*2.
        require(fp(lower) > limit >= fp(upper), "Could not bracket target within fixed numerical expansion safety bound")
        bracket = [lower, upper]
        for _ in range(CALIBRATION_PROTOCOL["bisection_iterations"]):
            middle = (lower+upper)/2.
            if fp(middle) <= limit:
                upper = middle
            else:
                lower = middle
        require(fp(lower) > limit >= fp(upper), "Final bracket lost feasibility separation")
        selected[str(alpha)] = {"target_fpr": alpha, "maximum_false_positives": limit,
            "status": "available", "total_tau": upper, "starting_interval": [-1.,3.],
            "initial_grid_bracket": initial, "expansion_direction": direction, "expansion_steps": expansion,
            "bracket_before_bisection": bracket, "final_bisection_interval": [lower,upper],
            "bisection_iterations": 50, "calibration_false_positives": fp(upper),
            "calibration_count": count, "calibration_fpr": fp(upper)/count,
            "calibration_lower_false_positives": fp(lower)}
    evaluations = [{"tau":t,"false_positives":v,"normal_count":count,"normal_fpr":v/count} for t,v in sorted(probes.items())]
    return grid, selected, evaluations, float(gaps.min())


def test_metrics(logits, prior, tau, y, names):
    scores = logits + tau*prior
    require(bool(torch.isfinite(scores).all()), "Nonfinite selected test scores")
    pred = scores.argmax(1).numpy()
    n = len(names)
    cm = np.bincount(y*n+pred, minlength=n*n).reshape(n,n)
    return old.cm_metrics(cm,names)


def validate_completed(row, identity, final):
    require(row.get("completed") and row["verification"]["all_requested_checks_passed"], "Incomplete cached calibration")
    require(all(row[k]==identity[k] for k in ("dataset","method","seed","class_order")), "Cached cell identity differs")
    cal = row["normal_holdout_calibration"]
    require(cal["protocol"] == CALIBRATION_PROTOCOL, "Cached calibration protocol changed")
    normal, names = identity["class_order"].index("Normal"), identity["class_order"]
    expected = identity["section26_holdout"][str(normal)]
    pool, grid, queries = cal["pool"], cal["grid"], cal["search_evaluations"]
    n = pool["count"]
    features = identity["data_binding"]["x_train"]["shape"][1]
    require(n == expected["validation_count"] and pool["indices_sha256"] == expected["validation_indices_sha256"]
            and [r["tau"] for r in grid] == old.TAUS and cal["monotonicity"]["minimum_normal_log_prior_gap"] > 0,
            "Cached calibration pool/grid/monotonicity differs")
    require(pool["feature_bytes"]==n*features*4 and pool["label_bytes"]==pool["index_bytes"]==n*8
            and pool["feature_and_label_storage_bytes"]==n*(features*4+8)
            and pool["outside_attack_replay_memory_budget"], "Cached calibration storage differs")
    lookup = {r["tau"]:r["false_positives"] for r in queries}
    require(len(lookup)==len(queries) and list(lookup)==sorted(lookup)
            and all(math.isfinite(t) and isinstance(v,int) and 0<=v<=n for t,v in lookup.items())
            and all(a>=b for a,b in zip(lookup.values(),list(lookup.values())[1:])), "Cached Normal query trace is invalid")
    require(all(p["normal_count"]==n and p["normal_fpr"]==p["false_positives"]/n
                and lookup[p["tau"]]==p["false_positives"] for p in queries+grid), "Cached Normal query arithmetic differs")
    require(set(cal["operating_points"])=={str(a) for a in ALPHAS}, "Cached alpha targets differ")
    for alpha in ALPHAS:
        op, limit = cal["operating_points"][str(alpha)], int(Fraction(str(alpha))*n)
        lo, hi = op["final_bisection_interval"]
        blo,bhi = op["bracket_before_bisection"]
        require(op["status"]=="available" and op["target_fpr"]==alpha and op["maximum_false_positives"]==limit
                and blo<=lo<=hi==op["total_tau"]<=bhi and lookup[blo]>limit>=lookup[bhi]
                and lookup[lo]>limit>=lookup[hi] and op["calibration_false_positives"]==lookup[hi]
                and op["calibration_lower_false_positives"]==lookup[lo] and op["calibration_count"]==n
                and op["calibration_fpr"]==lookup[hi]/n and op["bisection_iterations"]==50,
                "Cached calibrated endpoint is not a conservative feasible bracket")
        # Replay the recorded Normal-only search; no forward pass or test score is needed.
        first = next((j for j,p in enumerate(grid) if p["false_positives"]<=limit),None)
        if first is not None and first>0:
            lower,upper = old.TAUS[first-1],old.TAUS[first]
            require(op["initial_grid_bracket"]==[lower,upper] and op["expansion_direction"]=="none"
                    and not op["expansion_steps"], "Cached first bracket changed")
        else:
            negative = first==0
            lower=upper=old.TAUS[0] if negative else old.TAUS[-1]
            step = 4.
            require(op["initial_grid_bracket"] is None and op["expansion_direction"]==("negative" if negative else "positive")
                    and 1<=len(op["expansion_steps"])<=32, "Cached expansion direction/count differs")
            for j,probe in enumerate(op["expansion_steps"]):
                if negative:
                    lower -= step
                    value = lookup[lower]
                    require(probe=={"tau":lower,"false_positives":value}, "Cached negative expansion differs")
                    done = value>limit
                    if not done:
                        upper=lower
                else:
                    upper += step
                    value = lookup[upper]
                    require(probe=={"tau":upper,"false_positives":value}, "Cached positive expansion differs")
                    done = value<=limit
                    if not done:
                        lower=upper
                require(done==(j==len(op["expansion_steps"])-1), "Cached expansion did not stop at first valid bracket")
                step *= 2.
        require(op["starting_interval"]==[-1.,3.] and [lower,upper]==op["bracket_before_bisection"], "Cached pre-bisection bracket differs")
        for _ in range(50):
            middle = (lower+upper)/2.
            if lookup[middle]<=limit:
                upper=middle
            else:
                lower=middle
        require([lower,upper]==op["final_bisection_interval"], "Cached bisection trace does not lead to selected tau")
        require(old.cm_metrics(op["test"]["confusion_matrix"],names)==op["test"], "Cached selected test CM arithmetic differs")
        require(op["test"]["support"]==np.asarray(final["id"]["confusion_matrix"]).sum(1).tolist(), "Cached test population differs")
    reproduction = row["reproduction_checks"]
    require(reproduction["tau0"]["confusion_matrix"]==final["id"]["confusion_matrix"], "Cached tau0 reproduction differs")
    if identity["method"]=="replay":
        require(reproduction["tau1"]["confusion_matrix"]==final["la_id"]["confusion_matrix"], "Cached Replay tau1 reproduction differs")
        alias = cal["replay_la_alias"]
        require(alias["same_checkpoint_and_predictions"] and alias["initial_tau"]==1.
                and alias["extra_training_seconds"]==alias["extra_inference_seconds"]==0.
                and alias["operating_points"]=={k:dict(v,delta_tau=v["total_tau"]-1.) for k,v in cal["operating_points"].items()},
                "Cached Replay+LA alias differs")


def run_cell(cell, data, data_seconds, binding, output):
    identity, final, curve = cell["identity"],cell["final"],cell["input_curve"]
    if output.exists():
        cached = shared_read(output)
        require(cached.get("binding")==binding, "Cached calibration provenance differs; preserve output and use a new directory")
        validate_completed(cached,identity,final)
        return cached,True
    started = time.perf_counter()
    row = {"schema_version":2,"completed":False,"binding":binding,
           **{k:identity[k] for k in ("dataset","method","seed","class_order")},
           "analysis_role":"section27_9_extended_normal_holdout_calibration",
           "excluded_from_section26_10_judgement":True,
           "existing_curve_cost_seconds":curve["cost_seconds"],
           "existing_normal_holdout_calibration_cost_seconds":curve.get("normal_holdout_calibration",{}).get("cost_seconds",{}),
           "cost_seconds":{"data_verification_charged":data_seconds},
           "verification":{"actual_final_payload_sha":True,"final_sidecar_and_canonical_identity":False,
               "source_config_and_all_four_prepared_array_hashes":True,"normal_holdout_identity":False,
               "normal_prior_strict_max":False,"normal_only_selection":True,"extended_calibration_verified":False,
               "tau0_exact_cm":False,"replay_tau1_exact_cm":None,"all_requested_checks_passed":False},
           "cost_note":"Old and new physical analysis attempts are retained separately. Component times are nested inside their respective elapsed totals; training is not repeated or charged again by this script.",
           "limitations":["Calibration alpha is not a guarantee of the same actual test FPR.",
                          "The existing permanent holdout was used for allocation validation; it is not a fresh independent pool.",
                          "Normal-prior shifts change multiclass decisions; this does not establish a new training mechanism or close RQ1/RQ3/innovation.",
                          "Only final checkpoint payloads are reread here; the frozen helper checks all trajectory metadata."]}
    try:
        state = torch.load(cell["payload"],map_location="cpu",weights_only=True)
        require(state["identity"]==identity and state["task_index"]==final["task_index"]
                and state["seen_classes"]==final["seen_classes"] and state["allocation"]==final["allocation"],
                "Actual checkpoint identity/task/allocation differs")
        normal, indices, pool = normal_pool(identity,data)
        prior = torch.from_numpy(data["log_prior"])
        attack = [c for c in range(len(prior)) if c!=normal]
        require(bool(torch.all(prior[normal]>prior[attack])), "Normal prior is not strictly largest")
        model = old.MLP(data["x_test"].shape[1],int(identity["effective_config"]["hidden"]),len(identity["class_order"]))
        model.load_state_dict(state["model_state"],strict=True)
        model.eval()
        del state
        row["cost_seconds"]["checkpoint_load_and_holdout"] = time.perf_counter()-started
        row["verification"].update(final_sidecar_and_canonical_identity=True,normal_holdout_identity=True,normal_prior_strict_max=True)
        batch = binding["parameters"]["batch_size"]
        tick = time.perf_counter()
        test_logits = old.model_logits(model,data["x_test"],batch)
        row["cost_seconds"]["test_logit_inference"] = time.perf_counter()-tick
        tick = time.perf_counter()
        normal_logits = old.model_logits(model,data["x_train"],batch,indices)
        normal_seconds = time.perf_counter()-tick
        row["cost_seconds"]["normal_holdout_logit_inference"] = normal_seconds
        tick = time.perf_counter()
        names, y = identity["class_order"],data["y_test"]
        reproduction = {"tau0":test_metrics(test_logits,prior,0.,y,names)}
        row["reproduction_checks"] = reproduction
        require(reproduction["tau0"]["confusion_matrix"]==final["id"]["confusion_matrix"], "tau0 does not exactly reproduce original test CM")
        row["verification"]["tau0_exact_cm"] = True
        if identity["method"]=="replay":
            reproduction["tau1"] = test_metrics(test_logits,prior,1.,y,names)
            require(reproduction["tau1"]["confusion_matrix"]==final["la_id"]["confusion_matrix"], "Replay tau1 does not exactly reproduce original LA CM")
            row["verification"]["replay_tau1_exact_cm"] = True
        row["cost_seconds"]["reproduction_check_seconds"] = time.perf_counter()-tick
        tick = time.perf_counter()
        grid,points,queries,min_gap = select_taus(normal_logits,prior,normal)
        search_seconds = time.perf_counter()-tick
        row["cost_seconds"]["normal_only_search_seconds"] = search_seconds
        if "normal_holdout_calibration" in curve:
            require(grid==curve["normal_holdout_calibration"]["grid"], "Initial Normal-only grid does not reproduce saved calibration counts")
        tick = time.perf_counter()
        for point in points.values():
            point["test"] = test_metrics(test_logits,prior,point["total_tau"],y,names)
        row["cost_seconds"]["selected_test_evaluation_seconds"] = time.perf_counter()-tick
        cal = {"protocol":CALIBRATION_PROTOCOL,"pool":pool,"grid":grid,"search_evaluations":queries,
               "operating_points":points,"monotonicity":{"normal_prior_strictly_largest":True,
                   "minimum_normal_log_prior_gap":min_gap,"observed_queries_nondecreasing_tau_nonincreasing_fp":True},
               "cost_seconds":{"normal_holdout_logit_inference":normal_seconds,
                   "normal_only_search_seconds":search_seconds,"calibration_total":normal_seconds+search_seconds},
               "numerical_rule":"Float32 logits/prior and first-index argmax ties; feasible upper endpoint, no randomized ties or test-selected tau"}
        if identity["method"]=="replay":
            cal["replay_la_alias"] = {"same_checkpoint_and_predictions":True,"initial_tau":1.,
                "extra_training_seconds":0.,"extra_inference_seconds":0.,
                "operating_points":{k:dict(v,delta_tau=v["total_tau"]-1.) for k,v in points.items()}}
        row["normal_holdout_calibration"] = cal
        row["verification"].update(extended_calibration_verified=True,all_requested_checks_passed=True)
        row.update(completed=True,completed_at=now())
        row["cost_seconds"]["cell_attempt_elapsed"] = time.perf_counter()-started
        validate_completed(row,identity,final)
        old.save(output,row)
        return row,False
    except BaseException as exc:
        row.update(completed=False,failed_at=now(),error_type=type(exc).__name__,
                   error="Calibration stopped; inspect preserved stderr and source-bound reproduction evidence")
        row["verification"]["all_requested_checks_passed"] = False
        row["cost_seconds"]["cell_attempt_elapsed"] = time.perf_counter()-started
        old.save(output.with_name("result.failed."+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".json"),row)
        raise


def summary(cells,missing,args):
    lookup = {(r["dataset"],r["method"],r["seed"]):r for r in cells}
    groups = []
    for dataset in args.datasets:
        for method in (*args.methods,"replay_la"):
            if method=="replay_la" and "replay" not in args.methods:
                continue
            rows = [lookup.get((dataset,"replay" if method=="replay_la" else method,s)) for s in SEEDS]
            if not any(rows):
                continue
            group = {"dataset":dataset,"method":method,"completed_seeds":[s for s,r in zip(SEEDS,rows) if r],
                     "role":"offline reference" if method=="joint" else "supplementary" if method=="farq" else "Section27.4 baseline",
                     "physical_alias":method=="replay_la","operating_points":{}}
            for alpha in ALPHAS:
                if not all(rows):
                    point = {"status":"pending","reason":"All five prescribed seeds required; no missing-seed substitution"}
                else:
                    ops = [r["normal_holdout_calibration"]["operating_points"][str(alpha)] for r in rows]
                    point = {"status":"available","selected_total_taus":[r["total_tau"] for r in ops],
                             "calibration_fpr":old.stats([r["calibration_fpr"] for r in ops]),
                             "test":{m:old.stats([r["test"][m] for r in ops]) for m in old.METRICS}}
                group["operating_points"][str(alpha)] = point
            groups.append(group)
    return {"schema_version":2,"updated_at":now(),"protocol":CALIBRATION_PROTOCOL,
            "completed_physical_cells":len(cells),"requested_cells":len(args.datasets)*len(args.methods)*len(args.seeds),
            "missing_cells":missing,"groups":groups,"section26_10_unchanged":True,
            "cell_evidence":[{k:r[k] for k in ("dataset","method","seed","completed_at","binding","cost_seconds",
                            "existing_curve_cost_seconds","existing_normal_holdout_calibration_cost_seconds")} for r in cells],
            "cost_note":"Per-cell elapsed excludes shared source/data validation; driver elapsed includes it. Old/new elapsed and nested components are not additive. Replay+LA has no extra physical inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",action="append",type=Path,dest="inputs",help="Completed immutable operating_curves roots")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=list(DATASETS))
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--seeds",nargs="+",type=int,choices=SEEDS,default=list(SEEDS))
    parser.add_argument("--device",choices=("cpu",),default="cpu")
    parser.add_argument("--cpu-threads",type=int,default=2)
    parser.add_argument("--batch-size",type=int)
    args = parser.parse_args()
    require(args.cpu_threads in (1,2) and (args.batch_size is None or args.batch_size>0), "Use one or two CPU threads and positive batch size")
    require(all(len(v)==len(set(v)) for v in (args.datasets,args.methods,args.seeds)), "Duplicate requested cell selection")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    utility = ROOT/"ncd/analyze_farq_operating_curves.py"
    require(old.digest(utility)==FROZEN_UTILITY_SHA, "Frozen Section26.10 utility bytes changed")
    inputs = list(dict.fromkeys(p.resolve() for p in (args.inputs or [ROOT/"ncd/results/problem_scout/section26/operating_curves_v1"])))
    output = args.output_dir.resolve()
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in inputs), "Use a new output root separate from immutable inputs")
    output.mkdir(parents=True,exist_ok=True)
    prior = shared_read(output/"summary.json") if (output/"summary.json").exists() else {}
    history = prior.get("execution",{}).get("attempts",[])
    started, clock = now(),time.perf_counter()
    attempt = {"started_at":started,"pid":os.getpid(),"new_inference_cells_attempted":0,
               "new_inference_cells_completed":0,"reused_cells":0,"resume_validation_seconds":0.,
               "arguments":{"entrypoint":"python -I ncd/calibrate_aacil_baselines.py",
                    "inputs":[old.label(p) for p in inputs],"output_dir":old.label(output),
                    "datasets":args.datasets,"methods":args.methods,"seeds":args.seeds,
                    "device":"cpu","cpu_threads":args.cpu_threads,"batch_size":args.batch_size}}
    cells,missing,cache,matched,verified_holdouts = [],[],{},{},set()
    def snapshot(status):
        attempt.update(status=status,elapsed_seconds=time.perf_counter()-clock)
        if status!="running":
            attempt["finished_at"] = now()
        result = summary(cells,missing,args)
        result["execution"] = {"attempts":history+[dict(attempt)]}
        old.save(output/"summary.json",result)
    with (output/"run.log").open("a",encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
        print(json.dumps({"event":"attempt_start",**attempt}),flush=True)
        with (output/"driver_attempts.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps({"event":"attempt_start",**attempt},sort_keys=True,allow_nan=False)+"\n")
        try:
            sources = {p:old.digest(ROOT/p) for p in old.SOURCES}
            config_hash,script_hash = old.digest(ROOT/"ncd/configs/problem_scout_section21.json"),old.digest(Path(__file__))
            for dataset in args.datasets:
                slug = dataset.lower().replace("-","_")
                for method in args.methods:
                    for seed in args.seeds:
                        expected = (dataset,method,seed)
                        attempt["current_cell"] = dict(zip(("dataset","method","seed"),expected))
                        target = output/slug/f"{method}_seed{seed}"/"result.json"
                        reused,check_start = target.exists(),time.perf_counter()
                        try:
                            paths = [p/slug/f"{method}_seed{seed}"/"result.json" for p in inputs]
                            candidates = [inspect_input(p,expected,sources,config_hash) for p in paths if p.exists()]
                            candidates = sorted((c for c in candidates if c),key=lambda c:(c["completed"],c["curve_completed"],str(c["path"])))
                            if not candidates:
                                missing.append(dict(zip(("dataset","method","seed"),expected)))
                                continue
                            cell,identity = candidates[0],candidates[0]["identity"]
                            match = {k:identity[k] for k in ("data_binding","data_metadata_sha256","class_order","tasks","section26_holdout")}
                            key = (dataset,seed)
                            require(key not in matched or matched[key]==match, "Compared methods do not share prepared data and permanent holdout")
                            matched[key] = match
                            data,data_seconds = old.load_data(identity,args.data_root,cache)
                            holdout_key = old.canonical([identity["data_binding"],identity["section26_holdout"],seed])
                            if holdout_key not in verified_holdouts:
                                old.verify_holdout(identity,data["y_train"])
                                verified_holdouts.add(holdout_key)
                            b = cell["input_curve"]["binding"]
                            require(b["original_train_counts"]==data["counts"].tolist()
                                    and b["prior"]==(data["counts"]/data["counts"].sum()).tolist()
                                    and b["log_prior_float32"]==data["log_prior"].tolist()
                                    and b["receipt_sha256"]==data["receipt_sha256"], "Input curve TRAIN prior/receipt differs")
                            binding = {k:b[k] for k in ("source_result","source_result_sha256","source_completed_at","checkpoint_file",
                                "checkpoint_sha256","identity_sha256","training_source_commit","source_hashes","config_sha256",
                                "data_binding","receipt_sha256","original_train_counts","prior","log_prior_float32")}
                            binding.update(input_curve_result=old.label(cell["input_curve_path"]),input_curve_sha256=cell["input_curve_sha256"],
                                input_curve_binding_sha256=old.canonical(b),calibration_protocol_sha256=old.canonical(CALIBRATION_PROTOCOL),
                                analysis_script_sha256=script_hash,utility_source_sha256=FROZEN_UTILITY_SHA,
                                parameters={"device":"cpu","cpu_threads":args.cpu_threads,
                                    "batch_size":args.batch_size or identity["effective_config"]["eval_batch_size"],
                                    "calibrate_normal_holdout":True,"torch":str(torch.__version__),"numpy":np.__version__})
                            if not reused:
                                attempt["new_inference_cells_attempted"] += 1
                            row,was_reused = run_cell(cell,data,data_seconds,binding,target)
                            attempt["reused_cells" if was_reused else "new_inference_cells_completed"] += 1
                            cells.append(row)
                        finally:
                            if reused:
                                attempt["resume_validation_seconds"] += time.perf_counter()-check_start
                        snapshot("running")
                        print(json.dumps({"event":"cell_complete",**attempt["current_cell"],"reused":was_reused}),flush=True)
            attempt.pop("current_cell",None)
            snapshot("finished")
        except BaseException as exc:
            attempt.update(error_type=type(exc).__name__,error="Stopped; completed results, failed cell evidence and full logs preserved")
            traceback.print_exc()
            snapshot("failed")
            raise
        finally:
            with (output/"driver_attempts.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps({"event":"attempt_end",**attempt},sort_keys=True,allow_nan=False)+"\n")
            print(json.dumps({"event":"attempt_end",**attempt}),flush=True)


if __name__=="__main__":
    main()
