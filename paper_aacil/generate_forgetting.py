"""Render saved stagewise summary as LaTeX; no inference or metric recomputation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "replay": "Replay", "replay_la": "Replay+LA", "derpp": "DER++",
    "wacb": "WACB", "freq": "Freq", "aacil": "AA-CIL", "ts_mem": "TS-mem",
    "aa_noattmem": "AA-noAttMem", "replay_bs": "Replay-BS (post hoc)",
    "aacil_der": "AA-DER (post hoc)", "aacil_kd": "AA-CIL+KD",
    "aacil_nogkd": "AA-CIL-noGKD", "aacil_iso": "AA-CIL-ISO",
    "farq": "FARQ (supp.)", "farq_all": "FARQ-all (supp.)",
    "joint": "Joint (offline)",
}
DATA = {"NSL-KDD": "NSL", "UNSW-NB15": "UNSW", "CIC-IDS-2017": "CIC17"}
ORDER = {method: i for i, method in enumerate(METHODS)}


def escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(text))


def value(stat):
    if stat.get("mean") is None:
        return r"\textit{undefined}"
    mean, sd = 100*stat["mean"], stat.get("sample_sd")
    return f"{mean:.2f}" + (rf"$\pm${100*sd:.2f}" if sd is not None else r"$\pm$--")


def table(caption, label, columns, header, rows, note):
    out = [r"\begin{table*}[t]", r"\centering\scriptsize",
           r"\setlength{\tabcolsep}{3.3pt}",
           r"\caption{"+caption+r"}\label{tab:"+label+"}",
           r"\begin{tabular}{"+columns+"}", r"\toprule", " & ".join(header)+r" \\", r"\midrule"]
    out.extend(" & ".join(row)+r" \\" for row in rows)
    out += [r"\bottomrule", r"\end{tabular}",
            r"\par\smallskip\begin{minipage}{0.98\textwidth}\scriptsize "+note+r"\end{minipage}",
            r"\end{table*}", ""]
    return "\n".join(out)


def render(summary):
    groups = sorted(summary["groups"], key=lambda g: (list(DATA).index(g["dataset"]),
                    ORDER.get(g["method"], 999), g["method"]))
    grouped = defaultdict(list)
    for group in groups:
        grouped[group["dataset"]].append(group)
    results, decomposition, perclass = [], [], []
    keys = ("average_incremental_accuracy", "final_balanced_accuracy", "forgetting_all_old_classes",
            "forgetting_old_attack_classes", "final_attack_detection", "final_macro_f1")
    for dataset, cells in grouped.items():
        slug = dataset.lower().replace("-", "_")
        rows = []
        for group in cells:
            rows.append([escape(METHODS.get(group["method"], group["method"])), str(group["n"])] +
                        [value(group["metrics"][key]) for key in keys])
        results.append(table(
            escape(DATA[dataset])+r": checkpoint-calibrated Class-IL performance at $\alpha=2\%$.",
            "forgetting-"+slug, "lrrrrrrr",
            ["Method", "$n$", "Mean inc. BAcc", "Final BAcc", "Forget (all old)",
             "Forget (old att.)", "Final ADR", "Final MF1"], rows,
            r"Mean$\pm$sample SD across the available seeds; accuracies are percentages and forgetting is in percentage points. "
            r"Mean incremental BAcc averages all seen-class stage BAcc values. Signed forgetting compares each old class's "
            r"best pre-final recall with its final recall; the primary scope includes Normal. "
            r"Smaller forgetting alone is not superiority: inspect final and incremental BAcc together. "
            r"Groups with $n<5$ are incomplete, descriptive only. Joint is offline; FARQ is supplementary; "
            r"Replay+LA shares Replay's calibrated total $\tau$ and incurs no separate inference. "
            r"These post-hoc stagewise summaries do not change the registered Section 27.8 decision."))
        aa = [g for g in cells if g["method"] == "aacil"]
        if aa:
            rows = []
            for old in aa[0]["old_class_forgetting"]:
                if old["class_name"] == "Normal":
                    continue
                rows.append([escape(old["class_name"]), str(old["forgetting"]["n"]),
                    value(old["forgetting"]), value(old["gate_normal_miss_increase"]),
                    value(old["other_attack_family_error_increase"])])
            decomposition.append(table(
                escape(DATA[dataset])+r": AA-CIL old-attack recall-loss decomposition at $\alpha=2\%$.",
                "forgetting-decomposition-"+slug, "lrrrr",
                ["Old attack", "$n$", "Recall loss (pp)", "$\Delta$ G miss (pp)", "$\Delta$ F wrong (pp)"],
                rows,
                r"For each seed and class, the reference is its earliest highest-recall stage before the final task. "
                r"Normal-miss and wrong-attack-family rates use all true-class rows as denominator; "
                r"their signed changes add to recall loss. Means add, while component SDs do not. "
                r"This observed confusion-matrix decomposition does not infer F behavior on G-rejected rows."))
        for group in cells:
            rows = []
            for c in group["final_per_class"]:
                rows.append([escape(c["class_name"]), str(c["support"]),
                    value(c["precision"]), str(c["precision"]["defined_n"]),
                    value(c["recall"]), str(c["recall"]["defined_n"]),
                    value(c["f1"]), str(c["f1"]["defined_n"])])
            perclass.append(table(
                escape(DATA[dataset]+" / "+METHODS.get(group["method"], group["method"]))+
                rf": final per-class results at $\alpha=2\%$ ($n={group['n']}$ seeds).",
                "forgetting-class-"+slug+"-"+group["method"].replace("_", "-"), "lrrrrrrr",
                ["Class", "Support", "Precision", "$n_P$", "Recall", "$n_R$", "F1", "$n_{F1}$"], rows,
                r"Percentages, mean$\pm$sample SD. $n_P,n_R,n_{F1}$ count defined seeds separately. "
                r"A class receiving no predictions has undefined precision, retained as null in the raw seed result; "
                r"only defined values enter that metric's aggregate. No defined values is shown as undefined, "
                r"and one defined value has no sample SD. Support is the per-seed test class count. "
                r"Incomplete groups remain descriptive."))
    header = "% Generated solely from section28/forgetting_v1/result.json; no inference.\n"
    return {"forgetting_results.tex": header+"\n".join(results),
            "forgetting_decomposition.tex": header+"\n".join(decomposition),
            "forgetting_per_class.tex": header+"\n".join(perclass)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"ncd/results/problem_scout/section28/forgetting_v1/result.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"paper_aacil/generated")
    args = parser.parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    if not summary.get("undefined_value_aggregation"):
        raise ValueError("Refresh the saved summary with Section29.7 defined_n aggregation first; no checkpoint inference is required for cached cells")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in render(summary).items():
        (args.output_dir/name).write_text(content, encoding="utf-8")
    macros = []
    for group in summary["groups"]:
        if group["method"] != "aacil" or not group["complete_five_seed"]:
            continue
        prefix = {"NSL-KDD": "NSL", "UNSW-NB15": "UNSW", "CIC-IDS-2017": "CIC"}[group["dataset"]]
        for key, suffix in (("average_incremental_accuracy", "AAIncBAcc"),
                            ("final_balanced_accuracy", "AAFinalBAcc"),
                            ("forgetting_all_old_classes", "AAForget")):
            stat = group["metrics"][key]
            macros.append("\\newcommand{\\"+prefix+suffix+"}{"+f"{100*stat['mean']:.2f}"+"}")
            macros.append("\\newcommand{\\"+prefix+suffix+"SD}{"+f"{100*stat['sample_sd']:.2f}"+"}")
    (args.output_dir/"forgetting_macros.tex").write_text("\n".join(macros)+"\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "groups": len(summary["groups"]),
        "files": ["forgetting_results.tex", "forgetting_decomposition.tex", "forgetting_per_class.tex", "forgetting_macros.tex"]}))


if __name__ == "__main__":
    main()
