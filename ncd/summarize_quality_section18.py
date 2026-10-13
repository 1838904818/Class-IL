"""Reconcile read-only Section 18 audits and render the registered decisions.

No training, source-data mutation, deduplication or sampling is performed.
Run from the repository root: python -I ncd/summarize_quality_section18.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ncd/results/data_quality/section18"
REPORT = ROOT / "ncd/DATA_QUALITY_AUDIT_20261013.md"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(rows):
    """Integer comparisons preserve the inclusive preregistered boundaries."""
    total = sum(r["test_rows"] for r in rows)
    overlap = sum(r["test_overlap_rows"] for r in rows)
    assert total > 0
    triggers = [r["class_name"] for r in rows
                if r["test_rows"] and r["test_overlap_rows"] * 5 >= r["test_rows"]]
    return ("去重重跑" if overlap * 20 >= total or triggers else "保持"), triggers


def row(dataset, view, name, pre_train, pre_test, train, test, dup, overlap,
        conflict, task, handling):
    values = (pre_train, pre_test, train, test, dup, overlap, conflict)
    assert all(isinstance(v, int) and v >= 0 for v in values), values
    assert dup <= train and overlap <= test and conflict <= train + test
    assert train <= pre_train and test <= pre_test
    return dict(dataset=dataset, view=view, class_name=name,
                train_before_cap=pre_train, test_before_cap=pre_test,
                train_rows=train, test_rows=test,
                train_within_class_excess_rows=dup,
                train_within_class_excess_rate=dup / train if train else None,
                test_overlap_rows=overlap, test_overlap_rate=overlap / test if test else None,
                union_label_conflict_rows=conflict, task_index=task, rare_class_handling=handling)


def describe(rows, name, view, global_dup, conflict, expected):
    result, triggers = decision(rows)
    assert result == expected, (name, view, result, expected)
    ntrain, ntest = sum(r["train_rows"] for r in rows), sum(r["test_rows"] for r in rows)
    overlap = sum(r["test_overlap_rows"] for r in rows)
    assert sum(r["union_label_conflict_rows"] for r in rows) == conflict
    max_row = max(rows, key=lambda r: r["test_overlap_rate"] or 0)
    return dict(dataset=name, view=view, train_rows=ntrain, test_rows=ntest,
                global_train_excess_rows=global_dup, global_train_excess_rate=global_dup / ntrain,
                test_overlap_rows=overlap, test_overlap_rate=overlap / ntest,
                max_class=max_row["class_name"], max_class_overlap_rate=max_row["test_overlap_rate"],
                trigger_classes=triggers, union_label_conflict_rows=conflict, decision=result)


def pct(x):
    return "—" if x is None else f"{100*x:.3f}%"


def write_csv(path, records):
    assert records
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build():
    tabular = load("tabular/audit_tabular.json")
    malaya = load("malaya/audit_result.json")
    cic = load("cic/cic_audit.json")
    assert len(cic["datasets"]) == 2, "CIC audit is still being written"
    assert "exp_h_s1" in cic, "CIC audit is not final"
    rows, summaries, uncapped, rare = [], [], [], []
    for d in tabular["datasets"]:
        rr = []
        for r in d["family_rows"]:
            rr.append(row(d["dataset"], "original", r["class_name"], r["train_before_cap"],
                          r["test_before_cap"], r["train_after_cap"], r["test_after_cap"],
                          r["train_within_class_excess_duplicate_rows"], r["test_matching_any_train_rows"],
                          r["model_family_combined_conflict_rows"], r["task_index"],
                          "保留模型类；原始细标签见 rare_labels.csv；不扩样"))
        t = d["totals"]
        assert sum(r["train_rows"] for r in rr) == t["train_rows"]
        assert sum(r["test_rows"] for r in rr) == t["test_rows"]
        summaries.append(describe(rr, d["dataset"], "original", t["train_excess_duplicate_rows"],
                                  t["conflicts"]["model_family"]["combined"]["affected_rows"], d["decision"]))
        rows.extend(rr)

    for d in cic["datasets"]:
        assert d["provenance"]["recomputed_split_mismatches"] == 0
        pre = {r["class_name"]: r for r in d["views"]["uncapped"]["rows"]}
        rr = []
        for r in d["views"]["ncd_capped"]["rows"]:
            if r["class_name"] == "__all__":
                total = r
                continue
            task = next(i for i, cs in enumerate(d["tasks"]) if r["class_name"] in cs)
            p = pre[r["class_name"]]
            rr.append(row(d["dataset"], "ncd_capped", r["class_name"], p["train_rows"], p["test_rows"],
                          r["train_rows"], r["test_rows"], r["train_duplicate_excess"],
                          r["test_matches_any_train_rows"], r["conflict_rows"], task,
                          "保留模型类；稀有原始标签及合并见下表；不扩样"))
        assert sum(r["train_rows"] for r in rr) == total["train_rows"]
        assert sum(r["test_rows"] for r in rr) == total["test_rows"]
        summaries.append(describe(rr, d["dataset"], "ncd_capped", total["train_duplicate_excess"],
                                  total["conflict_rows"], d["views"]["ncd_capped"]["decision"]))
        rows.extend(rr)
        uncapped.append(dict(dataset=d["dataset"], **pre["__all__"]))
        for r in d["raw_label_counts"]["rows"]:
            if r["below_100"] or r.get("small_partition") is True:
                evidence = r["rare_split_evidence"]
                assert evidence and r["mapped_family"]
                rare.append(dict(dataset=d["dataset"], source_label=r["raw_label"], raw_rows=r["rows"],
                                 model_class=r["mapped_family"], task_index=r["task_index"],
                                 uncapped_train=evidence["uncapped"]["train"],
                                 uncapped_test=evidence["uncapped"]["test"],
                                 used_train=evidence["ncd_capped"]["train"], used_test=evidence["ncd_capped"]["test"],
                                 handling=("保留原类；不扩样" if r["raw_label"] == r["mapped_family"]
                                           else "并入既有模型类；保留全部记录；不扩样")))

    class_info = {r["class_name"]: r for r in malaya["class_counts_and_handling"]}
    for key, view in [("original_cache_raw_train_vs_test", "original"),
                      ("exp_h_raw_fit_vs_id", "ExpH_ID"), ("exp_h_raw_fit_vs_shift", "ExpH_shift")]:
        v = malaya["views"][key]
        rr = []
        for r in v["rows"]:
            info = class_info[r["class_name"]]
            rr.append(row("Malaya", view, r["class_name"], r["train_rows"], r["test_rows"],
                          r["train_rows"], r["test_rows"], r["train_duplicate_excess_rows"],
                          r["test_rows_matching_any_train"],
                          r["union_label_conflict_train_rows"] + r["union_label_conflict_test_rows"],
                          info["task_index"], "保留应用类，不合并/扩样；无cap；shift排除量另列"))
        t = v["total"]
        assert sum(r["train_rows"] for r in rr) == t["train_rows"]
        assert sum(r["test_rows"] for r in rr) == t["test_rows"]
        summaries.append(describe(rr, "Malaya", view, t["train_duplicate_excess_rows"],
                                  t["union_label_conflict_affected_rows"], v["decision_18_2"]))
        rows.extend(rr)

    source_paths = [OUT / "tabular/audit_tabular.json", OUT / "malaya/audit_result.json", OUT / "cic/cic_audit.json"]
    code_paths = [ROOT / "ncd" / f"audit_quality_{name}.py" for name in ("tabular", "malaya", "cic")] + [Path(__file__)]
    result = dict(work_order="docs/WORK_ORDER_2026-10-09.md section 18", date="2026-10-13",
                  threshold="test overlap >= 5% overall OR >=20% in any model class",
                  dataset_decisions={s["dataset"]: s["decision"] for s in summaries if not s["view"].startswith("ExpH")},
                  view_summaries=summaries, class_table_rows=len(rows), cic_uncapped_totals=uncapped,
                  limitations=["Exact equality at the existing model-feature representation; not a near-duplicate or packet-level audit.",
                               "CIC18 one original raw CSV is unavailable locally; all ten frozen cache shards were audited.",
                               "Malaya original split triggers dataset-level rerun. Zero overlap in current Exp H views does not waive global conflict removal.",
                               "Read-only data audit: no reruns, source-data changes, oversampling or historical-result changes were performed."],
                  source_receipts=[dict(path=p.relative_to(ROOT).as_posix(), sha256=digest(p)) for p in source_paths],
                  code_receipts=[dict(path=p.relative_to(ROOT).as_posix(), sha256=digest(p)) for p in code_paths])
    write_csv(OUT / "audit_table.csv", rows)
    write_csv(OUT / "cic_rare_class_table.csv", rare)
    (OUT / "audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# 数据质量审计 — 2026-10-13", "",
             "依据 WORK_ORDER §18；范围为只读统计。Exp H 持续运行，没有停止、重启、修改训练、数据或历史结果；没有过采样、合成或新增测试重采样。CRR 否决保持不变。", "",
             "## 结论（按 §18.2，不改变阈值）", "",
             "数据集级：**NSL-KDD、CIC-IDS-2017、CSE-CIC-IDS2018 保持；UNSW-NB15、Malaya 去重重跑。**", "",
             "总测试重复率 ≥5%，或任一模型类 ≥20%，即触发。训练内重复率和标签冲突另报，不替换预登记阈值。", "",
             "| 数据集 / 实际视图 | 训练 / 测试行数 | 训练内 excess | 测试匹配任意训练行 | 最大逐类比例 | 冲突行数（合并集） | 视图判定 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for s in summaries:
        lines.append(f"| {s['dataset']} / {s['view']} | {s['train_rows']:,} / {s['test_rows']:,} | {s['global_train_excess_rows']:,} ({pct(s['global_train_excess_rate'])}) | {s['test_overlap_rows']:,}/{s['test_rows']:,} ({pct(s['test_overlap_rate'])}) | {s['max_class']} {pct(s['max_class_overlap_rate'])} | {s['union_label_conflict_rows']:,} | {s['decision']} |")
    lines += ["", "- **UNSW-NB15（高风险）**：总比例及 DoS、Reconnaissance 逐类比例触发，当前划分可能高估同分布表现。下一阶段应全局去重、删除全部标签冲突组，再划分并重跑适用实验；保留旧版结果并列。",
              "- **Malaya（高风险）**：原 capture 划分触发。当前 Exp H 已采用 grouped ID、偏移测试排除原训练池的 3,780 个重复行，因此两个实际测试视图均低于阈值；但 fit 中仍有 5,366 行 / 40 组标签冲突。视图级‘保持’不豁免数据集级全局去重/冲突删除和适用实验重跑要求。本轮按 §18.1 只统计，当前 Exp H 不暂停。",
              "- **CIC（保持，但有明显训练重复与冲突）**：实际使用的是冻结 feature-hash-group 缓存，不是对原 CSV 临时调用随机行划分。全部缓存行独立复算划分零不一致；原始未截断及实际截断视图均无跨集重复。训练内重复仍保留，不能把‘零泄漏’写成‘已全局去重’或‘标签正确’。",
              "- **NSL-KDD（保持）**：KDDTest-21 是同一测试集的困难子集，不是新的外部数据集；529/11,850（4.464%）匹配训练，最大类 DoS 6.423%，也低于阈值。原始攻击细标签冲突 247 行，合并模型类后 210 行；粗化标签不能消除原始标注不确定性。", "",
              "## 口径、截断与验证", "",
              "一行对应现有流水线的一条特征记录。比较全部实际特征列，排除目标标签、记录 ID、NSL difficulty 等非输入字段。NSL/UNSW 使用原始混合类型数值/类别元组；CIC/Malaya 使用冻结 float32 输入特征（缩放前），故这是模型输入级精确重复，不是原 CSV 文本或近重复审计。signed zero 规范化；哈希仅分桶，最终完整特征字节/元组判等，非概率哈希相等。", "",
              "训练 excess=(N−唯一特征数)/N；逐类表在类内计算，汇总行跨类计算，遇跨标签碰撞时两者不可直接相加。测试重复分子为每个匹配任意训练行的测试记录（不论标签）；标签冲突为 train∪test 中同特征却不同模型标签组的所有成员，非组数或行对数。冲突不单独证明人工标签错误，也可能反映特征表达不足。", "",
              "实际 caps：NSL、UNSW、Malaya 无 cap；CIC17 仅 Normal 160,000/40,000，其他类没有 40k/10k cap；CIC18 每类 40,000/10,000，包括 Normal。与工作单举例不同之处按实际代码记录，没有在审计中更改。现存测试 caps 已改变原始测试分布，本轮不新增截断/重采样，不能称 capped 视图保留完整原始分布。", "",
              "CIC 缓存未截断敏感性：", "",
              "| 数据集 | train / test | 训练 excess | 跨集匹配 | 合并冲突行 |", "|---|---:|---:|---:|---:|"]
    for r in uncapped:
        lines.append(f"| {r['dataset']} | {r['train_rows']:,} / {r['test_rows']:,} | {r['train_duplicate_excess']:,} ({pct(r['train_duplicate_excess_rate'])}) | {r['test_matches_any_train_rows']:,} | {r['conflict_rows']:,} |")
    lines += ["", "## 主审计表：数据集 × 模型类", "",
              "括号前为 count，百分比以该行 train 或 test 为分母。任务编号从 0 开始。Malaya Exp H 的 shift before-cap 已是排除重叠后的集合；排除前后见下方。全部机器可读字段见 [audit_table.csv](results/data_quality/section18/audit_table.csv)。", "",
              "| 数据集 / 视图 | 类 | 训练 cap前→实际 | 测试 cap前→实际 | 类内训练 excess | 测试匹配训练 | 合并冲突行 | task / 处理 |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        small = "；小样本保留" if min(r['train_rows'], r['test_rows']) < 100 else "；保留"
        lines.append(f"| {r['dataset']} / {r['view']} | {r['class_name']} | {r['train_before_cap']:,}→{r['train_rows']:,} | {r['test_before_cap']:,}→{r['test_rows']:,} | {r['train_within_class_excess_rows']:,} ({pct(r['train_within_class_excess_rate'])}) | {r['test_overlap_rows']:,} ({pct(r['test_overlap_rate'])}) | {r['union_label_conflict_rows']:,} | {r['task_index']}{small} |")
    lines += ["", "### Malaya 偏移集合排除记录（历史处理，本次未改变）", "",
              "| 应用类 | 原偏移 test | 排除重叠 | 当前偏移 test |", "|---|---:|---:|---:|"]
    for r in malaya["class_counts_and_handling"]:
        lines.append(f"| {r['class_name']} | {r['shift_before_overlap_exclusion']} | {r['shift_overlap_excluded_rows']} | {r['shift_after_overlap_exclusion']} |")
    lines += ["", "## 稀有类处理（不扩样）", "",
              "区分原始细标签总数 <100 与某个 train/test 分割 <100。NSL/UNSW 的所有稀有原始标签及零样本分割保存在 [rare_labels.csv](results/data_quality/section18/tabular/rare_labels.csv)，两数据集全部细标签计数见 [all_source_labels.csv](results/data_quality/section18/tabular/all_source_labels.csv)。CIC 原始细标签计数见 cic/*_raw_labels.csv。", "",
              "下表含总量 <100，以及已核实某个分割 <100 的细标签；未逐个复原的大类细标签分割不宣称已核验。", "",
              "| CIC 小样本原始标签 | 原始总数 | 既有模型类 / task | cap 前 train/test | 实际 train/test | 处理 |", "|---|---:|---|---:|---:|---|"]
    for r in rare:
        lines.append(f"| {r['dataset']} / {r['source_label']} | {r['raw_rows']} | {r['model_class']} / {r['task_index']} | {r['uncapped_train']}/{r['uncapped_test']} | {r['used_train']}/{r['used_test']} | {r['handling']} |")
    lines += ["", "NSL U2R 模型类为 train 52/test 67（task3），UNSW Worms 为 130/44（task4），按原序列保留，不通过合成增加样本。Malaya TeamViewer 总量 814 并非总类 <100，但原 test 54、当前 shift 37，保留原应用标签（task3）；该类 recall 的离散性/方差需单独说明。Malaya 是应用识别，不把这些类写成攻击类型。CIC 细标签按既有语义映射进入攻击家族，非此次为增样本而合并。", "",
              "## CIC 已知标签问题与修正版来源", "",
              "已核实 Engelen、Rimmer、Joosen：*Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*，IEEE S&P Workshops 2021，pp.7–12，DOI 10.1109/SPW53761.2021.00009。[作者项目页](https://intrusion-detection.distrinet-research.be/WTMC2021/)及[工具/修正版数据页](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html)提供修正流构造与标签的材料。作者另提供覆盖 CIC17/18 的[2022 后续研究](https://intrusion-detection.distrinet-research.be/CNS2022/index.html)及[更新数据入口](https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html)。", "",
              "**本轮未采用修正版**：没有下载替换数据或改变冻结实验；当前缓存来自原始发布版 CSV。此处只核实来源与可得性，不宣称修正版已复核或能保证无错误。全局去重不等于修复流构造/标注问题；若后续迁移，应单列版本与新旧结果，不能静默覆盖。", "",
              "## 来源、可复现性和未验证边界", "",
              "完整扫描而非抽样：NSL/UNSW 的结果与独立 pandas 全特征 merge 交叉验证；Malaya 的31个原 CSV及31个缓存 shard 哈希核验；CIC 两套缓存所有行经精确分组和原 builder 划分复算。CIC17 原始 CSV全覆盖，CIC18 本地缺一份原始 CSV（见 JSON missing_files），但对应10份原始来源的426个冻结缓存分片全部覆盖。因而 CIC18 全量原始细标签频数不能宣称完整；缓存级重复率/冲突/阈值判定是完整的。", "",
              "审计脚本以有界 CPU 线程运行，不使用 GPU；未对运行中的 Exp H 代码或数据做修补。审计代码基于 ce48bfc（本次 git pull 后，新增审计文件在测量时尚未提交）；精确脚本 SHA256 和输入结果 SHA256 已记入 [audit_summary.json](results/data_quality/section18/audit_summary.json)。CIC 的533个实际缓存分片和8个附属文件均逐文件哈希匹配；清单文件字节哈希已记录但无外部预期值，原始CSV完整文件哈希没有在本轮重新核验，详见 [cic_input_receipts.json](results/data_quality/section18/cic/cic_input_receipts.json)。源文件、特征名、映射、caps、版本、命令及日志位置见各审计 JSON。日志原件留本地，含个人路径的日志不发布。", "",
              "复算命令（从仓库根，数据路径以现有配置为准）：", "", "```powershell",
              "python -I ncd/audit_quality_tabular.py --data-root ../kaggle_leon_release/datasets --output-dir ncd/results/data_quality/section18/tabular",
              "python -I ncd/audit_quality_malaya.py --config ncd/configs/exp_h_section17.json --output-dir ncd/results/data_quality/section18/malaya",
              "# CIC 精确命令/缓存路径见 cic/cic_audit.json 的 command/configuration",
              "python -I ncd/summarize_quality_section18.py", "```", "",
              "可读复算 companion：[section18_data_quality_audit.ipynb](notebooks/section18_data_quality_audit.ipynb)。该 notebook 的标准 Python cell 执行与 JSON 结构检查不等于 Jupyter kernel/前端渲染验收；环境未配置 Jupyter 时按 notebook 内命令完成该可选验证。主表和阈值判定由已执行的审计脚本与独立复算支撑。", "",
              "下一步依据 §18.2 保留去重前版本，实施全局去重（冲突组全部删除）再划分，并在受影响的适用实验中并列报告新旧结果；这不是本轮只读审计已完成的工作。保持 Macro-F1、平衡准确率、逐类 recall；不增加过采样、SMOTE或新的测试重采样。"]
    s1 = cic.get("exp_h_s1", {})
    if s1:
        lines += ["", "### Exp H S1 实际对齐特征的补充检查", "",
                  "| S1 视图 | 训练 / 测试 | 跨集匹配 | 视图阈值结论 |", "|---|---:|---:|---|"]
        for name in ("source_id", "target_shift"):
            v = s1[name]
            t = next(r for r in v["rows"] if r["class_name"] == "__all__")
            lines.append(f"| {name} | {t['train_rows']:,}/{t['test_rows']:,} | {t['test_matches_any_train_rows']:,} ({pct(t['test_matches_any_train_rate'])}) | {v['decision']} |")
        lines += ["", "此补充以 S1 对齐后的77个特征比较；仍不构成偏移危害成立/不成立的科学判定。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(dataset_decisions=result["dataset_decisions"], class_table_rows=len(rows),
                          report=REPORT.relative_to(ROOT).as_posix()), ensure_ascii=False))


if __name__ == "__main__":
    build()
