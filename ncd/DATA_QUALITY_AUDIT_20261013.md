# 数据质量审计 — 2026-10-13

依据 WORK_ORDER §18；范围为只读统计。Exp H 持续运行，没有停止、重启、修改训练、数据或历史结果；没有过采样、合成或新增测试重采样。CRR 否决保持不变。

## 结论（按 §18.2，不改变阈值）

数据集级：**NSL-KDD、CIC-IDS-2017、CSE-CIC-IDS2018 保持；UNSW-NB15、Malaya 去重重跑。**

总测试重复率 ≥5%，或任一模型类 ≥20%，即触发。训练内重复率和标签冲突另报，不替换预登记阈值。

| 数据集 / 实际视图 | 训练 / 测试行数 | 训练内 excess | 测试匹配任意训练行 | 最大逐类比例 | 冲突行数（合并集） | 视图判定 |
|---|---:|---:|---:|---:|---:|---|
| NSL-KDD / original | 125,973 / 22,544 | 16 (0.013%) | 664/22,544 (2.945%) | Probe 5.370% | 210 | 保持 |
| UNSW-NB15 / original | 175,341 / 82,332 | 74,301 (42.375%) | 8,541/82,332 (10.374%) | DoS 49.010% | 40,400 | 去重重跑 |
| CIC-IDS-2017 / ncd_capped | 610,447 / 146,109 | 114,563 (18.767%) | 0/146,109 (0.000%) | Normal 0.000% | 3,723 | 保持 |
| CSE-CIC-IDS2018 / ncd_capped | 240,730 / 60,198 | 79,403 (32.984%) | 0/60,198 (0.000%) | Normal 0.000% | 40,521 | 保持 |
| Malaya / original | 36,372 / 10,370 | 10,995 (30.229%) | 3,780/10,370 (36.451%) | BitTorrent 48.980% | 7,331 | 去重重跑 |
| Malaya / ExpH_ID | 29,098 / 7,274 | 10,456 (35.934%) | 0/7,274 (0.000%) | BitTorrent 0.000% | 5,660 | 保持 |
| Malaya / ExpH_shift | 29,098 / 6,590 | 10,456 (35.934%) | 0/6,590 (0.000%) | BitTorrent 0.000% | 5,366 | 保持 |

- **UNSW-NB15（高风险）**：总比例及 DoS、Reconnaissance 逐类比例触发，当前划分可能高估同分布表现。下一阶段应全局去重、删除全部标签冲突组，再划分并重跑适用实验；保留旧版结果并列。
- **Malaya（高风险）**：原 capture 划分触发。当前 Exp H 已采用 grouped ID、偏移测试排除原训练池的 3,780 个重复行，因此两个实际测试视图均低于阈值；但 fit 中仍有 5,366 行 / 40 组标签冲突。视图级‘保持’不豁免数据集级全局去重/冲突删除和适用实验重跑要求。本轮按 §18.1 只统计，当前 Exp H 不暂停。
- **CIC（保持，但有明显训练重复与冲突）**：实际使用的是冻结 feature-hash-group 缓存，不是对原 CSV 临时调用随机行划分。全部缓存行独立复算划分零不一致；原始未截断及实际截断视图均无跨集重复。训练内重复仍保留，不能把‘零泄漏’写成‘已全局去重’或‘标签正确’。
- **NSL-KDD（保持）**：KDDTest-21 是同一测试集的困难子集，不是新的外部数据集；529/11,850（4.464%）匹配训练，最大类 DoS 6.423%，也低于阈值。原始攻击细标签冲突 247 行，合并模型类后 210 行；粗化标签不能消除原始标注不确定性。

## 口径、截断与验证

一行对应现有流水线的一条特征记录。比较全部实际特征列，排除目标标签、记录 ID、NSL difficulty 等非输入字段。NSL/UNSW 使用原始混合类型数值/类别元组；CIC/Malaya 使用冻结 float32 输入特征（缩放前），故这是模型输入级精确重复，不是原 CSV 文本或近重复审计。signed zero 规范化；哈希仅分桶，最终完整特征字节/元组判等，非概率哈希相等。

训练 excess=(N−唯一特征数)/N；逐类表在类内计算，汇总行跨类计算，遇跨标签碰撞时两者不可直接相加。测试重复分子为每个匹配任意训练行的测试记录（不论标签）；标签冲突为 train∪test 中同特征却不同模型标签组的所有成员，非组数或行对数。冲突不单独证明人工标签错误，也可能反映特征表达不足。

实际 caps：NSL、UNSW、Malaya 无 cap；CIC17 仅 Normal 160,000/40,000，其他类没有 40k/10k cap；CIC18 每类 40,000/10,000，包括 Normal。与工作单举例不同之处按实际代码记录，没有在审计中更改。现存测试 caps 已改变原始测试分布，本轮不新增截断/重采样，不能称 capped 视图保留完整原始分布。

CIC 缓存未截断敏感性：

| 数据集 | train / test | 训练 excess | 跨集匹配 | 合并冲突行 |
|---|---:|---:|---:|---:|
| CIC-IDS-2017 | 2,262,203 / 565,673 | 264,889 (11.709%) | 0 | 6,790 |
| CSE-CIC-IDS2018 | 12,882,081 / 3,255,102 | 3,780,187 (29.345%) | 0 | 1,569,428 |

## 主审计表：数据集 × 模型类

括号前为 count，百分比以该行 train 或 test 为分母。任务编号从 0 开始。Malaya Exp H 的 shift before-cap 已是排除重叠后的集合；排除前后见下方。全部机器可读字段见 [audit_table.csv](results/data_quality/section18/audit_table.csv)。

| 数据集 / 视图 | 类 | 训练 cap前→实际 | 测试 cap前→实际 | 类内训练 excess | 测试匹配训练 | 合并冲突行 | task / 处理 |
|---|---|---:|---:|---:|---:|---:|---|
| NSL-KDD / original | Normal | 67,343→67,343 | 9,711→9,711 | 0 (0.000%) | 105 (1.081%) | 108 | 0；保留 |
| NSL-KDD / original | DoS | 45,927→45,927 | 7,460→7,460 | 0 (0.000%) | 400 (5.362%) | 27 | 0；保留 |
| NSL-KDD / original | Probe | 11,656→11,656 | 2,421→2,421 | 9 (0.077%) | 130 (5.370%) | 1 | 1；保留 |
| NSL-KDD / original | R2L | 995→995 | 2,885→2,885 | 0 (0.000%) | 29 (1.005%) | 73 | 2；保留 |
| NSL-KDD / original | U2R | 52→52 | 67→67 | 0 (0.000%) | 0 (0.000%) | 1 | 3；小样本保留 |
| UNSW-NB15 / original | Normal | 56,000→56,000 | 37,000→37,000 | 4,110 (7.339%) | 1,472 (3.978%) | 928 | 0；保留 |
| UNSW-NB15 / original | Generic | 40,000→40,000 | 18,871→18,871 | 35,819 (89.547%) | 1,722 (9.125%) | 373 | 0；保留 |
| UNSW-NB15 / original | Exploits | 33,393→33,393 | 11,132→11,132 | 13,549 (40.574%) | 1,802 (16.188%) | 16,507 | 1；保留 |
| UNSW-NB15 / original | Fuzzers | 18,184→18,184 | 6,062→6,062 | 2,034 (11.186%) | 261 (4.306%) | 3,617 | 1；保留 |
| UNSW-NB15 / original | DoS | 12,264→12,264 | 4,089→4,089 | 8,458 (68.966%) | 2,004 (49.010%) | 12,663 | 2；保留 |
| UNSW-NB15 / original | Reconnaissance | 10,491→10,491 | 3,496→3,496 | 2,969 (28.300%) | 1,138 (32.551%) | 2,246 | 2；保留 |
| UNSW-NB15 / original | Analysis | 2,000→2,000 | 677→677 | 406 (20.300%) | 125 (18.464%) | 2,061 | 3；保留 |
| UNSW-NB15 / original | Backdoor | 1,746→1,746 | 583→583 | 211 (12.085%) | 4 (0.686%) | 1,977 | 3；保留 |
| UNSW-NB15 / original | Shellcode | 1,133→1,133 | 378→378 | 42 (3.707%) | 13 (3.439%) | 13 | 4；保留 |
| UNSW-NB15 / original | Worms | 130→130 | 44→44 | 3 (2.308%) | 0 (0.000%) | 15 | 4；小样本保留 |
| CIC-IDS-2017 / ncd_capped | Normal | 1,811,756→160,000 | 459,564→40,000 | 4,732 (2.958%) | 0 (0.000%) | 68 | 0；保留 |
| CIC-IDS-2017 / ncd_capped | DoS | 206,147→206,147 | 45,576→45,576 | 51,153 (24.814%) | 0 (0.000%) | 3,561 | 0；保留 |
| CIC-IDS-2017 / ncd_capped | DDoS | 102,384→102,384 | 25,641→25,641 | 9 (0.009%) | 0 (0.000%) | 0 | 1；保留 |
| CIC-IDS-2017 / ncd_capped | Bruteforce | 11,374→11,374 | 2,458→2,458 | 4,074 (35.819%) | 0 (0.000%) | 0 | 1；保留 |
| CIC-IDS-2017 / ncd_capped | PortScan | 127,198→127,198 | 31,606→31,606 | 54,501 (42.847%) | 0 (0.000%) | 94 | 2；保留 |
| CIC-IDS-2017 / ncd_capped | WebAttack | 1,728→1,728 | 452→452 | 30 (1.736%) | 0 (0.000%) | 0 | 2；保留 |
| CIC-IDS-2017 / ncd_capped | Botnet | 1,586→1,586 | 370→370 | 8 (0.504%) | 0 (0.000%) | 0 | 3；保留 |
| CIC-IDS-2017 / ncd_capped | Infiltration | 30→30 | 6→6 | 0 (0.000%) | 0 (0.000%) | 0 | 3；小样本保留 |
| CSE-CIC-IDS2018 / ncd_capped | Normal | 10,702,724→40,000 | 2,687,525→10,000 | 1,755 (4.387%) | 0 (0.000%) | 2,351 | 0；保留 |
| CSE-CIC-IDS2018 / ncd_capped | DoS | 499,404→40,000 | 154,896→10,000 | 16,499 (41.247%) | 0 (0.000%) | 10,319 | 0；保留 |
| CSE-CIC-IDS2018 / ncd_capped | DDoS | 1,014,521→40,000 | 249,412→10,000 | 7,145 (17.863%) | 0 (0.000%) | 0 | 1；保留 |
| CSE-CIC-IDS2018 / ncd_capped | Bruteforce | 307,402→40,000 | 73,541→10,000 | 30,112 (75.280%) | 0 (0.000%) | 25,130 | 1；保留 |
| CSE-CIC-IDS2018 / ncd_capped | WebAttack | 730→730 | 198→198 | 49 (6.712%) | 0 (0.000%) | 1 | 2；保留 |
| CSE-CIC-IDS2018 / ncd_capped | Infiltration | 128,579→40,000 | 32,060→10,000 | 3,086 (7.715%) | 0 (0.000%) | 2,720 | 2；保留 |
| CSE-CIC-IDS2018 / ncd_capped | Botnet | 228,721→40,000 | 57,470→10,000 | 19,366 (48.415%) | 0 (0.000%) | 0 | 3；保留 |
| Malaya / original | BitTorrent | 21,719→21,719 | 5,537→5,537 | 9,141 (42.088%) | 2,712 (48.980%) | 4,508 | 0；保留 |
| Malaya / original | ChromeRDP | 900→900 | 381→381 | 143 (15.889%) | 108 (28.346%) | 259 | 0；保留 |
| Malaya / original | Discord | 605→605 | 227→227 | 102 (16.860%) | 40 (17.621%) | 161 | 1；保留 |
| Malaya / original | EA Origin | 3,360→3,360 | 1,200→1,200 | 192 (5.714%) | 193 (16.083%) | 412 | 1；保留 |
| Malaya / original | Microsoft Teams | 531→531 | 445→445 | 60 (11.299%) | 123 (27.640%) | 198 | 2；保留 |
| Malaya / original | Slack | 919→919 | 782→782 | 151 (16.431%) | 369 (47.187%) | 548 | 2；保留 |
| Malaya / original | Steam | 4,781→4,781 | 887→887 | 321 (6.714%) | 67 (7.554%) | 421 | 3；保留 |
| Malaya / original | TeamViewer | 760→760 | 54→54 | 149 (19.605%) | 17 (31.481%) | 194 | 3；小样本保留 |
| Malaya / original | Webex | 1,760→1,760 | 567→567 | 346 (19.659%) | 109 (19.224%) | 411 | 4；保留 |
| Malaya / original | Zoom | 1,037→1,037 | 290→290 | 163 (15.718%) | 42 (14.483%) | 219 | 4；保留 |
| Malaya / ExpH_ID | BitTorrent | 17,375→17,375 | 4,344→4,344 | 8,857 (50.976%) | 0 (0.000%) | 3,891 | 0；保留 |
| Malaya / ExpH_ID | ChromeRDP | 720→720 | 180→180 | 125 (17.361%) | 0 (0.000%) | 150 | 0；保留 |
| Malaya / ExpH_ID | Discord | 484→484 | 121→121 | 83 (17.149%) | 0 (0.000%) | 122 | 1；保留 |
| Malaya / ExpH_ID | EA Origin | 2,688→2,688 | 672→672 | 168 (6.250%) | 0 (0.000%) | 216 | 1；保留 |
| Malaya / ExpH_ID | Microsoft Teams | 425→425 | 106→106 | 46 (10.824%) | 0 (0.000%) | 75 | 2；保留 |
| Malaya / ExpH_ID | Slack | 735→735 | 184→184 | 127 (17.279%) | 0 (0.000%) | 180 | 2；保留 |
| Malaya / ExpH_ID | Steam | 3,825→3,825 | 956→956 | 300 (7.843%) | 0 (0.000%) | 354 | 3；保留 |
| Malaya / ExpH_ID | TeamViewer | 608→608 | 152→152 | 129 (21.217%) | 0 (0.000%) | 177 | 3；保留 |
| Malaya / ExpH_ID | Webex | 1,408→1,408 | 352→352 | 304 (21.591%) | 0 (0.000%) | 318 | 4；保留 |
| Malaya / ExpH_ID | Zoom | 830→830 | 207→207 | 123 (14.819%) | 0 (0.000%) | 177 | 4；保留 |
| Malaya / ExpH_shift | BitTorrent | 17,375→17,375 | 2,825→2,825 | 8,857 (50.976%) | 0 (0.000%) | 3,847 | 0；保留 |
| Malaya / ExpH_shift | ChromeRDP | 720→720 | 273→273 | 125 (17.361%) | 0 (0.000%) | 131 | 0；保留 |
| Malaya / ExpH_shift | Discord | 484→484 | 187→187 | 83 (17.149%) | 0 (0.000%) | 100 | 1；保留 |
| Malaya / ExpH_shift | EA Origin | 2,688→2,688 | 1,007→1,007 | 168 (6.250%) | 0 (0.000%) | 189 | 1；保留 |
| Malaya / ExpH_shift | Microsoft Teams | 425→425 | 322→322 | 46 (10.824%) | 0 (0.000%) | 59 | 2；保留 |
| Malaya / ExpH_shift | Slack | 735→735 | 413→413 | 127 (17.279%) | 0 (0.000%) | 152 | 2；保留 |
| Malaya / ExpH_shift | Steam | 3,825→3,825 | 820→820 | 300 (7.843%) | 0 (0.000%) | 328 | 3；保留 |
| Malaya / ExpH_shift | TeamViewer | 608→608 | 37→37 | 129 (21.217%) | 0 (0.000%) | 153 | 3；小样本保留 |
| Malaya / ExpH_shift | Webex | 1,408→1,408 | 458→458 | 304 (21.591%) | 0 (0.000%) | 270 | 4；保留 |
| Malaya / ExpH_shift | Zoom | 830→830 | 248→248 | 123 (14.819%) | 0 (0.000%) | 137 | 4；保留 |

### Malaya 偏移集合排除记录（历史处理，本次未改变）

| 应用类 | 原偏移 test | 排除重叠 | 当前偏移 test |
|---|---:|---:|---:|
| BitTorrent | 5537 | 2712 | 2825 |
| ChromeRDP | 381 | 108 | 273 |
| Discord | 227 | 40 | 187 |
| EA Origin | 1200 | 193 | 1007 |
| Microsoft Teams | 445 | 123 | 322 |
| Slack | 782 | 369 | 413 |
| Steam | 887 | 67 | 820 |
| TeamViewer | 54 | 17 | 37 |
| Webex | 567 | 109 | 458 |
| Zoom | 290 | 42 | 248 |

## 稀有类处理（不扩样）

区分原始细标签总数 <100 与某个 train/test 分割 <100。NSL/UNSW 的所有稀有原始标签及零样本分割保存在 [rare_labels.csv](results/data_quality/section18/tabular/rare_labels.csv)，两数据集全部细标签计数见 [all_source_labels.csv](results/data_quality/section18/tabular/all_source_labels.csv)。CIC 原始细标签计数见 cic/*_raw_labels.csv。

下表含总量 <100，以及已核实某个分割 <100 的细标签；未逐个复原的大类细标签分割不宣称已核验。

| CIC 小样本原始标签 | 原始总数 | 既有模型类 / task | cap 前 train/test | 实际 train/test | 处理 |
|---|---:|---|---:|---:|---|
| CIC-IDS-2017 / Heartbleed | 11 | DoS / 0 | 10/1 | 10/1 | 并入既有模型类；保留全部记录；不扩样 |
| CIC-IDS-2017 / Infiltration | 36 | Infiltration / 3 | 30/6 | 30/6 | 保留原类；不扩样 |
| CIC-IDS-2017 / Web Attack – Sql Injection | 21 | WebAttack / 2 | 19/2 | 19/2 | 并入既有模型类；保留全部记录；不扩样 |
| CSE-CIC-IDS2018 / Brute Force -XSS | 230 | WebAttack / 2 | 186/44 | 186/44 | 并入既有模型类；保留全部记录；不扩样 |
| CSE-CIC-IDS2018 / SQL Injection | 87 | WebAttack / 2 | 66/21 | 66/21 | 并入既有模型类；保留全部记录；不扩样 |

NSL U2R 模型类为 train 52/test 67（task3），UNSW Worms 为 130/44（task4），按原序列保留，不通过合成增加样本。Malaya TeamViewer 总量 814 并非总类 <100，但原 test 54、当前 shift 37，保留原应用标签（task3）；该类 recall 的离散性/方差需单独说明。Malaya 是应用识别，不把这些类写成攻击类型。CIC 细标签按既有语义映射进入攻击家族，非此次为增样本而合并。

## CIC 已知标签问题与修正版来源

已核实 Engelen、Rimmer、Joosen：*Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*，IEEE S&P Workshops 2021，pp.7–12，DOI 10.1109/SPW53761.2021.00009。[作者项目页](https://intrusion-detection.distrinet-research.be/WTMC2021/)及[工具/修正版数据页](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html)提供修正流构造与标签的材料。作者另提供覆盖 CIC17/18 的[2022 后续研究](https://intrusion-detection.distrinet-research.be/CNS2022/index.html)及[更新数据入口](https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html)。

**本轮未采用修正版**：没有下载替换数据或改变冻结实验；当前缓存来自原始发布版 CSV。此处只核实来源与可得性，不宣称修正版已复核或能保证无错误。全局去重不等于修复流构造/标注问题；若后续迁移，应单列版本与新旧结果，不能静默覆盖。

## 来源、可复现性和未验证边界

完整扫描而非抽样：NSL/UNSW 的结果与独立 pandas 全特征 merge 交叉验证；Malaya 的31个原 CSV及31个缓存 shard 哈希核验；CIC 两套缓存所有行经精确分组和原 builder 划分复算。CIC17 原始 CSV全覆盖，CIC18 本地缺一份原始 CSV（见 JSON missing_files），但对应10份原始来源的426个冻结缓存分片全部覆盖。因而 CIC18 全量原始细标签频数不能宣称完整；缓存级重复率/冲突/阈值判定是完整的。

审计脚本以有界 CPU 线程运行，不使用 GPU；未对运行中的 Exp H 代码或数据做修补。审计代码基于 ce48bfc（本次 git pull 后，新增审计文件在测量时尚未提交）；精确脚本 SHA256 和输入结果 SHA256 已记入 [audit_summary.json](results/data_quality/section18/audit_summary.json)。CIC 的533个实际缓存分片和8个附属文件均逐文件哈希匹配；清单文件字节哈希已记录但无外部预期值，原始CSV完整文件哈希没有在本轮重新核验，详见 [cic_input_receipts.json](results/data_quality/section18/cic/cic_input_receipts.json)。源文件、特征名、映射、caps、版本、命令及日志位置见各审计 JSON。日志原件留本地，含个人路径的日志不发布。

复算命令（从仓库根，数据路径以现有配置为准）：

```powershell
python -I ncd/audit_quality_tabular.py --data-root ../kaggle_leon_release/datasets --output-dir ncd/results/data_quality/section18/tabular
python -I ncd/audit_quality_malaya.py --config ncd/configs/exp_h_section17.json --output-dir ncd/results/data_quality/section18/malaya
# CIC 精确命令/缓存路径见 cic/cic_audit.json 的 command/configuration
python -I ncd/summarize_quality_section18.py
```

可读复算 companion：[section18_data_quality_audit.ipynb](notebooks/section18_data_quality_audit.ipynb)。该 notebook 的标准 Python cell 执行与 JSON 结构检查不等于 Jupyter kernel/前端渲染验收；环境未配置 Jupyter 时按 notebook 内命令完成该可选验证。主表和阈值判定由已执行的审计脚本与独立复算支撑。

下一步依据 §18.2 保留去重前版本，实施全局去重（冲突组全部删除）再划分，并在受影响的适用实验中并列报告新旧结果；这不是本轮只读审计已完成的工作。保持 Macro-F1、平衡准确率、逐类 recall；不增加过采样、SMOTE或新的测试重采样。

### Exp H S1 实际对齐特征的补充检查

| S1 视图 | 训练 / 测试 | 跨集匹配 | 视图阈值结论 |
|---|---:|---:|---|
| source_id | 610,447/146,109 | 0 (0.000%) | 保持 |
| target_shift | 610,447/60,198 | 46 (0.076%) | 保持 |

此补充以 S1 对齐后的77个特征比较；仍不构成偏移危害成立/不成立的科学判定。
