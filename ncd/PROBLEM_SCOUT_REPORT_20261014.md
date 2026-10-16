# 类增量 NIDS 选题侦察｜步数对照与历史资格判定

2026-10-16，Section22：**150组正式步数对照完成，按预声明22.3判为 A（协议假象），不是 B。** Replay S2000 对五个指定小类的五seed平均 recall 均达到原 Joint 的50%以上；这只否定以原短训练下的“学不进去”直接设计机制，不表示遗忘、总体差距或研究问题均已解决。这是一项 RQ2/RO2 协议敏感性检查，不提出机制，不改 RQ1/RQ3 或整体创新完成门槛。Section21 原始结果与分析保留不变。依22.4，三篇“全文不可得”的检索停止，未知改善不写成零。

## Section22 步数对照

[执行入口](run_step_budget.py)：NSL-KDD、去重 UNSW-NB15、CIC-IDS-2017；Replay、Replay+LwF、RCL-tab、iCaRL、DER++；seeds 1/2/3/4/42；S=500/2000，共150个完整增量序列。每任务更新数为 `max(8 * ceil(实际训练行数 / batch_size), S)`，补足部分允许在额外 epoch 中途停止，任务内 Adam 不重置。Replay/LwF/RCL/iCaRL 的实际训练行数包括已存回放；DER++ 保持原来的当前任务流和独立回放采样。每个 checkpoint 记录实际更新数、完整/部分 epoch 和样本呈现次数。

只改变训练步数下限。数据缓存、学习率、类别平衡 CE、任务顺序、200/类存储和各方法参数不变；不重跑 Joint/CSE18。第22.1节文字中的 batch384 与原始结果绑定配置不符，**实际 Section21 batch256，Section22 保持256**。正式训练/测试全量；冒烟的1 seed、1 epoch、每类至多64行不属于正式结果。

按原始 checkpoint 的实际行数重算，最后任务 Replay/LwF/RCL/iCaRL 的原更新数为 NSL32、UNSW96、CIC17 88（当前新类行数52/1284/1616，加已存旧类800/1600/1200行）；DER++ 当前流更新数为8/48/56，旧回放另在每步内采样。这修正了第22.1节基于batch384的24/64步估算，不改变实验问题。

**执行。** 源提交`0a16c4b0b2c407dc2a97bfd7658635106a681b26`；命令`python -I -u ncd/run_step_budget.py --output-dir ncd/results/problem_scout/section22`。2026-10-16 02:40:14–09:21:09 MYT，本地RTX4060Ti依次完成S2000/S500各75组、各325个checkpoint，两档退出码均为0。[驱动结果](results/problem_scout/section22/result.json)与[尝试日志](results/problem_scout/section22/driver_attempts.jsonl)记录子进程耗时13725.18/10330.29s；这是进程经过时间，不是优化时间。30个真实数据冒烟单元不纳入科学结果。

**22.3判定数字。** 单位%，五seed均值±样本SD；Joint列为复用的原协议均值。比例为均值之比，不是逐seed比率的平均。

| Replay最后任务新类 | 原协议 | S500 | S2000 | Joint | S2000/Joint |
|---|---:|---:|---:|---:|---:|
| NSL U2R | 11.94±16.65 | 64.48±8.00 | 61.79±7.94 | 43.58 | 1.418 |
| UNSW Shellcode | 0.00±0.00 | 66.09±4.24 | 76.89±3.36 | 73.56 | 1.045 |
| UNSW Worms | 0.00±0.00 | 81.25±3.83 | 73.75±6.48 | 51.88 | 1.422 |
| CIC17 Botnet | 0.00±0.00 | 56.70±50.12 | 99.68±0.12 | 98.00 | 1.017 |
| CIC17 Infiltration | 0.00±0.00 | 16.67±28.87 | 66.67±0.00 | 26.67 | 2.500 |

五类全部≥0.5，A满足；全部<0.2的B不满足。CIC17 Infiltration测试只有6条，S2000每seed均为4/6；Joint中两个seed为0，逐seed比率保留null，不能据此声称稳健的大样本泛化。S500的Botnet跨seed差异很大；增加步数不是所有指标单调改善的保证。

**旧类代价与整体指标。** Replay相对原协议的Macro-F1变化，NSL/UNSW/CIC17在S500为−2.16/+4.02/+4.94pp，在S2000为−1.50/+3.39/+8.35pp。S2000旧类recall的配对均值变化：NSL Probe−11.16、R2L−4.28pp；UNSW Fuzzers−17.22、Reconnaissance−13.46、Exploits−7.92pp，而Backdoor+52.86pp；CIC17六个旧类均值未下降，但不代表每seed均改善。这里比较相同旧类的最终recall，不把所有低recall都解释为遗忘；最后任务前后变化另列。

**重算G1/G2。** 每个数据集/指标按五seed均值固定选择最佳CIL，再配对计算Joint−CIL，单位pp，±为差距样本SD。Joint是参考而非必然上界。

| 数据集 | Macro-F1原协议 | S500 | S2000 | S2000最佳CIL | S2000 BAcc差距 |
|---|---:|---:|---:|---|---:|
| NSL | −4.17±5.75 | −3.12±2.28 | −3.46±2.10 | LwF | −8.94±2.09（LwF） |
| UNSW | 9.82±2.21 | 6.31±1.56 | 5.14±1.56 | RCL | −1.08±0.93（RCL） |
| CIC17 | 11.68±2.97 | 9.78±3.27 | 8.98±3.42 | DER++ | −3.29±3.20（DER++） |

UNSW/CIC17的S2000 Macro-F1缺口缩小但仍均值≥5pp、五seed全正。S2000 G2逐类包络中，UNSW Fuzzers20.14±2.54pp（RCL）及CIC17 Normal6.18±1.95pp（DER++）仍均值≥5且五seed全正；Exploits/Reconnaissance的均值缺口虽≥5，方向不满足五seed全正。包络不是一个分类器，也不把异类拼成同一个两数据集问题；这些数字供负责人重新选题，不据此宣称新机制或新颖性。

**完整可复算交付。** [S500原始小结果](results/problem_scout/section22/steps_500_v1/result.json)、[S2000原始小结果](results/problem_scout/section22/steps_2000_v1/result.json)保留各checkpoint混淆计数、步数、配置、源提交及日志；权重和数据不随Git发布，权重路径/hash仅为原运行来源记录。[分析JSON](results/problem_scout/section22/analysis_v1/analysis.json)包含完整90+75+75覆盖、G1/G2、判定及输入文件hash；[新/旧类汇总](results/problem_scout/section22/analysis_v1/recall_summary.csv)、[逐seed recall及最后任务前后变化](results/problem_scout/section22/analysis_v1/recall_by_seed.csv)、[全部方法Macro-F1/BAcc汇总](results/problem_scout/section22/analysis_v1/metrics_summary.csv)、[逐seed指标](results/problem_scout/section22/analysis_v1/metrics_by_seed.csv)均含原协议/500/2000，保留均值、样本SD及配对变化；CSV率值为0–1，差值为pp。分析入口为[analyze_step_budget.py](analyze_step_budget.py)，不需要权重或数据即可从所附混淆计数复算。

**下一步。** 按22.3，后续实验采用训练步数下限，由负责人根据新G1/G2重新选题；本次不设计机制，不重跑已完成单元。以下Section21数字只作8-epoch历史记录，不再作为新小类无法学习的直接证据。

## Section21 历史结果（8 epochs）

2026-10-15，Section21：**21.5 合格候选为 0；21.4 仍缺三篇全文的协议/成对改善幅度，故第21节尚未整体完成。** 不提出机制、不改 DGC 否决或 RQ1/RQ3，不等于整个研究完成。

**已运行与核验。** [原入口](run_problem_scout.py)六方法×四数据集×五seed，全量训练/测试、8 epochs、无截断或补样；科学源 ab33ca6/1f35f65。先完成有效包（不按分数选源）合并为[120单元/510 checkpoint/1140行](results/problem_scout/section21/merged_v1/result.json)，含30个不支持的S1 PortScan recall=null；实际权重SHA与独立混淆复算41,890标量通过。原始小结果及[G1–G7分析](results/problem_scout/section21/analysis_v1.json)已在私库6761732b/导师库31cfc5a0发布；[20个job](results/problem_scout/section21/hpc_jobs.json)全部终态，无需再训练。

**主要数字。** 固定五seed均值最优CIL（非逐seed择优）；Macro-F1/BAcc为Joint−CIL，FPR为CIL−Joint，单位pp。±为差距样本SD；★为均值≥5且五seed全正。Joint是参考而非必然最优。

| 数据集 | G1 Macro-F1 | G1 BAcc | G5 Normal FPR |
|---|---:|---:|---:|
| NSL | −4.17±5.75（Replay） | −6.77±2.01（iCaRL） | +1.17（RCL） |
| UNSW | +9.82±2.21★（RCL） | +12.41±1.33★（iCaRL） | −1.55（DER++） |
| CIC17 | +11.68±2.97★（DER++） | −0.02±2.21（DER++） | +8.08★（LwF） |
| CSE18 | +1.04±1.92（DER++） | −1.94±1.04（DER++） | +1.75（LwF） |

G2合格逐类缺口：UNSW Backdoor12.29/Fuzzers8.17、CIC17 Botnet8.38/Normal8.08；包络不是单个分类器，异类不能拼成同一问题。G3区分未学会与遗忘：UNSW RCL Backdoor自身→最终0→3.43%，Exploits90.25→49.82%。G4按各数据集Macro-F1均值最优CIL，“更新类”占旧类错误87.34/84.69/96.20%（UNSW/CIC17/CSE18，汇总计数比，非seed比例均值）；仅旧攻击作分母则73.25/71.29/45.00%。G6 S1 Normal差19.44pp、五seed全正，但只有CIC17→18一个源目标设置。完整逐类/攻击recall见分析文件。

**三方向文献与资格。** [11篇一手来源、协议、表值和访问记录](results/problem_scout/section21/literature_gap_map_v1.json)为有界定向核查，非穷尽综述或跨指标统一“前三”排名。

- G1总体多类差距：UNSW/CIC17数量门槛通过，分数10.7521×2=21.5042；但[Two-Stage](https://doi.org/10.1109/GLOBECOM52923.2024.10901809)/[AdvAs-IIDS](https://doi.org/10.1016/j.eswa.2025.126632)/[ExBCIL](https://doi.org/10.1007/s13042-024-02486-9)等已≥3篇直接工作，不满足≤2篇，也未在本协议测为无效。Attack-adaptive IoT的配置间Macro-F1增益8.69/2.83/18.36/5.60pp不是匹配预算因果对照。
- G2局部类别：无相同类别在两数据集达标。[ACIL作者全文](https://www.researchgate.net/publication/400834463_Anomaly-aware_class-incremental_learning_ACIL_framework_for_network_intrusion_detection_systems)表12–14中Replay−Naive F1为14.7/14.6/16.4pp（依次CIC17/CIC18/NSL）；总buffer1000、50epochs。表17区间与表13/14不一致，F1聚合未明，不能称本协议有效或新ACIL算法提升。
- G6跨环境Normal：[Network-Agnostic CIL 2025](https://doi.org/10.1109/RTSI64020.2025.11212335)直接研究目标正常流量适配；图5 iCaRL+/BiC正常recall99.88/99.96%，前者低0.08pp，不是正常类改善。它训练目标benign，与本次未适应S1不同；本次单设置仍不达两数据集门槛。

**边界与成本。** G3绝对遗忘、G4错误组成不是额外Joint差距，不凑候选。CL旧类仅200/类存储；DER++为适配的观察/logit槽位，可重复源样本；iCaRL精确herding。NSL官方重合664；UNSW原始重合0但模型碰撞13；CIC原始ID重合0但模型碰撞179/4030，冻结重复/冲突保留，非全局去重无噪声。S1有0.559%源目标重合及269异标签，非独立干净复现。稀有类/映射限制保留；历史第15节、第17/19节Exp H旧类完整池重访结果不是本协议证据。近期新增匹配可运行NIDS基线0，最佳仅限已运行集合，缺项不算败北。

G7累计优化/评估/选择99035.11/419.22/648.47s，**不是墙钟**；训练阶段PyTorch allocated/reserved峰值72597504/96468992B、回放payload/allocated624080B，不是整体设备/进程峰值。[尝试成本及测量限制](results/problem_scout/section21/support_v1/attempt_costs.json)：HPC分配85177s=选中77967+完成重复2285+取消重复4925；runner不叠加。连续GPU遥测缺失，本地退出码/部分中止成本未知，异硬件不比速度；未改论文、科学源或数据。

**历史未核项。** Two-Stage、AdvAs-IIDS、ExBCIL经有界检索仍为“全文不可得”，改善幅度/详细协议未知；依Section22.4停止检索。原21.5零候选是原协议下的历史判断；Section22已按A报告，由负责人据步数对照后的数字决定后续选题。
