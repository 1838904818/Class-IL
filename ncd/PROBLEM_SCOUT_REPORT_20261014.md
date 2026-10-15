# 类增量 NIDS 选题侦察｜第 2 天量化结果

2026-10-15，Section21。**全量训练与独立 G1–G7 已完成；定向文献的改进幅度核验尚未齐全，因此第 21 节最终交付仍未完成。** 只量化问题，不提出机制；DGC 否决及 RQ1/RQ3 保持，本阶段不回答后二者。

## 正式证据与 G1–G7

固定采用先完整有效完成的包，不按分数选源：[合并原始小结果](results/problem_scout/section21/merged_v1/result.json)有 **120 唯一单元、510 checkpoint、1140 最终行**（30 个不支持的 S1 PortScan recall=null）。342 个本地实际权重哈希与 168 个远端实际观测核对通过；独立从混淆计数复算 41,890 个标量无差异（容差 2e-12）。[完整逐类、逐 seed、混淆与成本分析](results/problem_scout/section21/analysis_v1.json)保留全部 G1–G7；[可查验的核验日志与尝试成本](results/problem_scout/section21/support_v1/attempt_costs.json)保留原始→脱敏 SHA。无权重或数据集上传。

下表为五 seed **固定均值最优 CIL 方法**对应差距，单位 pp；Macro-F1/BAcc/攻击 recall 为 Joint−CIL，FPR 为 CIL−Joint。Joint 是参考比较，不是实测中必然最优；不同列可能选不同方法。括号为方法，± 为差距的样本标准差，★ 表示均值≥5pp且五 seed 全正。

| 数据集 | G1 Macro-F1 | G1 BAcc | G5 攻击 recall | G5 Normal FPR |
|---|---:|---:|---:|---:|
| NSL | −4.17±5.75（Replay） | −6.77±2.01（iCaRL） | −13.58（iCaRL） | +1.17（RCL） |
| UNSW | **+9.82±2.21★（RCL）** | +12.41±1.33★（iCaRL） | −0.09（iCaRL） | −1.55（DER++） |
| CIC17 | **+11.68±2.97★（DER++）** | −0.02±2.21（DER++） | +0.21（iCaRL） | +8.08★（LwF） |
| CSE18 | +1.04±1.92（DER++） | −1.94±1.04（DER++） | +0.44（DER++） | +1.75（LwF） |

| 项 | 主要观察及解释边界 |
|---|---|
| G2 逐类 | 达到≥5pp且五 seed 全正：UNSW Backdoor **12.29**（280训练行）、Fuzzers **8.17**（14999）；CIC17 Botnet **8.38**（1586）、Normal **8.08**（1811756）。是逐类方法包络，不是一个分类器；不同类别不可拼成同一跨数据集候选。CSE18 Infiltration均值7.00，但方向不一致。 |
| G3 学会/遗忘 | 固定各数据集 Macro-F1 最优方法后，UNSW RCL 的 Exploits 自身任务→最终 recall **90.25%→49.82%**；Backdoor **0→3.43%**，不能都叫遗忘。CIC17 DER++ Normal **99.47%→84.36%**；CSE18 DER++ Infiltration **91.03%→51.43%**。 |
| G4 错误流向 | 对同一固定方法，汇总五 seed 错误计数后，UNSW/CIC17/CSE18 的“更新类”占全部旧类错误 **87.34/84.69/96.20%**；仅旧攻击类错误作分母则 **73.25/71.29/45.00%**。非逐seed比例均值；Normal、更新类、其他旧类/同任务互斥，不把此描述当额外Joint差距。 |
| G6 S1 | Normal：Joint93.76%、DER++74.33%，差距 **19.44pp**、五 seed 全正；仅 CIC17→18 一个源设置，不能满足两数据集门槛。PortScan无支持不是零表现；S1并非独立干净复现。 |
| G7 成本 | 120单元累计优化 **99035.11s**、评估 **419.22s**、选择 **648.47s**。记录的最大 CUDA allocated/reserved为 **72597504/96468992B**；最大回放payload/allocated均 **624080B**。这是训练阶段PyTorch记录和回放存储，不是设备总显存、整体峰值或总墙钟；近零DER++选择时间不代表无内存维护成本。 |

## 文献与候选门槛（仍在补齐）

定向核查三个不重复问题方向：**总体多类差距、局部逐类 recall 缺口、S1 Normal 跨环境缺口**；G3/G4用于解释而不另凑候选。目前只有 Macro-F1 满足同指标“两数据集、≥5pp、五 seed 同向”：UNSW五 seed差距 **11.97/8.89/9.06/12.18/7.00**，CIC17 **15.33/9.68/9.64/9.26/14.51**；平均缺口10.7521pp×2＝**21.5042**，这只是数量资格分数。

总体保持/多类性能已有至少三篇直接相关工作：[Two-Stage，GLOBECOM2024](https://doi.org/10.1109/GLOBECOM52923.2024.10901809)、[AdvAs-IIDS，ESWA2025](https://doi.org/10.1016/j.eswa.2025.126632)、[ExBCIL，online2024](https://doi.org/10.1007/s13042-024-02486-9)。**不能把宽泛总体差距称为“≤2篇”的稀缺空白**，也没有近期方法在本协议下已被测为无效。完整[定向一手文献表](results/problem_scout/section21/literature_gap_map_v1.json)记录10篇不同工作、定位、数据集/任务、数值与未核项；非穷尽系统综述。

| 问题方向 | 已核实改善幅度与协议差异 |
|---|---|
| G1 总体多类保持 | [Attack-adaptive IoT 2025作者全文](https://www.researchgate.net/publication/389901457_Attack-Adaptive_Network_Intrusion_Detection_Systems_for_IoT_Networks_through_Class_Incremental_Learning) Table7：IoT-23/TON-IoT/IoT-NID/Edge-IIoT Macro-F1提高 **8.69/2.83/18.36/5.60pp**；packet-CNN、逐攻击留出，内存/输入联合变化，非本协议或匹配预算因果对照。上述三篇的成对改善幅度尚未核实；ExBCIL绝对准确率不当作提升。 |
| G2 局部逐类缺口 | Two-Stage与[ACIL2026](https://doi.org/10.1007/s13042-025-02840-5)直接讨论长尾/少数类，但尚无可核验的本次具体类别改善幅度。框架不是新增已跑通方法；不同弱类不能拼接达门槛。 |
| G6 跨环境Normal | [SSF主文TableI](https://arxiv.org/html/2412.16264v1)二分类F1相对表内最强CIL对照提高 **4.25/3.66pp**（NSL/UNSW），非S1 Normal FPR；[DIL2026](https://doi.org/10.1016/j.asoc.2026.116022)假设Normal分布稳定。CND-IDS的倍数、OLIDA/DRIFT-CL绝对分数均不能改写为此端点改善。单源不能晋级两数据集候选。 |

**当前认证候选0项，非断言不存在研究问题。** 部分全文/对照改善幅度尚待补齐；未知不算零提升、无代码或无效。不凑三个候选，暂不作最终排序。

## 协议、数据与成本边界

六方法 joint/replay/icarl/replay_lwf/rcl_tab/derpp×seeds1/2/3/4/42；每任务8 epochs，全部训练与测试不截断、不补样。Joint每 checkpoint重训全部已见类；CL旧类只访问已存200/类回放。DER++为共享MLP/Adam/类平衡CE、逐类reservoir的显式适配，不是原论文协议原样复现；200/类是观察/logit槽位，八轮接纳可重复源样本，不保证200个独立样本。iCaRL精确200步float32 herding，mmap不改变采样。60个早期本地单元源ab33ca6、其余源1f35f65；12个科学Git文件相同，实际字节差异为换行，有效配置一致。78本地/42HPC单元的Python、CUDA及sklearn版本不同，不作硬件匹配速度比较。[可复现入口与配置](run_problem_scout.py)、[分析入口](analyze_problem_scout.py)。

| 数据集 | 全量训练 / ID测试 | 原始 / 模型输入跨划分重合 |
|---|---:|---:|
| NSL | 125973 / 22544 | 664 / 664，保留官方划分 |
| UNSW | 120991 / 30248 | 0 / 13，float32 Normal碰撞保留 |
| CIC17 | 2262203 / 565673 | 0 / 179 |
| CSE18 | 12882081 / 3255102 | 0 / 4030 |

UNSW从257673行剔除2445个冲突组全部40400行及同标签重复excess66034，再seed42划分。18数组已重新核实际文件/负载SHA、形状、全部有限值与类计数；533原始分片的既有审计保持。CIC训练重复excess264889/3780187、冲突成员6068/1259263按冻结协议保留，**非全局去重/无噪声**。S1目标3255102行中18183与源训练重合（0.559%，含269异标签）；映射不事后改。CIC17 Infiltration仅30/6，UNSW Worms127/32，无补样。历史第15节、第17/19节Exp H结论保留并限定为“在旧类完整池重访的旧回放协议下得到”，不能当本次新协议证据。

[全部20 job ID及提交记录](results/problem_scout/section21/hpc_jobs.json)：17完成、3重复取消、无运行；无拆Joint或截断。HPC累计分配 **85177s＝77967s选中＋2285s已完成重复＋4925s取消重复**，不是GPU忙时或日历墙钟；runner与Slurm不相加。CSE18四个完整HPC实测17217/16915/17374/16897s；本地CIC17 seeds42/4及CSE18 seed42创建→日志区间2818.57/2976.99/14817.44s，不是独立退出墙钟，退出码未知。初始本地CIC17中止启动的独立成本未知。连续GPU利用率缺失，不记0；不再轮询终态作业。[各次实测与日志](results/problem_scout/section21/support_v1/attempt_costs.json)。

[近期基线可用性](results/problem_scout/section21/baseline_availability.json)：75条筛选，2024–26新增NIDS按本协议跑通为0；ACIL框架/PAPA未激活、packet-CNN适配及KD梯度问题、二分类RPL/SSF、SPIDER仅README等限制保留，未确认不等于无代码。最佳只在已运行方法中成立，缺项不是败北。论文不改；后续只补21.4一手协议/改进幅度、完成21.5审慎判定与双仓核验，保留heartbeat至合格报告交付。
