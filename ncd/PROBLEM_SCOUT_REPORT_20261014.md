# 类增量 NIDS 选题侦察｜资格判定与文献未核项

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

**下一步/阻塞。** Two-Stage、AdvAs-IIDS、ExBCIL经出版社、作者平台及机构入口有界检索仍无所需全文表；未知不记零或无效。需要合法可读全文/实验页后补21.4；停止重复同一入口检索，保留heartbeat至合格交付，10月16日仍缺则如实报告。
