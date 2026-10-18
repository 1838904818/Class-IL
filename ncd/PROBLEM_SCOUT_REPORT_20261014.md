# 类增量 NIDS 选题侦察｜剩余差距、步数对照与历史资格判定

## Section25 良性误报快速诊断（首批结果，尚未完成）

**首批交付：2026-10-19（MYT），并非18日按期完成。** D4/D5已完成：CIC17 seeds1/2/3、NSL seeds1/2/3/4/42；16个原始保存模型的完整测试混淆矩阵精确复现，无重训或调参。D1/D3本地批次已启动（D1先行）；D2预算解释待确定。CIC17的HPC包装/归档均通过同入口真实冒烟，但部署前连接超时，尚无作业ID；未完成不因准备成功而改为完成。

表中依次为 **Normal recall / Macro-F1 / 攻击类宏平均recall / 闭合率**，单位%。闭合率为逐seed `(处理−Replay)/(Joint−Replay)` 后平均，非均值之比；未完成不作零分。

| 处理 | CIC17（3 seeds） | NSL（5 seeds） |
|---|---:|---:|
| Replay S2000参考 | 85.49 / 52.23 / 93.27 / 0 | 91.62 / 59.90 / 62.98 / 0 |
| Joint原协议参考 | 95.43 / 64.16 / 89.84 / 100 | 94.62 / 57.22 / 53.17 / 100 |
| D1：Normal 2000，攻击200/类 | 未完成 | 未完成 |
| D2：Normal占总预算50% | 未完成 | 未完成 |
| D3：Normal k-means选200 | 未完成 | 未完成 |
| D4：测试logit加log训练先验 | 94.63 / 58.02 / 79.24 / 91.00 | 93.52 / 60.57 / 60.03 / −385.43 |
| D5：训练支持域kNN门控 | 89.91 / 59.86 / 89.84 / 44.08 | 95.09 / 54.05 / 49.71 / −734.71 |
| Joint+D4（补充） | 98.83 / 67.12 / 66.48 / 134.91 | 95.65 / 51.48 / 39.61 / −68.77 |

**25.4判定：当前没有合格阳性结果。** D4/D5的攻击召回下降：CIC17 14.03/3.43pp，NSL 2.95/13.27pp，均超2pp护栏；NSL D5的MF1另降5.85pp。因此不支持H-prior/H-supp；H-cov待D2/D3。Joint D4的Normal增益为+3.40/+1.03pp，小于Replay的+9.14/+1.90pp，且CIC17 Joint攻击召回下降23.36pp。

**限定与成本。** NSL seeds1/42分母为−0.525/−0.072pp，比率不稳定；不删seed。D4加号为指定适配，非[Menon ICLR2021](https://arxiv.org/html/2007.07314) Eq.9测试减号复现。D5用保存memory的最终128维嵌入、平均10NN欧氏距离、类内留一95%分位；严格低于阈值保留攻击，否则Normal，测试集不定阈值。CPU模型单元合计25.00s（不含数据加载/共享哈希），D4额外0.62s、D5额外11.51s；训练成本待齐。诊断不证明因果或新机制，不关闭RQ1/RQ3。

**复现。** [D4/D5入口](run_benign_posthoc.py)，源码`efd1ec04d7228cb70bed263eb74538fdedbcb302`；运行`python -I ncd/run_benign_posthoc.py --output-dir ncd/results/problem_scout/section25/posthoc_v1`并显式设置同值`NCD_SOURCE_COMMIT`，原Joint权重另传`--joint-weights-root`。[原始结果](results/problem_scout/section25/posthoc_v1/result.json)保存实际命令/权重SHA/日志；[独立分析](analyze_benign_diagnostics.py)复算40组CM、逐类P/R/F1和[配对变化](results/problem_scout/section25/analysis_v1/result.json)。D1/D3固定训练源码`5657b468d0c83af58ec60dcd60ecf83bf61594f5`，S2000不变；D3固定k200/seed42/n_init1/max_iter100，训练空间选实际行并共用缓存，成本单列。

**最近先例（5篇）。** [GSS，NeurIPS2019](https://proceedings.neurips.cc/paper/2019/hash/e562cd9c0768d5464b64cf61da7fc6bb-Abstract.html)：梯度多样性；[Rainbow Memory，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Bang_Rainbow_Memory_Continual_Learning_With_a_Memory_of_Diverse_Samples_CVPR_2021_paper.html)：类内覆盖；[CBRS，ICML2020补充B](https://proceedings.mlr.press/v119/chrysakis20a/chrysakis20a-supp.pdf)：类别重要性配额；[ECBRS，NeurIPS2023](https://papers.neurips.cc/paper_files/paper/2023/hash/3755a02b1035fbadd5f93a022170e46f-Abstract-Conference.html)：NIDS多数类替换，二元增量协议不同；[Ariffin等，arXiv:2608.04602v1](https://arxiv.org/html/2608.04602)：2026-08-05预印本，benign持续回放/按类量保留，未核实正式发表。Normal增额或多样性选样本身不新颖。

下一步：完成D1/D3、确定D2预算口径并执行；保留未完成状态，按同一公式更新本表，不因等待而推迟首批交付。

## Section23 剩余差距与现成偏差校正（完整结果）

2026-10-18：**84个开发单元、24个冻结选择及120个正式单元全部完成；23.4判定为“差距未解决”。** UNSW与CIC17各自在全部校正/base中的最高闭合率仅为4.92%和19.23%，均不足30%。这是本协议下现成方法适配的负结果，不是“偏差校正普遍无效”或新机制的证据，也不关闭RQ1/RQ3或总体创新门槛。

**23.2已有结果拆解。** 下表固定使用Section22各数据集五seed平均Macro-F1最高的方法，不逐seed换方法。Joint仍为原协议参考，不是保证成立的上界。完整逐类precision/recall/F1差、带符号的F1贡献、FP来源、G4和NormalFPR见[拆解结果](results/problem_scout/section23/decomposition_v1/analysis.json)及同目录CSV；这些是算术分解，不是因果机制证明。

| 数据集 / 固定S2000方法 | Joint−CIL Macro-F1，pp，均值±SD | 主要正贡献，pp | benign FPR差 CIL−Joint，pp |
|---|---:|---|---:|
| NSL / Replay+LwF | −3.457±2.104 | U2R +1.258，被其他类负贡献抵消 | +2.910 |
| UNSW / RCL-tab | +5.138±1.555 | Exploits 1.304；DoS 0.982；Shellcode 0.751 | −1.740 |
| CIC17 / DER++ | +8.975±3.419 | Bruteforce 3.666；WebAttack 2.137 | +6.178 |

每类贡献为`100*(F1_Joint−F1_CIL)/类别数`；15个配对seed的贡献加和均还原总差距。CIC17上述两类合计5.803pp，占净差距64.66%；Bruteforce precision由72.16%降至37.69%，WebAttack由18.03%降至7.08%，而CIL recall约99.4%和93.8%。五seed合并错误计数中，Bruteforce、WebAttack、Botnet收到的FP分别有96.94%、97.15%、99.75%来自Normal。UNSW不同：DoS的FP有74.47%来自Exploits，Shellcode的FP来自Normal/Fuzzers/Exploits/Recon等；Exploits/Fuzzers/Recon的recall也下降，不能概括为纯precision问题。

G4按真实类引入时间定义“更新类”，Normal优先、其后更新类、其余旧/同任务类，三者互斥。全体旧类错误流向更新类的合并计数比例，Joint→CIL：UNSW 90.53%→87.86%，CIC17 76.34%→83.61%；只含旧攻击类时分别为78.45%→77.94%、37.84%→34.98%。旧Normal是重要来源，不能混淆这两个人口；合并计数比率也不等于逐seed比例均值，后者单列在结果中。

**原文与适配。** 已核对[WA](https://arxiv.org/html/1911.07053) Eq.6–9及bias消融、[BiC](https://arxiv.org/html/1905.13260) Sec.5、[SS-IL](https://arxiv.org/html/2003.13947) Eq.3/7与Sec.5、[EEIL平衡微调](https://arxiv.org/html/1807.09536) Sec.4–5。WA用新旧平均L2范数比同时缩放新类weight/bias；BiC用未进入特征训练的平衡留出样本拟合新类alpha/beta；SS-IL分离新旧CE并加旧任务级KL；CBF在当前类与已存旧类的等量子集上小学习率微调，蒸馏包含新任务。保留MLP/Adam/共享类别平衡CE，不复现图像论文整体协议、不宣称新机制。详见[实际运行配置](configs/bias_correction_section23.json)的adaptations：BiC留出与训练memory合计≤200/类，持久留出从类首次出现即隔离；SS-IL明确保留base辅助损失并追加旧回放，单类task的分离CE/KL恒为0；CBF随机选样替代herding，额外更新单独计费。

**运行设计。** 四种校正同时叠加Replay与本表固定最佳base，共120个正式单元。基础任务保持`max(8 epochs,2000 steps)`、batch256；不重跑Joint/CSE18、不动21/22原始结果。WA无待选超参；BiC校正100/300步、SS-IL任务KD权重0.5/1、CBF微调10/30epochs，其他值固定。84个development单元仅seed5/6，按分层训练留出10%的最终Macro-F1均值选择，平分选低编号；官方ID test不参与选择。选定后冻结，各组合以1/2/3/4/42报告；正式训练恢复完整原训练池（BiC内部留出除外）。候选、阶段、数据角色、源码提交和额外成本写入结果；冒烟不是正式证据。

**执行与恢复。** [总控入口](run_bias_control.py)命令为`python -I -u ncd/run_bias_control.py --output-dir ncd/results/problem_scout/section23/controls_v1`，顺序完成84个开发单元、冻结选择后才执行120个正式单元。训练源码、实际配置与历史lambda文件均绑定SHA；每任务checkpoint包含已完成校正和memory状态，完成单元经hash核验后跳过。此前24个真实数据冒烟中，一次Windows状态文件替换失败保留原日志并已增加有界重试；恢复没有重复训练。补丁后的额外1个真实数据冒烟和108项相关测试通过。冒烟及测试不提供校正有效性证据，不计入下述科学结果与成本。

完整矩阵于2026-10-17 23:32:10 MYT在本地RTX4060Ti启动，2026-10-18 20:14:48 MYT完成；204个科学单元均为首次尝试、退出码0，时间区间不重叠，无重复单元。训练源提交固定为`b19de59991ea744305f710ad4e6ca3ff9b860c93`，通过`NCD_SOURCE_COMMIT`传递给每个单元；11个训练源文件及4个配置/输入文件均与该提交的Git字节一致。实际配置SHA为`7ba1972b6624597a5578933d1ccf778787c33844a4f315ac5a59eff5d6cc401f`，选择文件SHA为`b508a41a90b1ff71214127aa4d5905775b920ccb2cee37355defb2ba49a93f72`。24个选择在18日08:50:36.471510 MYT冻结，早于首个正式单元启动；84个开发分数、选择证明和候选均值均复算一致。报告/分析代码更新不改变本次训练来源。

**可复算分析。** [分析入口](analyze_bias_controls.py)从混淆计数重算各checkpoint的Macro-F1/BAcc、逐类precision/recall/F1、新旧类平均recall和Normal FPR，分别报告相对同base S2000与Joint的配对变化。它重新核算84个开发单元的候选均值与24个冻结选择，并要求120个正式单元完整才给23.4判定；缺失为null，不作零分。命令：`python -I ncd/analyze_bias_controls.py --baseline ncd/results/problem_scout/section21/merged_v1 --steps-2000 ncd/results/problem_scout/section22/steps_2000_v1 --controls ncd/results/problem_scout/section23/controls_v1 --output-dir ncd/results/problem_scout/section23/analysis_v1 --weights portable`。portable模式验证小结果、sidecar和选择文件的实际字节，不读取权重；本地使用同一入口的`--weights verify`另核验权重。独立计算未导入该分析器的数值函数：正式120单元/520检查点、S2000 75单元/325检查点、Joint 15单元/65检查点的指标、逐类值、配对差及均值/样本SD均一致，分析数值最大误差1.42e-14pp。开发84单元的最终Macro-F1亦独立复算一致。正式数据绑定与原S2000一致；这不等于重新读取并哈希全部数据载荷。

本地最终分析以`--weights verify`完成，884/884个实际checkpoint载荷SHA256通过，状态为`actual_payload_sha256_checked`；公开包不包含这些权重，读者使用portable模式只能复核小结果与数值，不能据此声称重新验证了权重载荷。新增成本聚合及已有分析器共22项测试通过，成本阶段合计与独立日志加总一致。

**全部正式配置。** 下表为最后任务、五seed均值±样本SD；MF1/BAcc单位%，配对ΔMF1单位pp，比较对象为同一个未校正S2000 base。该Δ不是23.4闭合率的分子（除非该base本身就是固定最好）。不逐seed换方法，不以正式seed重新调参。

| 数据集 | base | 校正 | Macro-F1 | BAcc | ΔMF1 vs 同base |
|---|---|---|---:|---:|---:|
| NSL | Replay | WA | 60.027±2.672 | 68.939±4.040 | +0.132±1.092 |
| NSL | Replay | BiC | 60.393±1.057 | 69.094±1.547 | +0.498±2.059 |
| NSL | Replay | SS-IL | 56.750±1.362 | 55.401±1.437 | −3.145±2.547 |
| NSL | Replay | CBF | 61.075±2.754 | 69.945±3.945 | +1.180±1.646 |
| NSL | LwF | WA | 59.245±2.119 | 70.424±1.665 | −1.436±1.321 |
| NSL | LwF | BiC | 59.891±2.750 | 69.611±2.281 | −0.790±2.485 |
| NSL | LwF | SS-IL | 40.854±2.550 | 41.373±1.943 | −19.826±2.301 |
| NSL | LwF | CBF | 60.399±2.197 | 70.317±3.734 | −0.281±0.975 |
| UNSW | Replay | WA | 42.386±0.691 | 64.935±1.698 | +0.701±1.031 |
| UNSW | Replay | BiC | 41.914±0.577 | 65.397±1.777 | +0.229±1.035 |
| UNSW | Replay | SS-IL | 39.914±0.674 | 54.735±2.284 | −1.770±0.608 |
| UNSW | Replay | CBF | 41.926±0.416 | 67.468±1.097 | +0.241±0.521 |
| UNSW | RCL-tab | WA | 34.042±2.437 | 58.517±2.423 | −9.639±1.830 |
| UNSW | RCL-tab | BiC | 43.211±0.276 | 68.367±0.359 | −0.470±0.621 |
| UNSW | RCL-tab | SS-IL | 41.817±1.287 | 56.077±2.177 | −1.863±1.197 |
| UNSW | RCL-tab | CBF | 43.934±0.530 | 69.185±0.569 | +0.253±0.259 |
| CIC17 | Replay | WA | 39.001±1.994 | 64.561±6.065 | −13.717±1.314 |
| CIC17 | Replay | BiC | 53.043±3.128 | 90.717±1.522 | +0.325±3.452 |
| CIC17 | Replay | SS-IL | 53.422±1.432 | 72.571±0.211 | +0.703±0.512 |
| CIC17 | Replay | CBF | 52.368±2.116 | 87.975±4.119 | −0.350±1.320 |
| CIC17 | DER++ | WA | 54.453±1.834 | 92.930±0.327 | −0.652±0.785 |
| CIC17 | DER++ | BiC | 54.245±2.236 | 90.564±2.285 | −0.860±2.481 |
| CIC17 | DER++ | SS-IL | 56.830±1.886 | 88.574±1.140 | +1.726±1.208 |
| CIC17 | DER++ | CBF | 54.456±1.992 | 92.670±1.106 | −0.649±0.533 |

**23.4判定：未解决。** 按`(校正后Macro-F1−原固定最佳Macro-F1)/(Joint−原固定最佳Macro-F1)`计算闭合率；分子、分母均为五seed配对差的均值，判定不是逐seed比率的平均。不截断负值或超过100%的值；NSL分母为−3.457pp，闭合率不适用。达到70%须同一种校正、同一base角色在UNSW和CIC17都满足，即都叠加Replay或都叠加各库固定最好。下表保留全部组合，单位%。

| base角色 | 校正 | UNSW闭合率 | CIC17闭合率 |
|---|---|---:|---:|
| Replay | WA | −25.20 | −179.42 |
| Replay | BiC | −34.38 | −22.97 |
| Replay | SS-IL | −73.30 | −18.75 |
| Replay | CBF | −34.15 | −30.49 |
| 固定最好 | WA | −187.59 | −7.27 |
| 固定最好 | BiC | −9.14 | −9.58 |
| 固定最好 | SS-IL | −36.26 | 19.23 |
| 固定最好 | CBF | 4.92 | −7.23 |

没有同校正/同角色组合达到双库70%；即使只作补充敏感性、允许跨库选不同base也未达到。每库在全部校正/base中的最高值仍均低于30%，满足“未解决”，不需要跨库拼接两个不同的最优校正来作结论。UNSW最佳RCL+CBF仍距Joint 4.885pp；CIC17最佳DER+++SS-IL仍距Joint 7.250pp。这里的“最佳”只是完整五seed配置均值的描述性上限，不是新增超参选择或独立确认实验。

**新旧类与安全代价。** 最后任务的新类指该任务首次引入的类，旧类指此前所有类（含Normal）；均为类均衡平均recall，不是样本加权攻击检测率。以下是两库最高闭合配置的五seed均值及相对同base配对变化；完整SD与全部24配置在链接的指标表中。

| 数据集 / 配置 | 新类recall % / Δpp | 旧类recall % / Δpp | Normal FPR % / Δpp |
|---|---:|---:|---:|
| UNSW / RCL+CBF | 74.558 / +0.656 | 67.842 / +0.016 | 25.421 / −0.224 |
| CIC17 / DER+++SS-IL | 63.000 / −17.000 | 97.099 / +0.597 | 7.893 / −2.722 |

CIC17的MF1改善并非无代价：DER+++SS-IL的BAcc下降3.802pp，新类recall由80%降至63%；Infiltration由60%降至30%，且测试仅6条，Botnet由100%降至96%。Replay+SS-IL虽提高MF1 0.703pp，最后新类recall却由83.171%降至0.108%（Botnet 0.216%、Infiltration 0%）。部分seed没有Botnet预测，precision保留未定义及有效n，不能填0后称完整五seed均值。NSL LwF+SS-IL的MF1下降19.826pp，Probe/R2L recall分别下降59.612/27.161pp，U2R为0%。UNSW RCL+CBF的Exploits/DoS F1增加1.301/0.278pp，但Shellcode F1下降0.412pp。不能只报平均MF1增益而隐藏小类退化，也不能把低recall本身等同于遗忘机制。

**完整指标与原始证据。** [控制器结果](results/problem_scout/section23/controls_v1/result.json)、[冻结选择](results/problem_scout/section23/controls_v1/selection.json)和[每次进程日志](results/problem_scout/section23/controls_v1/driver_attempts.jsonl)索引所有204单元及884检查点的小结果/混淆矩阵/sidecar/日志；数据与权重不随Git发布。[完整分析JSON](results/problem_scout/section23/analysis_v1/analysis.json)、[配置及闭合率](results/problem_scout/section23/analysis_v1/configurations.csv)、[所有检查点指标均值/SD/配对变化](results/problem_scout/section23/analysis_v1/checkpoint_metrics_summary.csv)、[逐seed指标](results/problem_scout/section23/analysis_v1/checkpoint_metrics_by_seed.csv)、[逐类precision/recall/F1及计数汇总](results/problem_scout/section23/analysis_v1/checkpoint_classes_summary.csv)、[逐类逐seed及配对变化](results/problem_scout/section23/analysis_v1/checkpoint_classes_by_seed.csv)可在无数据/权重的环境复算。CSV率值为0–1，差值为pp；计数差单位为样本，不混用。

**成本。** 整个控制器墙钟74,557.979s（20.711h），含开发和正式阶段；顺序子进程经过时间合计74,529.076s。下表单位秒，计时来自实际日志与checkpoint，不是估计GPU小时。[成本明细](results/problem_scout/section23/analysis_v1/costs.csv)同时按阶段/数据集/base/校正拆分。

| 阶段 | 单元 / 检查点 | 子进程经过时间 | 基础训练计时 | 额外校正计时 | 评估计时 | buffer选择计时 |
|---|---:|---:|---:|---:|---:|---:|
| 开发5/6 | 84 / 364 | 33,488.903 | 32,597.507 | 140.356 | 22.395 | 0.445 |
| 正式1/2/3/4/42 | 120 / 520 | 41,040.173 | 40,175.339 | 111.854 | 57.971 | 0.539 |

开发/正式基础optimizer更新分别2,376,944/3,701,440步，独立后处理校正更新22,240/23,900步。SS-IL额外旧样本呈现11,700,224/15,109,120次已包含在基础训练计时，不是零开销，也不能再加一遍耗时。基础流examples_presented不完整覆盖普通DER++辅助回放；相同基础步数不代表相同FLOPs或总曝光。组件计时不含导入、缓存/哈希、序列化及进程设置等全部开销，不能替代子进程经过时间。以上不含此前冒烟、Section21/22、数据准备和随后分析。正式运行最大记录GPU allocated约68.98MiB、reserved 70MiB，但采样在训练/校正后、buffer选择/评估前，不是整进程峰值；未测主机RSS、能耗或整机利用率。不将原Joint的不同硬件/训练预算当作等成本对照。

**结论边界与下一步。** 只否定“这四种已声明适配、候选范围和基座足以填补本次剩余MF1差距”。BiC隔离留出改变训练曝光；SS-IL单类任务分离CE/KL为0且保留base辅助损失；WA缩放bias；CBF随机选样，均与图像原论文完整方案有差异。五个正式seed不是新数据源上的独立外部验证；没有显著性检验或机制因果证明。保留全部负结果、选择规则和成本，交负责人据残余误差及安全代价决定后续问题，不自动设计或启动新机制；锁定RQ1/RQ3及整体研究仍未完成。

---

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
