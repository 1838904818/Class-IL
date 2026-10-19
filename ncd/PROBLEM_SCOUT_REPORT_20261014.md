# 类增量 NIDS 选题侦察｜剩余差距、步数对照与历史资格判定

## Section27 AA-CIL：第1天实现与先行工作核实（部分交付）

**2026-10-19 15:14 MYT；尚无AA-CIL正式效能结果。** 已按27.9实现独立检测门G和仅攻击类家族分类器F，以及必需的TS-mem、AA-noAttMem。G/F各用max(8epochs,2000步)、batch256，分别报告时间/步数及合计；不声称与单模型算力相同。AA-CIL的M全部分给攻击，G每任务从永久Normal留出池之外抽至多50,000条良性样本；TS-mem仅用Normal回放配额，AA-noAttMem不把Normal份额转给攻击。旧攻击训练/验证只保留内存，不回访原池。各方法共用永久Normal留出池校准α=1%/2%/5%，不优化、不计入M，校准/额外良性流另列条数与字节；校准α不保证测试FPR恰等于α。TS-mem的20条Normal验证与全校准池重叠，单独记录重叠，不能重复计算独立存储。

**27.3新颖性：未通过，精确否决条件亦未证实。** [原文核实表](LITERATURE_DIFFERENCE_TABLE.md#section-273-aa-cil-先行工作核实2026-10-19)记录Network-Agnostic CIL已经在旧攻击专用内存之外引入新域良性数据；不能声称“新良性数据+攻击内存”组合首次出现。该文并非独立G→攻击专用F，也未验证无限持续良性流的多任务方案。GLOBECOM 2024 Two-Stage全文不可得，其良性数据角色仍不可得；因此不把精确双条件未知写成新颖性已成立。TS-mem是本研究限定良性配额的消融，不是该文已核实的忠实复现。既有结果也不证明所有误报的根因均是良性覆盖不足、或所有方法全局等价。

**实现核验与排程。** 冻结源`414136b`的同入口CPU smoke完成3方法、12实际检查点；独立复现36个校准阈值及36个整数CM，旧攻击子集、良性留出排除、G/F成本与步数审计通过。该小样本检查仅验证实现，不是正式2000步效能或资源测量。NSL/UNSW新方法正式批次尚未启动：等待当前CIC批次正常结束后串行运行，不增加第二个GPU驱动。27.8五项仍待正式五seed结果，保留RQ1/RQ3和实质创新门槛。

**27.9已完成的基线校准。** 首批81个物理单元按授权仅用Normal留出计数向外扩τ、保守二分取可行端点，α=1%/2%/5%均有值；未使用测试标签选点。下表是完整五seed的α=2%测试均值±样本SD（%），不是26.10的测试FPR插值。Replay+LA使用同权重和校准后总τ，与Replay重合，不重复计费；FARQ仅附列，未进入七对照最强基线选择。TS-mem/AA-noAttMem尚缺，故不提前确定27.8最强基线或显著性结论。

| 方法（NSL/UNSW均5/5） | NSL测试FPR | NSL检出率 | NSL Macro-F1 | UNSW测试FPR | UNSW检出率 | UNSW Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Replay / Replay+LA | 4.080±2.162 | 65.386±5.301 | 46.829±4.930 | 1.784±0.201 | 38.521±3.773 | 26.266±0.855 |
| WACB | 3.623±0.871 | 66.829±3.773 | 57.029±3.910 | 1.931±0.239 | 48.970±4.140 | 32.807±1.176 |
| Freq | 4.928±2.423 | 69.667±2.534 | 56.843±4.322 | 2.009±0.264 | 50.419±2.932 | 31.224±1.729 |
| DER++ | 2.506±0.452 | 65.275±2.730 | 50.460±3.152 | 1.985±0.268 | 18.477±4.124 | 14.470±1.436 |
| FARQ（附列） | 3.680±0.989 | 67.448±2.773 | 58.860±5.788 | 1.838±0.184 | 45.099±4.200 | 30.247±1.676 |
| Joint（离线参考） | 6.786±1.002 | 66.633±1.795 | 59.371±2.898 | 1.928±0.152 | 67.109±1.453 | 37.219±1.000 |

NSL/UNSW校准FPR分别1.989607%/1.992674%，与表中测试FPR明显不同，不能称为同一实测误报率比较。每seed校准池NSL6,735条、UNSW6,825条、CIC181,176条；逻辑特征+标签字节分别3,178,920/1,911,000/57,251,616，索引另53,880/54,600/1,449,408字节，不计M，非新增序列化文件大小。NSL Replay的α2总τ=3.975–24.585，UNSW为7.795–10.047；扩展仅适用于此校准，不重写26.10的范围外“不可得”。[校准原始证据](results/problem_scout/section27/baseline_calibration_v2/summary.json)与[完整分析/各α/逐类/成本/存储](results/problem_scout/section27/analysis_v1/result.json)保留全部seed；354个实际训练轨迹PT/sidecar核验通过。新CPU校准79.789s与旧曲线89.136s为不同物理尝试，各自已含组件时间，不再累加子项。AA-CIL/TS-mem/AA-noAttMem正式结果仍未完成。

## Section26.9 FARQ：统一良性刷新规则下的误报与攻击检出权衡

**2026-10-19 15:02 MYT；26.9第十批部分结果。** NSL/UNSW全部方法及Joint、CIC Replay/LA/WACB/Freq/FARQ/FARQ-all五seed已齐。FARQ在UNSW/NSL的不伤害标准失败，在CIC又低于WACB：**预登记主判定不成立**，不因次要分析改判。所有部署回放方法每任务刷新良性训练池；旧攻击训练/验证样本只能取已存内存，永不回访旧池。旧口径 `local_v1` 已停止并排除，保留返工成本。

**动机与条件。** [六篇已发表论文核实表](LITERATURE_DIFFERENCE_TABLE.md#section-268-核实补充2026-10-19)显示类别平衡、频数、代表性采用预设准则，而非良性FPR反馈；预设不等于数值固定。五篇主结果未单列固定工作点FPR，仅限该样本，不代表全领域。共同M=200×已见类数，永久10%训练holdout不优化，常驻验证计入M。WACB适配CBRS的良性/合并攻击50:50；Freq先20/类再按原始训练频数最大余数分配，非ECBRS精确复现。FARQ仅改Replay配额，m是家族错分率；8epochs或≥2000步、batch256不变。LA为+log训练先验、τ=1。Joint同holdout但可访问所有已见训练数据，是离线例外，非匹配部署方法或保证的实测上界。

**表1：正式完整组均值±样本SD（%）。** 二元攻击检出率为攻击被判为任意攻击类的比例；护栏为测试支持≥20的攻击家族宏召回，排除类别仍计入MF1/BAcc及逐类表。未齐五seed不作效能判定。

| 数据集 / 方法（完成/5） | 良性FPR ↓ | 攻击检出率 ↑ | Macro-F1 | BAcc | Normal召回 | 攻击护栏召回 |
|---|---:|---:|---:|---:|---:|---:|
| CIC17 / Replay（5/5） | 13.15±2.16 | 99.19±0.19 | 52.70±2.44 | 92.53±0.29 | 86.85±2.16 | 97.79±0.17 |
| CIC17 / WACB（5/5） | 8.64±1.23 | 98.28±0.43 | 57.44±1.27 | 92.82±0.20 | 91.36±1.23 | 97.42±0.20 |
| CIC17 / Freq（5/5） | 8.27±0.65 | 97.00±0.91 | 55.81±1.44 | 89.60±2.36 | 91.73±0.65 | 95.28±0.60 |
| CIC17 / Replay+LA（5/5） | 4.39±1.09 | 93.30±2.36 | 56.84±1.91 | 72.91±7.89 | 95.61±1.09 | 74.06±8.58 |
| CIC17 / DER++（1/5） | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |
| CIC17 / FARQ（5/5） | 9.83±2.66 | 98.59±0.87 | 55.90±1.50 | 92.64±0.24 | 90.17±2.66 | 97.39±0.48 |
| CIC17 / FARQ-all（5/5） | 13.22±4.07 | 96.54±1.69 | 52.75±2.70 | 89.78±1.45 | 86.78±4.07 | 94.13±1.98 |
| CIC17 / Joint（离线参考，0/5） | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |
| UNSW / Replay（5/5） | 23.04±1.64 | 94.90±1.01 | 41.60±0.92 | 64.98±1.54 | 76.96±1.64 | 63.65±1.77 |
| UNSW / WACB（5/5） | 14.79±0.52 | 85.98±0.94 | 39.87±0.68 | 61.68±0.91 | 85.21±0.52 | 59.06±1.06 |
| UNSW / Freq（5/5） | 14.41±1.09 | 86.49±0.97 | 41.27±0.30 | 56.48±1.08 | 85.59±1.09 | 53.25±1.25 |
| UNSW / Replay+LA（5/5） | 17.56±1.87 | 89.31±1.37 | 45.51±1.21 | 56.27±2.50 | 82.44±1.87 | 53.36±2.90 |
| UNSW / DER++（5/5） | 25.18±0.76 | 96.88±0.65 | 43.14±0.45 | 67.85±0.69 | 74.82±0.76 | 67.08±0.77 |
| UNSW / FARQ（5/5） | 16.50±0.39 | 88.14±0.94 | 39.52±1.03 | 63.40±1.62 | 83.50±0.39 | 61.17±1.82 |
| UNSW / FARQ-all（5/5） | 23.71±2.57 | 95.06±2.08 | 36.39±2.75 | 59.90±0.98 | 76.29±2.57 | 58.08±1.14 |
| UNSW / Joint（离线参考，5/5） | 27.63±0.30 | 99.79±0.14 | 47.25±0.77 | 67.33±0.59 | 72.37±0.30 | 66.77±0.68 |
| NSL / Replay（5/5） | 7.23±2.62 | 81.28±4.19 | 59.02±1.92 | 67.57±1.69 | 92.77±2.62 | 61.27±2.18 |
| NSL / WACB（5/5） | 5.25±2.33 | 74.49±5.42 | 57.77±2.16 | 64.48±2.83 | 94.75±2.33 | 56.92±3.47 |
| NSL / Freq（5/5） | 5.69±2.40 | 73.65±3.78 | 55.72±2.66 | 64.66±2.40 | 94.31±2.40 | 57.25±3.40 |
| NSL / Replay+LA（5/5） | 6.35±2.72 | 79.12±4.45 | 60.03±2.95 | 65.67±1.89 | 93.65±2.72 | 58.67±2.42 |
| NSL / DER++（5/5） | 6.56±2.16 | 79.32±0.64 | 59.44±2.83 | 67.31±1.57 | 93.44±2.16 | 60.77±1.50 |
| NSL / FARQ（5/5） | 5.21±1.96 | 74.26±4.36 | 57.85±2.14 | 65.82±3.13 | 94.79±1.96 | 58.58±4.12 |
| NSL / FARQ-all（5/5） | 5.00±2.21 | 69.33±4.29 | 54.49±3.95 | 63.29±5.00 | 95.00±2.21 | 55.36±6.71 |
| NSL / Joint（离线参考，5/5） | 7.25±0.64 | 69.37±2.83 | 57.79±1.58 | 61.50±3.07 | 92.75±0.64 | 53.69±3.71 |

**表2：FARQ每任务Normal配额（均值±样本SD，%）。** 旧攻击损失不能补回：只匹配容量上限，非相同占用。NSL FARQ最终用670/572/764/952/924条，FARQ-all用238/148/182/1000/384条（上限1000）；UNSW对应1606/1771/1862/1703/1800及721/1252/994/1231/809条（上限2000），空槽限制消融解释。DER++槽位可含既有reservoir重复观测，非旧池补样或独立样本数。

| 数据集 / task | 完成/5 | 公式q | 实际/容量 | 实际/已用 |
|---|---:|---:|---:|---:|
| CIC17 / 0 | 5/5 | 50.00±0.00 | 50.00±0.00 | 50.00±0.00 |
| CIC17 / 1 | 5/5 | 39.00±19.17 | 39.00±19.17 | 39.00±19.17 |
| CIC17 / 2 | 5/5 | 35.76±15.79 | 35.75±15.78 | 37.70±15.82 |
| CIC17 / 3 | 5/5 | 49.02±13.01 | 49.01±13.01 | 53.97±9.52 |
| UNSW / 0 | 5/5 | 52.00±4.47 | 52.00±4.47 | 52.00±4.47 |
| UNSW / 1 | 5/5 | 57.43±4.33 | 57.43±4.33 | 57.43±4.33 |
| UNSW / 2 | 5/5 | 37.34±5.95 | 37.32±5.94 | 41.06±5.50 |
| UNSW / 3 | 5/5 | 34.48±4.51 | 34.44±4.51 | 37.24±3.10 |
| UNSW / 4 | 5/5 | 26.75±5.48 | 26.72±5.48 | 30.37±4.81 |
| NSL / 0 | 5/5 | 50.00±0.00 | 50.00±0.00 | 50.00±0.00 |
| NSL / 1 | 5/5 | 54.67±11.93 | 54.67±11.93 | 54.67±11.93 |
| NSL / 2 | 5/5 | 47.67±17.14 | 47.65±17.16 | 50.03±13.92 |
| NSL / 3 | 5/5 | 38.07±18.06 | 38.04±18.04 | 46.87±13.98 |

**26.5逐条：** ①CIC MF1增益+3.1990pp、护栏−0.4005pp满足各自门槛，配对FPR闭合待Joint，整条未完成；②CIC同时不劣WACB和Freq：**不满足**，MF1分别−1.5409/+0.0899pp；③UNSW不劣Replay超过1pp：**不满足，−2.0728pp**（WACB−1.7235pp）；④NSL同标准：**不满足，−1.1781pp**；⑤CIC严格优于FARQ-all：**满足，+3.1527pp**（配对SD2.1856pp，五seed均正）。UNSW FARQ五seed均下降>1pp，配对ΔFPR/检出/护栏=−6.5409/−6.7632/−2.4838pp；NSL为−2.0183/−7.0241/−2.6913pp。五条件须全满足，整体失败并非一般否定反馈。NSL闭合221.26%±995.05%受负/近零分母影响；UNSW分母全负，FARQ闭合−175.19%±107.71%不表示FPR恶化。Joint非实测FPR上界；保留全部有符号配对比值。

**CIC权衡与限制。** 相对Replay配对ΔFPR/检出/MF1/护栏：FARQ为−3.3262/−0.5962/+3.1990/−0.4005pp，WACB为−4.5136/−0.9076/+4.7400/−0.3687pp，Freq为−4.8845/−2.1855/+3.1092/−2.5011pp，LA为−8.7633/−5.8866/+4.1420/−23.7258pp。FARQ最终占用1539/1378/1539/1527/1187（上限1600）；Replay/WACB/Freq为1430/1516/1452。FARQ seed3各任务CM与Replay相同，最终训练后q升至0.6不能解释此前收益。Freq task2为3648步，Replay/WACB/FARQ为3656步，均遵守8epochs/≥2000规则，非相同实际步数。Infiltration测试6条保留MF1/BAcc/逐类，无预测precision保留null。

**证据/成本/状态。** [分析与逐类P/R、配额、有符号比值、成本](results/problem_scout/section26/analysis_v2/result.json)：101/105主表、10/15 Joint，余9输出未完成；419实际权重SHA通过。独立CM误差<1e−15，holdout/刷新/旧攻击子集及来源核验通过。96次训练9195.440s、选择5.722s、评估42.209s，不含加载，LA不重复计费；源`b835a93`。CIC FARQ-all最终占用360–603/1600，空槽混杂消融解释；DER++已完成seed1的Infiltration为180训练槽但仅25独立ID，非180独立样本。CIC剩余批次继续，不重跑完成单元。HPC488962低GPU采样0/5/8%触发保护器失败（wall137s、CPU105.468s、峰值15.96GiB/16GiB），0检查点；非管理员取消/已证实OOM，不重提。旧[口径](results/problem_scout/section26/analysis_v1/result.json)22次训练735.223s及1次未完成成本保留；旧HPC wall453s/CPU410.899s/峰值11.13GiB，其余4个未启动，遥测未知。任务/Slurm耗时不重复相加；首个有效完成择源，非按得分。RQ1/RQ3及实质创新未关闭。

## Section26.10：主判定失败后追加的描述性误报–检出曲线

**2026-10-19 14:53 MYT首批计算；15:03复核。** 75个主方法单元完成71个，CIC DER++仅1/5；另10个Joint仅供27.9离线校准参考，不进入本节候选比较。[完整曲线/逐seed插值/MF1/SD/不可得原因](results/problem_scout/section26/operating_curves_v1/summary.json)保留τ=−1…3的41点。每seed按相邻τ点线性插值，再聚合五seed；不外推、不筛seed。MF1也是端点指标插值，不是某个单一分类器的实测MF1。图线为同τ下的五seed均值，不能用它代替逐seed工作点插值：[CIC17](results/problem_scout/section26/operating_curves_v1/figures/cic_ids_2017_mean_tau_fpr_adr.png)、[UNSW](results/problem_scout/section26/operating_curves_v1/figures/unsw_nb15_mean_tau_fpr_adr.png)、[NSL](results/problem_scout/section26/operating_curves_v1/figures/nsl_kdd_mean_tau_fpr_adr.png)；各图内附三个工作点的检出率/MF1均值±SD。

| CIC17完整五seed方法 | 检出率@FPR2% | 检出率@FPR5% | 检出率@FPR10% |
|---|---:|---:|---:|
| Replay | 79.2876% | 95.1466% | 98.9675% |
| WACB | 89.3512% | 97.3198% | 98.3984% |
| Freq | 88.8414% | 96.5080% | 97.1480% |
| FARQ | 85.7812% | 97.1107% | 98.6255% |

**26.10(3)不满足，剩余曲线仍补齐。** FARQ−Replay配对检出率差在2/5/10%为+6.4936/+1.9641/−0.3420pp；WACB为+10.0636/+2.1732/−0.5691pp。两者均未满足CIC10%的同向改善，FARQ还未达5%的≥2pp，因此即使UNSW补齐也不能满足两数据集的合取条件。UNSW Replay在2%/5%五seed均无bracket，10%有两seed无bracket，三目标完整组均“不可得”；不将NA置零或改区间补结论。JSON中的`pending_or_unavailable`描述信息缺失，不能覆盖已失败的必要条件。局部收益不等于全局支配；不满足也不证明方法普遍等价或仅移动阈值。

**核验/成本。** 81个实际最终PT与来源哈希、3321个曲线整数CM及155个可用校准后测试CM独立复算，最大差1.11e−16；81个τ0及15个Replay τ1均精确复现原CM。CPU首批89.136s含数据/来源核验、前向、扫描和校准，不含训练；代表性UNSW Replay1校准前向独立复现全部41个FP计数。26.10使用测试FPR的事后描述性插值，绝不是部署校准保证；27.9另用共同Normal留出池定阈值再报实际测试FPR。用户批准的27.9校准扩展另存，不改变这里的41点、不可得项或原主/次判定。

## Section25 后续：攻击家族替代与同状态学习干扰（探索性，完整）

**2026-10-19。** 既有task末轨迹确认恶化发生于task2→3：Replay原正确R2L转U2R平均362.4条、转Normal111.8条；原正确WebAttack转Bruteforce12.33条、转Normal1条（各5/3 seeds）。这只能定位任务更新，不能恢复历史优化步。以下逐步时间是重新精确复现的训练路径，不是历史测试样本的逐步轨迹。

**匹配小对照。** 两数据集×MIR/额外稀有CE×seeds1/2，共8条路径，各2000真实步；起点、当前样本顺序、回放配额/选样、常驻记忆及Adam均保持原设定。每50步的更新前状态固定取40点，以相同参数、Adam矩及批次做full、rare-only、nonrare-only、zero-gradient、without-competitor五个一次Adam克隆，全部丢弃；另有不更新参照。保留原整批CE权重/分母，不对子集重新平衡。竞争类预定NSL R2L/CIC DoS；去除其真标签损失贡献，不删除logit。CIC nonrare含当前Botnet及旧类，并非仅旧类；zero-gradient仍可被Adam动量推动。

下表“失误步”是探针中最早一行从正确变错误，依次seed1/2；新/旧分别为U2R/R2L、Infiltration/WebAttack。ΔCE单位为10⁻³，正值更差；先取每seed全部40点均值，再给两seed均值±样本SD，事件不作独立重复。新类Δ=`full−rare-only`，旧类Δ=`full−nonrare-only`。

| 数据集 / 路径 | 新类首次失误步 | 旧类首次失误步 | 新类ΔCE | 旧类ΔCE |
|---|---:|---:|---:|---:|
| NSL / MIR | 214 / 212 | 79 / 24 | +1.172 ± 0.055 | +0.544 ± 0.064 |
| NSL / 额外CE | 233 / 194 | 36 / 22 | +0.407 ± 0.081 | +1.467 ± 0.205 |
| CIC / MIR | 1060 / 1663 | 1 / 47 | +4.876 ± 0.639 | −0.073 ± 0.157 |
| CIC / 额外CE | 无 / 1828 | 1 / 38 | +0.994 ± 1.683 | −0.043 ± 0.172 |

**替代去向。** NSL新U2R首次丢失均转R2L；旧R2L最早一行在3/4路径先转Normal，但各行首次失误的累计去向以U2R为主。四路径最终仍52/52、200/200正确，属于可恢复干扰。CIC MIR各一行Infiltration短暂转Botnet后恢复，最终均30/30；额外CE则29/30、28/30行从未正确，最终DoS22/28行、Normal7/0行，主要是未学会而非遗忘。其seed2仅一行首次于1828转Normal、1871恢复，随后还发生过翻转。旧WebAttack主要被Bruteforce替代，最终正确187/183（MIR）、186/183（CE），并非所有首次恢复都能保持至终点。

**判断。** NSL双向局部效应跨seed同向，新类效应在四个固定500步窗口均为正；但真实完整更新相对不更新仍平均降低新类CE，不能说每步恶化。CIC额外CE的新类效应跨seed反号；MIR各路径40点新类ΔCE绝对值之和的92–95%来自首点，不能当作持续效应量。`full−without-DoS`在两个臂均跨seed反号，不支持“预测为DoS，所以DoS梯度是稳定致因”；NSL R2L特定效应也有CE seed2均值为负的例外。主分析保留全部点及无rare行事件，不按符号、窗口或结果筛选。下一项有价值的假设是区分当前Botnet与旧类梯度、并检查Adam/整批权重尺度，而非继续排斥Normal或直接增大稀有CE；尚未追加该实验。

**复核/成本/边界。** [完整分析](results/problem_scout/rare_boundary/interference_analysis_v1/result.json)及[原始探针](results/problem_scout/rare_boundary/interference_v1/result.json)：8终点模型/完整Adam/记忆/CM/6曝光链精确复现；320预状态及8逐行NPZ实际哈希通过，full克隆对真实下一步的参数/Adam各320次精确一致。另独立重执行各路径pre0/500/1500共24点，1152标量比较最大差0。正式16000真实步、1600诊断Adam步；driver外层cell wall合计331.295s，其中不重叠组件292.632s（采用含诊断的优化区间，不重复相加其子项）、未分项38.663s；独立复核另120克隆步、6.421s。匹配数据/真实步/每条件一次Adam及探针量，不匹配损失项总权重、梯度尺度或FLOPs。仅两已暴露seed，训练/常驻探针、事后选定竞争类；路径间状态不同，不能作额外CE的同状态因果比较，单步效应不证明长期机制或测试改善。源`79991f2f`；`python -I ncd/analyze_family_interference.py --weights verify`复核。权重/逐行NPZ仅本地；RQ1/RQ3及创新仍未关闭。

## Section25 后续：稀有攻击检查点诊断与预算匹配小对照（探索性，完整）

**2026-10-19：64个历史检查点及16个对照均完成。** [历史诊断](results/problem_scout/rare_boundary/diagnosis_v1/result.json)重现64/64原CM。仅有task末检查点：R2L/WebAttack可定位到task2→3（零起算）；U2R/Botnet/Infiltration首次出现在最后task3，不能称为“后续遗忘”或定位历史优化步。D2的R2L判Normal平均1745.4→1551.6条，却有家族正确1033→694条；正确R2L→U2R的251.4条多于→Normal的103.6条，并非统一的Normal吞噬。

**固定对照。** 两数据集seeds1/2，从同一Replay task2分叉，只重做task3；四臂同记忆（800/1200）、当前样本顺序、逐步类别配额和2000次真实更新。MIR类内选回放；另两臂对当前稀有类（训练<200条）加CE或`softplus(zNormal−zTrue)`，系数1、不调参。四个random终点参数/CM均精确复现S2000。[MIR适配](https://github.com/optimass/Maximally_Interfered_Retrieval/blob/35eda78bcdd35025b16b0b1039a926d34aad4851/mir.py)使用Adam虚拟步/类配额/类平衡CE，非逐项复现或创新。

下表均为两seed均值（%）；MF1附样本SD。FPR越低越好；稀有类为NSL U2R、CIC Infiltration。

| 数据集 / 臂 | Macro-F1 ± SD | Normal FPR | 攻击家族宏召回 | 稀有类 P / R |
|---|---:|---:|---:|---:|
| NSL / random | 58.14 ± 2.63 | 8.89 | 60.05 | 5.73 / 55.22 |
| NSL / MIR | 59.44 ± 3.61 | 8.21 | 60.49 | 5.12 / 52.24 |
| NSL / MIR+稀有CE | 58.20 ± 4.31 | 8.73 | 58.81 | 4.55 / 52.24 |
| NSL / MIR+边界项 | 58.27 ± 5.56 | 7.17 | 60.00 | 5.40 / 58.21 |
| CIC / random | 52.82 ± 0.28 | 14.29 | 93.34 | 0.059 / 66.67 |
| CIC / MIR | 54.11 ± 0.31 | 11.30 | 93.12 | 0.096 / 66.67 |
| CIC / MIR+稀有CE | 52.39 ± 0.72 | 10.97 | 80.92 | 0.000 / 0.00 |
| CIC / MIR+边界项 | 53.56 ± 0.57 | 11.78 | 88.13 | 0.041 / 33.33 |

**判断。** MIR的MF1提高1.30/1.29pp，但U2R召回下降2.99pp、Infiltration不变。NSL边界项相对MIR提高U2R召回5.97pp，判Normal却由7增至8条，旧攻击宏召回下降2.64pp。CIC Infiltration正确数：MIR[4/6,4/6]、CE[0/6,0/6]、边界[4/6,0/6]；新增错误流向其他攻击。低precision及不稳定seed均保留。

**轨迹。** 每50步只看训练探针，非历史恢复。NSL八单元终点均正确识别52/52训练U2R，测试44.78–61.19%；CIC额外CE训练Infiltration仅1/30、2/30，边界项28/30、0/30，不能统一归因于泛化。下一问题应是“真实家族与竞争攻击的学习/干扰”，非继续排斥Normal；本轮不追加实验。

**成本/限定。** 四臂单cell组件秒：NSL13.75/27.37/28.12/28.55，CIC14.60/27.59/28.27/29.40，总395.29s（加载/部分序列化另计）。匹配真实更新与回放，不匹配FLOPs/时间/工作显存或梯度尺度。两已暴露seed共享U2R67/Infiltration6条测试记录，非独立确认；Replay分叉不因果解释D2历史。SHAP缺依赖未执行；RQ1/RQ3及创新不关闭。

**复现。** [独立分析及完整命令](results/problem_scout/rare_boundary/analysis_v1/result.json)含逐类P/R/F1、BAcc、新旧召回、二元/家族错误、配对差、SD、曝光链与成本；`--weights verify`核验16终点+4起点实际权重。训练/诊断/分析源分别`fafcdd9f/b9227b1f/fe5fdf15`。冒烟非科学证据；权重、逐行NPZ不发布。

## Section25 良性误报快速诊断（完整结果）

**2026-10-19（MYT）结果交付，未在18日按期完成。** D1–D5及补充Joint+D4均齐。CIC17 D1采用先完成的HPC三seed（488759）；D2逐seed采用本地1（03:05:16）、HPC2/3（03:14:59/03:20:52，488784），均验证后按完成时间选源，不按分数。

表中依次为 **Normal recall / Macro-F1 / 攻击类宏平均recall / 闭合率**，单位%。闭合率为逐seed `(处理−Replay)/(Joint−Replay)` 后平均，非均值之比；未完成不作零分。

| 处理 | CIC17（3 seeds） | NSL（5 seeds） |
|---|---:|---:|
| Replay S2000参考 | 85.49 / 52.23 / 93.27 / 0 | 91.62 / 59.90 / 62.98 / 0 |
| Joint原协议参考 | 95.43 / 64.16 / 89.84 / 100 | 94.62 / 57.22 / 53.17 / 100 |
| D1：Normal 2000，攻击200/类 | 92.69 / 58.65 / 89.65 / 70.99 | 96.83 / 58.50 / 56.39 / −969.20 |
| D2：固定最终容量，Normal配额50% | 92.26 / 58.15 / 87.63 / 67.77 | 95.39 / 59.57 / 61.49 / −733.62 |
| D3：Normal k-means选200 | 85.09 / 49.95 / 90.29 / −5.06 | 90.79 / 59.44 / 61.06 / 774.31 |
| D4：测试logit加log训练先验 | 94.63 / 58.02 / 79.24 / 91.00 | 93.52 / 60.57 / 60.03 / −385.43 |
| D5：训练支持域kNN门控 | 89.91 / 59.86 / 89.84 / 44.08 | 95.09 / 54.05 / 49.71 / −734.71 |
| Joint+D4（补充） | 98.83 / 67.12 / 66.48 / 134.91 | 95.65 / 51.48 / 39.61 / −68.77 |

**25.4：无合格阳性。** H-cov须同一D2/D3在两数据集闭合≥50%、ΔMF1≥0且Δ攻击宏召回≥−2pp。D2的ΔNormal/ΔMF1/Δ攻击召回为CIC17 +6.77/+5.92/−5.63pp（攻击护栏失败）、NSL +3.78/−0.33/−1.49pp（MF1失败）；D3亦失败。因此D2/D3均不满足跨数据集H-cov，不否定一般覆盖假设。D1仅增预算参考，两组均失败；D4/D5四组攻击降幅均超2pp，不支持H-prior/H-supp。保留CIC17 D2 Infiltration [4/6,1/6,0/6]，其下降算术贡献攻击宏召回−5.56pp，非机制证据；D1/D3零召回seed亦保留。RQ1/RQ3及创新未关闭。

**限定。** NSL seeds1/42分母−0.525/−0.072pp，保留全部signed比率；D2五seed的Normal均改善而闭合为负，D3正闭合不代表改善。D2容量上限1000/1600，Normal首次500/800；早期高于Replay，不回访旧池、不补稀有类空额（NSL/CIC17最终实际927/1516条）。D4加号是适配，非[Menon Eq.9](https://arxiv.org/html/2007.07314)复现。D5：memory的128维嵌入、平均10NN距离、类内留一95%阈值，严格低于才保留攻击。

**成本（训练/选择/评估秒）。** D1 NSL 1099.96/0.054/3.33、CIC17 922.47/0.064/3.83；D2 NSL 397.67/0.037/0.817、CIC17 1263.49/0.411/4.885；D3 NSL 392.41/5.434/0.730、CIC17 1857.18/97.638/8.010，聚类5.390/97.400s各只计一次。另计D1重复4542.94/0.093/35.10s、D2重复1518.58/0.315/6.565s。D1/D2 HPC墙时1034/1057s、MaxRSS9.96/11.96GiB，均退出0。加载另计，混合硬件/共享GPU不作独占加速比较。D4/D5模型25.00s、校正0.62/11.51s。

**复现。** [分析器](analyze_benign_diagnostics.py)复算64组CM（16参考+48处理），`--weights verify`核验96个所选权重；[完整命令/结果](results/problem_scout/section25/analysis_v1/result.json)含BAcc、逐类P/R/F1、新旧召回、SD、配对差及来源。HPC D2优先目录仅含先完成的seeds2/3，seed1用本地；未采用结果保留，另24个重复权重亦核验。D4/D5复现16模型原CM，无重训；HPC结果脱敏5个路径字段，权重不入库。D3固定k200/随机种42/n_init1/max_iter100；两数据集各选200条Normal，均达100次上限；缓存绑定源码、训练数据和sklearn。

**最近先例（5篇）。** [GSS，NeurIPS2019](https://proceedings.neurips.cc/paper/2019/hash/e562cd9c0768d5464b64cf61da7fc6bb-Abstract.html)：梯度多样性；[Rainbow Memory，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Bang_Rainbow_Memory_Continual_Learning_With_a_Memory_of_Diverse_Samples_CVPR_2021_paper.html)：类内覆盖；[CBRS，ICML2020补充B](https://proceedings.mlr.press/v119/chrysakis20a/chrysakis20a-supp.pdf)：类别重要性配额；[ECBRS，NeurIPS2023](https://papers.neurips.cc/paper_files/paper/2023/hash/3755a02b1035fbadd5f93a022170e46f-Abstract-Conference.html)：NIDS多数类替换，二元增量协议不同；[Ariffin等，arXiv:2608.04602v1](https://arxiv.org/html/2608.04602)：2026-08-05预印本，benign持续回放/按类量保留，未核实正式发表。Normal增额或多样性选样本身不新颖。

执行状态：本地D1/D3及D2、HPC D1/D2均完成；不再轮询或重跑。D3 HPC副本未提交，负责人于19日明确取消重复运行，不将取消记为实验完成。同入口冒烟非科学结果。本地覆盖及D2训练源`5657b468`，CIC17 D1 HPC源`5a5a8828`，D4/D5源`efd1ec04`，D2 HPC包装源`5622e4b`。本节结项不关闭RQ1/RQ3或总体研究。

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
