# 类增量 NIDS 选题侦察 第 1 天

2026-10-14 第 21 节阶段报告。DGC 否决保持；只量化问题，不提出机制。四套全量输入已验证，训练/测试不截断、不补样。正式矩阵在运行，尚未完成“各数据集实际开跑”的第 1 天要求。

## 跑了什么

执行源 `ab33ca61747b0e499c4006b02e4eeda29589d7f3`。命令：`python -I -u ncd/run_problem_scout.py --config ncd/configs/problem_scout_section21.json --output-dir ncd/results/problem_scout/section21/formal_v1`。12:05 MYT 在 RTX 4060 Ti 启动，按 NSL→UNSW→CIC17→CSE18 串行执行六方法 × 五 seeds（120 单元、510 checkpoints）；日志 `results/problem_scout/section21/formal_v1/run.log`。同入口 GPU 冒烟完成 24 单元/102 checkpoints、退出 0，混淆指标独立复算一致，仅作实现证据。

首个正式单元 [NSL Joint seed 1](results/problem_scout/section21/formal_v1/nsl_kdd/joint_seed1/result.json)：最终 Macro-F1 **55.89%**、平衡准确率 **60.62%**，四阶段累计训练 **55.54 s**；不是五 seed 上限或 G1 差距结论。

## 数据结果

“重合”指测试行匹配任一训练行，原始特征和模型输入分别完整值比较。

| 数据集 | 完整训练行 | ID 测试行 | 原始重合 | 模型表示重合 |
|---|---:|---:|---:|---:|
| NSL-KDD | 125,973 | 22,544 | 664 | 664 |
| UNSW-NB15 | 120,991 | 30,248 | 0 | 13 |
| CIC-IDS-2017 | 2,262,203 | 565,673 | 0 | 179 |
| CSE-CIC-IDS2018 | 12,882,081 | 3,255,102 | 0 | 4,030 |

UNSW 从 257,673 行删除 2,445 个冲突组的全部 40,400 行及同标签重复 excess 66,034 行，seed 42 重划分；独立复核一致。13 条 Normal 碰撞来自 float32 舍入，不事后删测试样本。NSL 保留官方划分。CIC 缓存**仅保证原始跨划分不重合，并非全局去重/无标签噪声**：训练重复 excess 264,889 / 3,780,187，冲突成员 6,068 / 1,259,263。533 个分片及 18 个准备后数组已验证。

S1 全部 3,255,102 条目标测试行中，18,183 条与 CIC17 训练原始重合（0.559%，含 269 条异标签），限制 G6 的独立跨环境解释；既定映射不变。稀有类不补样，例如 CIC17 Infiltration 训练 30 / 测试 6。[原始数据结果与诊断](results/problem_scout/section21)。

## 基线代码可用性

[DER++ 官方源码](https://github.com/aimagelab/mammoth/tree/e75a491c69fd729edeb01431afb753d9157d9a81)及 MIT 许可已核实；声明适配为 200/类 reservoir、共享 MLP/Adam/类平衡 CE，保留双独立回放项及更新前 logits，不称原协议复现。Replay、iCaRL、Replay+LwF、RCL-tab 的旧类仅访问已存回放（修正历史旧完整池重访）；Joint 每 checkpoint 从头训练全部已见类。

母表 75 条已筛选，**近期 NIDS 按本协议跑通为 0**：ACIL 是既有策略框架且 PAPA 未激活；2025 EAAI `incremental-nids` 有代码，但包序列输入不匹配且 KD 转 float 丢梯度；RPL/SSF 为二分类；SPIDER 仅 README/LICENSE；其余身份/代码绑定未解决不等于无代码。[逐项一手来源与检查](results/problem_scout/section21/baseline_availability.json)。不重写无代码论文，不把缺项当败北。

## 判定与下一步

数据与代码核查完成，正式比较未完成，**第 21 节尚未达标**。继续矩阵并实测期限；10 月 16 日在本报告更新 G1–G7、最大差距的文献图及按 21.5 合格的候选（不足三项不凑数），保留基线缺项与数据限制。当前不作空白或排名结论。
