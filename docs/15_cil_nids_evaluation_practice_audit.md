# 类增量 / 持续学习 NIDS 论文的评估做法核对（2026-11-02）

目的：回答“已发表的论文是不是也有和我们一样的局限”。逐篇读全文（或作者公开全文），只记原文写了什么；原文没写的标“未写明”，读不到全文的单列，不做推断。
这不是全领域统计，是定向样本：6 篇逐项核对全部列，2 篇只沿用仓库已核的部分列，另有 2 篇读不到全文。

## 1. 核对表

列含义：
- **重复次数**：独立重复/seed 数，是否报标准差。
- **显著性检验**：评估结果是否做了统计检验（方法内部用到的检验不算）。
- **固定误报工作点**：是否报告在某个固定 FPR 下的检出率/BAcc（ROC-AUC、PR-AUC 不算固定工作点）。
- **数据划分**：训练/测试怎么分，是否去重。
- **正常流量进回放内存**：旧类回放里是否保留良性样本。

| 论文（出处） | 设定 | 重复次数 | 显著性检验 | 固定误报工作点 | 数据划分 | 正常流量进内存 | 读取来源 |
|---|---|---|---|---|---|---|---|
| **ECBRS/PAPA**，Amalapuram, Channappayya, Tamma，NeurIPS 2023 | 二元 domain-incremental，每个任务含良性+攻击 | 5 次，表中只报均值（Table 3：“arithmetic mean … repeated five times”） | 无 | 无；ROC-AUC、PR-AUC(A/B) | 正文未写明（细节在补充材料，未读） | 是；良性在缓冲区内，替换时优先替换良性（§3.1） | 官方全文 PDF |
| **CBRS-NIDS**，Amalapuram 等，AIMLSystems 2021 | 二元，CICIDS2017 分 5 个任务 | 5 种任务顺序；无标准差 | 无 | 无；Accuracy、P、R、F1 | 正文未写明 | 是；WACB 把攻击合并后与良性平衡（§3.1） | 作者全文 PDF |
| **SPIDER**，Amalapuram, Tamma, Channappayya，IEEE INFOCOM 2024 | 半监督、二元 domain-incremental | 5 种任务顺序/seed，报算术均值（Table IV） | 无 | 无；ROC-AUC、PR-AUC | KDD/NSL 去除重复行；其余未写明 | 是；保留上一任务样本（含良性） | 作者全文 PDF |
| **SSF**，Zhang 等，IEEE INFOCOM 2025 | 二元、概念漂移 | 未写明 | 无（K-S 检验是方法内的漂移检测，不是评估检验） | 无；Acc、Pre、Rec、F1 | NSL-KDD、UNSW-NB15 官方训练/测试集 | 是；固定缓冲区存代表性样本（二元，含正常） | arXiv 全文 v4 |
| **CND-IDS**，Fuhrman, Gungor, Rosing，DAC 2025 | 无监督持续新颖性检测 | 未写明 | 无 | 无；F1、PR-AUC、前/后向迁移 | 每个 experience 内部划分训练/测试；另取 10% 正常流量建参考集 | 不回放旧攻击；保留一份干净正常流量参考集 | arXiv 全文 |
| **Cerasuolo 等**，*Computer Networks* 263 (2025) 111228 | 多类 CIL（IoT），基类含 Normal，新类为攻击 | 留一攻击轮换：TON_IoT、IoT-NID 各 9 轮，Edge-IIoT 14 轮；报均值±标准差 | 无 | **无固定工作点，但报 FPR ≤ 1% 区间的 partial AUC（macro-PAUC）** | 70%/30% hold-out；未提去重 | 是；每类 25 个 exemplar，含 Normal | 作者博士论文第 4 章（基于该期刊论文）；期刊版正文未直接读 |
| **Cerasuolo 等**，*Engineering Applications of AI* 144 (2025) 110143 | 跨域 + CIL（IoT） | 见下注 | 见下注 | 见下注 | 见下注 | Normal 是源/目标共同类，FT-Mem 保留源类子集 | 本仓库 `ncd/LITERATURE_DIFFERENCE_TABLE.md` §27.3（作者全文已核） |
| **ACIL**，*Int. J. Mach. Learn. & Cybern.* 17:104 (2026) | 二元及多类 | 见下注 | 见下注 | **部分**：定义了 FPR/TNR，主表未单列 FPR 数值 | 见下注 | 多种策略（reservoir、herding、CBRS），总缓冲 K=1000 | 本仓库同一文件 §26.8（作者全文已核） |

注：EAAI 2025 和 ACIL 这两行只沿用仓库里已核过的列（内存、FPR），重复次数、检验和划分这次没有重新核对，所以不填。

**读不到全文、未纳入判断的期刊论文**（只核实了题目和出处，没有猜测其评估做法）：
- Du 等，**BFS-NID**，*IEEE TNSM* 21(2) 2024，DOI 10.1109/TNSM.2023.3332284（付费墙；数据集为 CIC-IDS2017、CSE-CIC-IDS2018）。
- Di Monda 等，*IEEE OJ-COMS* 5 (2024) 6736–6757，DOI 10.1109/OJCOMS.2024.3481895：本次仓储站点连接失败；但 2026-10-22 曾逐篇核实（手册 §10.1b）：100 个 episode 取平均并报 95% 置信区间（无配对检验）；报告 FPR，但是攻击合成一类后在默认判定下计算，不是固定工作点；分层留出 30% 测试，未提去重。手册 §10.1b 另有 T-DFNN（IEEE Access 2021）、Guo 等（Future Internet 2025）、Park 等（Applied Sciences 2025）、Wu 等（Sensors 2026）的核实记录：均无固定误报、均无配对显著性检验。

## 2. 和本研究的对照

| 项目 | 上表论文的常见做法 | 本研究 |
|---|---|---|
| 重复次数 | 5 次或按任务顺序/留一轮换；ECBRS、CBRS、SPIDER 只报均值 | 5 个 seed，报均值±标准差 |
| 显著性检验 | 逐项核对的 6 篇中 0 篇对评估结果做统计检验 | 预先登记：配对单侧 t 检验 + 数据集内 Holm，另报精确符号翻转 |
| 固定误报工作点 | 逐项核对的 6 篇中 0 篇报固定工作点，其中 1 篇（Cerasuolo CompNet）报 FPR≤1% 的 partial AUC；另 ACIL 定义了 FPR 但主表未单列 | 留出池上校准到 1% 误报（另报 2%、5%），同时报默认判定 |
| 数据划分 | 官方划分或随机 hold-out；只有 SPIDER 写了对 KDD/NSL 去重 | UNSW 去除冲突重复后分层划分；CIC-2017 分组划分；NSL 用官方 KDDTest+ 并单独标注偏移 |
| 正常流量进内存 | 8 篇中 7 篇在回放/缓冲区里保留良性（ACIL 依所选策略而定）；CND-IDS 不回放攻击，但保留正常参考集 | 内存只存攻击；每个任务重新采集 5 万条正常流量（R3-L 对照检验了“良性进内存”的做法） |

## 3. 能说什么、不能说什么

可以说（有出处支撑）：
- “在我们逐项核对的 6 篇 CIL/CL-NIDS 论文中，没有一篇对评估结果做统计显著性检验，也没有一篇报告固定误报工作点下的检出性能。”
- “与本研究最接近的低误报评估是 Cerasuolo 等（Computer Networks 2025）的 FPR ≤ 1% partial AUC；我们在此基础上改为在留出良性池上校准的单一部署工作点。”这句同时回答了“1% 误报参考了谁”，可以补进第 6 页和论文。
- “多数方法让良性样本与攻击样本竞争同一回放内存。”

不能说：
- 不能说“领域内都不报 FPR”或给出全领域比例：这是 8 篇的定向样本。
- 不能说 Cerasuolo 没关注低误报：他们报了 FPR≤1% 的 PAUC。
- 不能把“只存攻击 + 内存外新良性”说成从未出现：Cerasuolo 等 RTSI 2025 的一次实验已有相邻设定（见仓库 §27.3）。
- 不能评价 BFS-NID、Di Monda 的评估做法：全文没读到。

## 4. 待办

1. 已核对（2026-11-03）：作者在 [ResearchGate 上传的 Computer Networks 期刊论文全文](https://www.researchgate.net/publication/389901457_Attack-Adaptive_Network_Intrusion_Detection_Systems_for_IoT_Networks_through_Class_Incremental_Learning)，§4.1 Evaluation metrics（正文第7–8页），明确将 PAUC 限定在 FPR 上限为1%的区间；公式(6)对各攻击类相对良性类的二元(P)AUC作不加权宏平均，随后简称(P)AUC。因此确有 FPR≤1% 的 macro partial AUC，不是单一固定FPR工作点。本次读取的是该论文全文，不再仅依赖博士论文；ScienceDirect 直接正文访问403，大学仓储无附件。按 docs/09_citation_audit.md，已以 [出版社元数据](https://www.sciencedirect.com/science/article/pii/S1389128625001963)及[作者大学仓储](https://www.iris.unina.it/handle/11588/1013147)交叉核对：DOI 10.1016/j.comnet.2025.111228；作者依次为 Francesco Cerasuolo、Giampaolo Bovenzi、Domenico Ciuonzo、Antonio Pescapè。未新增或导入任何 bib 条目；其他条目导入前仍需逐项核对。
2. 如需扩大样本，优先补读 BFS-NID（TNSM）和 Di Monda（OJ-COMS）全文。
3. 任何条目导入 bib 前按 `docs/09_citation_audit.md` 规则核对 DOI 和作者名单。

## 来源

- ECBRS/PAPA：<https://papers.neurips.cc/paper_files/paper/2023/file/3755a02b1035fbadd5f93a022170e46f-Paper-Conference.pdf>
- CBRS-NIDS：<https://newslab.iith.ac.in/publications/on-handling-class-imbalance-in-continual-learning-based-network-intrusion-detection-systems/download/>
- SPIDER：<https://newslab.iith.ac.in/media/publications/pdfs/SPIDER-Suresh.pdf>
- SSF：<https://arxiv.org/abs/2412.16264>（Comments 写明 INFOCOM 2025 录用）
- CND-IDS：<https://arxiv.org/abs/2502.14094>
- Cerasuolo 博士论文：<https://www.fedoa.unina.it/15959/1/_PhD_Thesis__Francesco_Cerasuolo.pdf>（第 4 章 §4.2，70%/30% hold-out、25 exemplar/类、macro-PAUC@FPR≤1%）
- Cerasuolo CompNet 2025 出处：<https://dl.acm.org/doi/10.1016/j.comnet.2025.111228>
- BFS-NID 出处：<https://dl.acm.org/doi/abs/10.1109/TNSM.2023.3332284>
- Di Monda 出处：<https://researchr.org/publication/MondaMPVIP24/bibtex>
