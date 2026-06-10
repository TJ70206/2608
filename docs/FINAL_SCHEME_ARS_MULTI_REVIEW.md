# XA-202608 最终方案多角色评审备忘录

生成时间：2026-06-09  
用途：本文件用于最终收尾、技术报告撰写和答辩口径统一。它不是新的实验结果文件，而是对当前方案的客观评审、风险梳理和改进优先级记录。

## 1. 总体结论

当前方案不是普通“中上水平”，而是具备比赛前排竞争力的完整方案。若只看学术算法原创性，它更接近“工程应用强创新、学术方法中等偏上创新”；若按比赛评分维度看，它的优势更明显，因为比赛同时考察问题建模、仿真场景、跨域迁移效果、工程复现和应用表达。

最稳妥的总体定位是：

> 面向航天器关键组件寿命预测的机理引导伪阶段跨域迁移框架。方案统一覆盖 `XJTU-SY -> reaction_wheel_sim` 与 `NASA Battery -> satellite_battery_sim` 两个迁移任务，在公开退化源域监督训练基础上，通过目标域伪时间阶段、阶段感知局部分布对齐、阶段辅助约束、可靠性加权阶段原型对齐和验证集输出校准，实现航天仿真目标域 RUL 预测。

## 2. 竞赛评分视角

| 评分维度 | 满分 | 当前判断 | 估计区间 |
|---|---:|---|---:|
| 问题建模与仿真场景设计 | 30 | 两个场景正好覆盖反作用轮和卫星电池；退化机理、可观测量、EOL、泄漏控制和数据划分均较完整。 | 27-29 |
| 工程实现与可复现性 | 50 | 代码、配置、测试、结果、证据包已有较好基础；但 Docker、一键复跑、依赖锁定和最终权重归档仍需收尾。 | 当前 39-43；完善后 46-49 |
| 寿命预测算法效果、迁移能力与应用表达 | 20 | strict raw 迁移优于主要无监督基线；最终 TC 管线指标强。 | 17-19 |
| 额外加分 | +5 | 可视化、PHM 流程图、机理-遥测图、预测轨迹和应用看板已有优势。 | 4-5 |

最终冲高分的关键不在继续无限堆模型，而在把证据链做得无可挑剔：公平口径、Docker、复跑脚本、数据边界、报告叙事。

## 3. 当前方案的核心优点

### 3.1 赛题覆盖完整

- 第一迁移：`XJTU-SY -> reaction_wheel_sim`，对应反作用轮长期运行摩擦退化场景。
- 第二迁移：`NASA Battery -> satellite_battery_sim`，对应卫星电池容量下降、内阻增加和轨道周期充放电场景。
- 两个目标域均为航天关键组件机理约束仿真数据，满足至少两个典型预测场景的硬要求。

### 3.2 方法链条连贯

最终方法可拆为两层：

- `PG-STDA-SAC-RSPA`：严格 raw 迁移模型。
- `PG-STDA-SAC-RSPA-TC`：验证集校准后的最终工程预测管线。

核心模块逻辑：

- `P-SA-MCD-TCN`：以 TCN 为基础时序骨干，叠加多尺度/域适配结构。
- `PG-STDA`：使用机理代理特征和伪时间阶段，避免只做全局分布对齐。
- `SAC`：阶段辅助约束，让特征同时服务 RUL 回归和退化阶段判别。
- `R-SPA`：用阶段头置信度加权原型对齐，减轻伪阶段不可靠样本的负迁移。
- `TC`：只用目标验证集 `y_pred` 和 time index 做输出校准，修正跨域输出动态范围压缩。

### 3.3 严格 raw 对比结果成立

当前最重要的公平证据是 strict raw 表，而不是 TC 后结果。

| 迁移任务 | raw 最终模型 | 关键结论 |
|---|---|---|
| 第一迁移 | `PG-STDA-SAC-RSPA`，RMSE 0.1274 | 优于 `source-only`、`global MMD`、`stage LMMD` 和常见骨干迁移基线。 |
| 第二迁移 | `PG-STDA-SAC-RSPA`，RMSE 0.2888 | 优于主要 strict unsupervised baselines，说明第二迁移不是只靠 TC 后处理。 |

最终 TC 管线结果：

| 迁移任务 | 最终工程管线 | RMSE | MAE | RA |
|---|---|---:|---:|---:|
| 第一迁移 | `PG-STDA-SAC-RSPA-TC` | 0.0891 | 0.0691 | 0.7418 |
| 第二迁移 | `PG-STDA-SAC-RSPA-TC` | 0.1525 | 0.1161 | 0.6706 |

### 3.4 可视化和应用表达有加分潜力

已有图件方向是对的：

- 严格 raw 迁移性能对比图。
- 最终 TC 管线预测轨迹图。
- 机理-遥测映射图。
- PHM 应用流程和风险看板。
- AI 生成概念图可作为视觉增强，但不能替代实验图。

## 4. 最大风险与防守口径

### 4.1 TC 校准边界

风险：`PG-STDA-SAC-RSPA-TC` 使用目标验证集标签做输出校准。它没有使用测试标签，不是测试泄漏；但它也不是严格无目标标签训练。

必须这样表述：

- `PG-STDA-SAC-RSPA` 是 strict raw 迁移模型。
- `PG-STDA-SAC-RSPA-TC` 是 validation-only calibrated final pipeline。
- strict raw 表用于公平无监督迁移比较。
- TC 表用于工程部署口径展示。

不要这样表述：

- 不要说 `PG-STDA-SAC-RSPA-TC` 是“完全无监督迁移训练模型”。
- 不要把 TC 后结果直接和 raw baseline 混在同一公平排行榜里。
- 不要声称“无监督迁移模型全面超过监督模型”。

### 4.2 自建仿真数据集的自证风险

风险：`reaction_wheel_sim` 和 `satellite_battery_sim` 是自研机理约束仿真数据，不是真实在轨飞行遥测，也不是 STK/Simulink/AMESim 等高保真整星仿真。

防守口径：

- 主动称为“机理约束航天目标域仿真 testbed”。
- 强调公开源域数据、可复现生成器、单元级划分、泄漏列禁用、故障模式和参数敏感性。
- 不宣称替代真实在轨试验。

### 4.3 伪时间阶段可能被质疑

风险：目标域用 `time_progress` 或 pseudo-time 近似退化阶段，TC 又使用 time index。反方可能认为这是寿命进度代理。

防守口径：

- 伪时间阶段是记录时间先验，不是 RUL 标签。
- 目标训练 RUL 标签不进入 strict raw 训练损失。
- 已补充 `time-only`、`y_pred-only`、`y_pred + time` 的 TC 消融，见 `docs/TC_CALIBRATION_ABLATION.md` 和 `competition_artifacts/03_results/tc_ablation/tc_ablation_summary.md`。
- 消融显示第二迁移 `time-only TC` 很强，报告中应把第二迁移 strict raw 结果作为迁移表征主证据，把 TC 后结果定位为验证集校准后的工程预测管线。

### 4.4 工程复现仍是最大硬分风险

当前工程已有基础，但最终提交前还需要：

- 顶层一键复跑脚本。
- Docker build/run 实测记录。
- 最终权重、配置、预测、指标、环境信息统一归档。
- 去除本机绝对解释器路径依赖。
- 依赖版本锁定或关键版本说明。

## 5. 创新性边界

### 5.1 可以主张的创新

- 面向航天器关键组件 RUL 的双目标域迁移闭环。
- 机理代理特征与可观测遥测量结合。
- 无目标训练 RUL 标签的伪时间阶段对齐。
- `SAC + R-SPA` 的阶段可靠性建模。
- 验证集输出校准用于部署阶段动态范围修正。
- 严格 raw 和 final TC 分层评估。

### 5.2 不应主张的创新

- 不要说首次提出 `MMD`、`LMMD`、`TCN`、`RUL` 迁移。
- 不要说实现了 `JAN/JMMD` 改进算法；当前不是联合核对齐实现。
- 不要说达到 `EviAdapt` 式 evidential uncertainty alignment；当前没有 NIG、证据学习或不确定性分布对齐。
- 不要说是严格 physics-informed neural network；更准确是 physics-guided / 机理代理特征。
- 不要把 `TC` 包装成严格无监督训练创新。

### 5.3 推荐报告主句

> 本方案提出一种面向航天器关键组件寿命预测的机理引导伪阶段跨域迁移框架 `PG-STDA-SAC-RSPA-TC`。训练阶段利用公开退化源域监督、目标域伪时间阶段和阶段感知局部分布对齐完成无目标训练 RUL 标签的迁移适配，并通过阶段辅助约束与置信度加权阶段原型对齐缓解跨域退化阶段错配。最终预测阶段采用仅基于目标验证集的时间感知输出校准，以修正跨域模型输出动态范围压缩问题。

### 5.4 推荐文献关系句

> 与全局 `MMD/JAN` 类通用分布对齐不同，本方案不主张提出新的核距离，而是将对齐粒度约束在退化阶段内；与 `EviAdapt` 相比，本方案吸收其阶段错配问题设定，但未采用 evidential uncertainty alignment，而采用更轻量的阶段头置信度原型对齐，换取实现稳定性和比赛复现性。

## 6. 最终报告中的表格口径

建议最终只保留三类主表，避免混乱：

| 表格 | 是否使用目标训练标签 | 是否使用目标验证标签 | 是否使用 TC | 用途 |
|---|---|---|---|---|
| strict raw UDA 对比表 | 否 | 只用于模型选择或不用于训练损失 | 否 | 证明迁移方法本体优于无监督基线。 |
| validation-calibrated final 表 | 否 | 是 | 是 | 展示最终工程预测器性能。 |
| supervised reference 表 | 是 | 是 | 可单列说明 | 作为监督参考或上界，不进入 strict UDA 主结论。 |

不要把这三类结果混成一个“总排行榜”。

## 7. 收尾前优先级清单

### P0：必须完成

- 明确 `PG-STDA-SAC-RSPA` 与 `PG-STDA-SAC-RSPA-TC` 的边界。
- Docker build/run 至少验证一次，并保留命令和结果。
- 增加或整理顶层复跑入口。
- 最终权重、配置、预测、指标、环境信息归档到统一证据包。
- 报告中明确目标测试标签只用于最终评估。

### P1：强烈建议

- TC 消融已完成；后续报告直接引用 `TC_CALIBRATION_ABLATION.md`。
- 对 final raw 和 final TC 做 3 seed 或 bootstrap CI，至少第二迁移补稳定性。
- 增加 R-SPA 置信门控或 prototype weight 敏感性说明。
- 加一段自建仿真数据可信度和局限性说明。
- 给所有 AI 生成图标注 conceptual schematic，实验图只用代码生成图。

### P2：可以作为附录或答辩补充

- 负控实验：打乱 pseudo-time stage、禁用物理代理、禁用阶段头。
- 高/中/低 RUL 分段误差，解释 TC 修正的是尺度偏差。
- 数据集生成器关键参数表和泄漏列禁用表。
- 运行时间、模型参数量、推理效率统计。

## 8. 最终答辩时的安全表述

可以说：

- “在固定数据划分和固定 seed 的复现实验中，我们的 strict raw 迁移模型优于主要无监督迁移基线。”
- “最终工程管线使用目标验证集进行输出校准，不使用测试标签。”
- “两个目标域均为机理约束航天仿真数据，用于验证跨领域退化迁移流程。”
- “监督模型作为参考上界，不与无监督迁移主结论混用。”

避免说：

- “完全无监督模型超过所有监督模型。”
- “TC 是无监督迁移训练的一部分。”
- “仿真结果等同真实在轨验证。”
- “我们首次提出 MMD/LMMD/TCN/RUL 迁移。”
- “AI 生成图证明了实验效果。”

## 9. 参考定位

外部方法定位可参考：

- `MMD` / global distribution alignment：作为全局对齐基线。
- `LMMD` / subdomain adaptation：作为阶段或子域对齐思想来源。
- `JAN/JMMD`：作为联合分布对齐相关工作，但当前方案不是 JAN 实现。
- `EviAdapt`：作为 RUL 场景阶段错配和不确定性域适应相关工作，但当前方案没有 evidential uncertainty alignment。
- `NASA Battery Aging Dataset`：作为第二迁移公开源域数据依据。

本项目本地相关文件：

- `docs/METHOD_NOVELTY_FRONTIER_RESEARCH_REVIEW.md`
- `docs/PG_STDA_SAC_FINAL_CROSS_TRANSFER_RESULTS.md`
- `docs/STRICT_UNSUPERVISED_TRANSFER_COMPARISON.md`
- `docs/TRANSFER_COMPARISON_MASTER_TABLES.md`
- `docs/REACTION_WHEEL_SIM_DATASET_DESIGN_REVIEW.md`
- `docs/SATELLITE_BATTERY_SIM_DATASET_DESIGN_REVIEW.md`
- `competition_artifacts/03_results/final_recommendation.md`
- `competition_artifacts/03_results/strict_unsupervised_comparison.md`
- `competition_artifacts/00_requirement_mapping/requirement_coverage_matrix.md`

## 10. 一句话结论

当前方案已经具备争第一梯队的基础。接下来最重要的不是继续无限增加复杂模型，而是把 `strict raw`、`final TC`、监督参考、仿真可信度和工程复现这几条证据线分清楚、做扎实。只要报告和封装不出硬伤，这套方案有较强竞争力。
