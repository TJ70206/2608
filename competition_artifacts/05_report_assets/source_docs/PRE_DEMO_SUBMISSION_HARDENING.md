# 动态 Demo 前提交级加固清单

日期：2026-06-10

用途：本文件记录在 Docker 收尾和动态 demo 开发之前，当前方案已经完成和仍需保持的报告口径。它面向最终报告、答辩材料和后续 dashboard/demo 开发。

## 1. 当前已完成的非 Docker 加固

### 1.1 TC 消融

已新增：

- `scripts/time_calibration_ablation.py`
- `docs/TC_CALIBRATION_ABLATION.md`
- `competition_artifacts/03_results/tc_ablation/tc_ablation_summary.md`
- `competition_artifacts/03_results/tc_ablation/tc_ablation_summary.csv`

该消融不重训模型，只读取最终 raw `PG-STDA-SAC-RSPA` 的 `predictions_val.csv` 与 `predictions_test.csv`。

消融口径：

| 口径 | 拟合特征 | 是否用目标验证标签 | 是否用测试标签 |
|---|---|---|---|
| raw | 无 | 否 | 否 |
| y_pred-only TC | raw `y_pred` | 是 | 否 |
| time-only TC | `time_index` | 是 | 否 |
| y_pred+time TC | raw `y_pred` + `time_index` | 是 | 否 |

关键结论：

- 第一迁移：`y_pred+time TC` 最好，说明 raw 迁移输出与时间先验互补。
- 第二迁移：`time-only TC` 的 RMSE 略低于 `y_pred+time TC`，说明卫星电池仿真域存在很强时间/生命周期先验。
- 因此 strict raw 迁移表仍是迁移表征能力主证据；TC 后结果用于最终工程预测管线展示。

### 1.2 证据包自动生成集成

`scripts/prepare_competition_artifacts.py` 已集成 TC 消融。每次重建 `competition_artifacts` 时都会自动生成：

- TC 消融表；
- TC 消融预测 CSV；
- TC 校准系数 JSON；
- `docs/TC_CALIBRATION_ABLATION.md` 副本。

最近一次验证结果：

```text
Metrics rows: 81; TC ablation rows: 8; missing: 0
```

### 1.3 方法创新口径修正

`docs/METHOD_NOVELTY_FRONTIER_RESEARCH_REVIEW.md` 已修正：

- 无标签单调/平滑一致性不再表述为两个迁移任务共同必需的核心模块；
- 第一迁移最终配置不依赖该项；
- 第二迁移最终配置轻量启用 `target_sequence_monotonic_weight=0.001`；
- 最终主贡献应集中在机理代理、伪时间阶段对齐、SAC、R-SPA 和 validation-only TC 的分层证据。

## 2. Conformal 口径

当前最终 `PG-STDA-SAC-RSPA-TC` 主结果不依赖 Conformal Prediction。

后续报告建议：

- 不把 Conformal 写成最终主模型核心模块；
- 可以作为“可选不确定性/风险区间表达模块”放入附录或 demo 扩展；
- 若正式展示 Conformal 区间，必须单独给出 calibration split、coverage 和 average width；
- 在没有最终双迁移 conformal 结果前，不使用“已完成不确定性量化”作为主张。

## 3. Demo 前固定输入产物

动态 demo 或静态 dashboard 不应现场训练模型，建议只读取以下已生成产物：

| 用途 | 推荐读取路径 |
|---|---|
| 第一迁移最终预测 | `competition_artifacts/02_experiments/final_outputs/first_transfer/predictions_test.csv` |
| 第一迁移最终指标 | `competition_artifacts/02_experiments/final_outputs/first_transfer/transfer_metrics.json` |
| 第二迁移最终预测 | `competition_artifacts/02_experiments/final_outputs/second_transfer/predictions_test.csv` |
| 第二迁移最终指标 | `competition_artifacts/02_experiments/final_outputs/second_transfer/transfer_metrics.json` |
| 严格 raw 对比 | `competition_artifacts/03_results/strict_unsupervised_comparison.md` |
| TC 消融 | `competition_artifacts/03_results/tc_ablation/tc_ablation_summary.csv` |
| 总指标 | `competition_artifacts/03_results/metrics_master.csv` |
| 预测轨迹图 | `competition_artifacts/04_figures/figure_3_representative_predictions.png` |
| PHM 看板图 | `competition_artifacts/04_figures/figure_6_phm_application_dashboard.png` |
| 机理-遥测图 | `competition_artifacts/04_figures/figure_5_mechanism_observable_map.png` |

## 4. Demo 推荐页面结构

后续动态 demo 建议做成“只读结果展示”，不要做训练入口。

建议包含：

1. 任务切换：反作用轮 / 卫星电池。
2. 核心指标卡：RMSE、MAE、NASA、RA、末窗口 RMSE。
3. 预测轨迹：true RUL、raw prediction、final TC prediction。
4. TC 消融面板：raw、y_pred-only、time-only、y_pred+time。
5. PHM 风险卡：健康 / 关注 / 严重。
6. 机理-遥测说明：可观测量如何对应退化机制。

注意：

- demo 中实验结论只引用代码生成图表和 CSV/JSON 指标；
- AI 生成图只能标注为 conceptual schematic；
- 不要把 `time-only TC` 的第二迁移强结果隐藏，应作为诚实边界说明。

## 5. 最终报告安全表述

可以写：

> `PG-STDA-SAC-RSPA` 是 strict raw 迁移模型，证明在不使用目标训练 RUL 标签的条件下，相比 source-only、global MMD、stage LMMD 等严格基线具有更好的目标域泛化表现。

可以写：

> `PG-STDA-SAC-RSPA-TC` 是 validation-only calibrated final engineering pipeline，使用目标验证集进行输出尺度校准，不使用测试集标签。

必须避免：

- “TC 是完全无监督训练模块”；
- “第二迁移 TC 后提升完全来自迁移表征学习”；
- “Conformal 是最终核心模块”；
- “AI 图证明实验结果”。

## 6. Docker 和动态 Demo 前剩余项

后续再做：

- Docker build/run 验证；
- 一键复跑脚本；
- 数据打包与 checksum；
- 动态 demo 或静态 dashboard；
- 可选 Conformal 展示。

当前阶段不建议继续更换主模型或新增复杂深度主干。
