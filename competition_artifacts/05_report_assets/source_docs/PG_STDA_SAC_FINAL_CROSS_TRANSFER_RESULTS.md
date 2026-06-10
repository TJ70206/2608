# PG-STDA-SAC-RSPA-TC 最终双迁移结果

日期：2026-06-07

本文档汇总 XA-202608 两个迁移任务当前采用的统一最终方法、报告口径和关键结果。

## 最终方法

推荐完整名称：

```text
Physics-Guided Stage-aware Transfer Domain Adaptation with Stage Auxiliary Calibration,
Reliability-weighted Stage Prototype Alignment, and Validation-only Time-aware Output Calibration
(PG-STDA-SAC-RSPA-TC)
```

核心机制：

- `P-SA-MCD-TCN` 时序表征骨干。
- 使用目标域伪时间阶段的 `Stage-aware LMMD`。
- `Stage Auxiliary Calibration (SAC)`：
  - 源域阶段 CE 来自源域退化阶段标签；
  - 目标域阶段 CE 只来自无标签伪时间阶段。
- 源域监督平衡：
  - `source_supervised_weight=0.70`。
- `Reliability-weighted Stage Prototype Alignment (R-SPA)`：
  - 在退化阶段内部对齐源域与目标域特征原型；
  - 目标域原型按 detached 阶段头置信度加权；
  - 低置信伪阶段样本由 `prototype_min_confidence` 门控。
- `Validation-only Time-aware Output Calibration (TC)`：
  - 在 `predictions_val.csv` 上拟合一个二阶 ridge 校准映射；
  - 特征只使用 raw `y_pred` 与绝对 `time_index`；
  - 将拟合好的映射应用到测试集预测；
  - 不使用目标测试集标签、测试集派生阶段或每个单元未来长度。

迁移训练损失不使用目标域训练集 RUL 标签。最终 TC 只使用目标域验证集标签做输出校准，定位等同于模型选择/校准环节，而不是测试集拟合。

## 最终配置

| 任务 | 最终配置 |
|---|---|
| 第一迁移：`XJTU-SY -> reaction_wheel_sim` | `configs/xjtu_to_reaction_wheel_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml` |
| 第二迁移：`NASA Battery -> satellite_battery_sim` | `configs/nasa_to_satellite_battery_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml` |

## 主结果

指标方向：RMSE、MAE、NASA得分、末窗口RMSE、末5窗口RMSE越低越好；RA与alpha诊断越高越好。

| 任务 | 方法 | RMSE | MAE | NASA | RA | alpha 0.5 | alpha 0.8 | Last RMSE | Last-5 RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 第一迁移 | P-SA-MCD stage LMMD + pseudo-time | 0.1943 | 0.1535 | 10.6761 | 0.6024 | 0.2561 | 0.2000 | 0.2134 | 0.2079 |
| 第一迁移 | PG-STDA-SAC-RSPA, uncalibrated | 0.1274 | 0.1004 | 7.0249 | 0.6902 | 0.4024 | 0.3529 | 0.1732 | 0.1620 |
| 第一迁移 | PG-STDA-SAC-RSPA-TC, final | 0.0891 | 0.0691 | 4.8414 | 0.7418 | 0.6951 | 0.3412 | 0.1400 | 0.1451 |
| 第二迁移 | AD-TCN-MSC-DIM stage LMMD + pseudo-time | 0.3407 | 0.2789 | 71.7953 | 0.4564 | 0.4783 | 0.1149 | 0.4225 | 0.4202 |
| 第二迁移 | P-SA-MCD stage LMMD + pseudo-time | 0.3571 | 0.2913 | 72.6033 | 0.4435 | 0.3913 | 0.1892 | 0.3550 | 0.3541 |
| 第二迁移 | PG-STDA-SAC-RSPA, uncalibrated | 0.2888 | 0.2316 | 58.9301 | 0.5117 | 0.4749 | 0.1216 | 0.3111 | 0.3166 |
| 第二迁移 | PG-STDA-SAC-RSPA-TC, final | 0.1525 | 0.1161 | 29.5495 | 0.6706 | 0.5351 | 0.2838 | 0.2052 | 0.2108 |

## 解释

`PG-STDA-SAC-RSPA-TC` 是当前推荐的统一最终方法：

- 第一迁移 RMSE 从阶段 LMMD 强基线的 `0.1943` 降至 `0.0891`；
- 第二迁移 RMSE 从最强严格未校准基线 `0.3407` 降至 `0.1525`；
- 第二迁移未校准模型存在输出动态范围压缩，尤其是高 RUL 窗口低估；TC 使用验证集校准修正这一问题，不进行测试集拟合；
- 两个迁移任务使用同一框架与同一 TC 设置：`degree=2`，`ridge=0.01`。

报告边界：

- 最终比赛模型使用 `PG-STDA-SAC-RSPA-TC`。
- 严格迁移消融使用未校准的 `PG-STDA-SAC-RSPA`。
- `target-only` 和 `target-supervised` 方法只作为监督参考，不与严格无监督迁移主张混为一类。
