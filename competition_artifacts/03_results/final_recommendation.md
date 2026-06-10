# 最终推荐方案

统一推荐方法：Physics-Guided Stage-aware Transfer Domain Adaptation with Stage Auxiliary Calibration, Reliability-weighted Stage Prototype Alignment, and Validation-only Time-aware Output Calibration。

图表和结果表中使用的短名：`PG-STDA-SAC-RSPA-TC`。

核心组成：

- `P-SA-MCD-TCN` 时序表征骨干。
- 针对不同航天组件的物理代理特征和可观测遥测特征。
- 基于目标域伪时间阶段的 `Stage-aware LMMD`。
- `Stage Auxiliary Calibration (SAC)`：用源域阶段标签和目标域无标签伪阶段辅助校准退化阶段表征。
- `Reliability-weighted Stage Prototype Alignment (R-SPA)`：按目标阶段置信度加权对齐阶段原型。
- `Validation-only Time-aware Output Calibration (TC)`：只用目标域验证集拟合时间感知输出校准，不使用测试集标签。

推荐最终配置：

| 任务 | 配置 | RMSE | MAE | NASA | RA |
|---|---|---:|---:|---:|---:|
| 第一迁移 | `configs/xjtu_to_reaction_wheel_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml` | 0.0891 | 0.0691 | 4.8414 | 0.7418 |
| 第二迁移 | `configs/nasa_to_satellite_battery_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml` | 0.1525 | 0.1161 | 29.5495 | 0.6706 |

报告口径边界：

- 严格无监督迁移训练不在迁移损失中使用目标域训练集 RUL 标签。
- 最终 TC 只使用目标域验证集标签做输出校准，绝不使用目标测试集标签或测试集派生阶段。
- `target-only` 与 `target-supervised` 结果属于监督参考/上界，不作为严格无监督迁移主张。
- 最终方法作为两个场景统一的迁移框架提交，避免第一迁移和第二迁移分别使用完全不同模型造成方案割裂。
