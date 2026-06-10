# 第二迁移优化与基础对照复查

日期：2026-06-05

任务：NASA Battery -> satellite_battery_sim，基于电压、电流、温度遥测量预测归一化 RUL。

## 1. 比赛要求复核

比赛方案在“寿命预测算法效果、迁移能力与应用表达”中要求：

- 完成“公开退化数据集训练 -> 航天仿真数据集微调/适配”的迁移学习流程；
- 与至少一种现有寿命预测或退化建模方法在相同数据条件下进行对比；
- 对预测误差、稳定性或提前性等指标给出清晰分析。

第二迁移当前已覆盖：

- 公开源域：NASA Battery；
- 航天目标域：satellite_battery_sim；
- 目标域可观测量：voltage/current/temperature；
- 目标场景：卫星蓄电池容量下降与内阻增长；
- 对照方法：target-only P-TCN、target-only PSA-MCD-TCN、target-only Transformer、source-only、global MMD、unsupervised stage LMMD、target-supervised stage LMMD。

## 2. 指标方向

- RMSE、MAE、NASA score、last-window RMSE：越低越好。
- RA、alpha0.5、alpha0.8：越高越好。

主排序指标采用整体 RMSE、MAE、NASA score、RA。`alpha0.8` 与 `last-window RMSE` 作为临近 EOL 的高精度/末期预测补充指标。

## 3. 50 轮同条件结果

| 方法 | RMSE↓ | MAE↓ | NASA score↓ | RA↑ | alpha0.5↑ | alpha0.8↑ | last RMSE↓ | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NASA stage LMMD + target sup, src0.02 w0.0005 | **0.188959** | **0.140652** | **38.607417** | **0.645611** | 0.394649 | 0.263514 | 0.214319 | 新主结果 |
| NASA stage LMMD + target sup, src0.05 w0.0005 | 0.189715 | 0.141262 | 39.172401 | 0.644670 | **0.408027** | 0.229730 | 0.228339 | 旧主结果 |
| NASA stage LMMD + target sup, src0.05 w0.0005 late | 0.192478 | 0.146341 | 40.238365 | 0.635215 | 0.394649 | 0.300676 | **0.203176** | 末期增强消融 |
| target-only PSA-MCD-TCN | 0.193828 | 0.148007 | 41.052099 | 0.629387 | 0.354515 | 0.277027 | 0.225459 | 强目标域基线 |
| target-only PSA-MCD-TCN late | 0.198435 | 0.146824 | 40.539256 | 0.640002 | 0.381271 | **0.314189** | 0.221198 | 目标域末期增强基线 |
| NASA stage LMMD + target sup, src0.02 w0.0002 | 0.199175 | 0.150228 | 42.702189 | 0.623238 | 0.361204 | 0.297297 | 0.263990 | LMMD 过弱后整体退化 |
| target-only Transformer | 0.204802 | 0.158728 | 42.222982 | 0.613664 | 0.304348 | 0.094595 | 0.211296 | 基础深度时序基线 |
| NASA stage LMMD + target sup, src0.25 w0.003 | 0.211776 | 0.161315 | 44.868932 | 0.605341 | 0.307692 | 0.111486 | 0.251162 | 未调低源域权重 |
| target-only P-TCN | 0.232734 | 0.182578 | 48.214073 | 0.581383 | 0.227425 | 0.155405 | 0.213472 | 基础 TCN 基线 |
| NASA stage LMMD unsupervised | 0.357965 | 0.291860 | 72.792792 | 0.442575 | 0.401338 | 0.189189 | 0.356899 | 无目标监督迁移 |
| NASA source-only | 0.509407 | 0.426242 | 100.724558 | 0.198321 | 0.000000 | 0.097973 | 0.123436 | 纯源域迁移基线 |
| NASA global MMD | 0.513024 | 0.430535 | 103.134011 | 0.158597 | 0.107023 | 0.000000 | 0.245762 | 全局对齐基线 |

## 4. 结论

1. 第二迁移还存在可优化空间。将源域监督权重从 0.05 降到 0.02 后，主结果从 RMSE 0.189715 提升到 0.188959，NASA score 从 39.172401 降到 38.607417。
2. 过弱的 LMMD 不合适。`lmmd_weight=0.0002` 使 RMSE 退化到 0.199175，说明仍需要适度阶段对齐。
3. late-stage 加权能够改善末期指标，但会牺牲整体指标。full-transfer late 的 `last-window RMSE=0.203176` 最好，`alpha0.8=0.300676` 也明显高于主模型，但整体 RMSE 退化到 0.192478。因此它适合写作消融/工程取舍，不适合作为主模型。
4. 与基础方法相比，新主模型在 RMSE、MAE、NASA score、RA 上优于 target-only P-TCN、target-only PSA-MCD-TCN、target-only Transformer、source-only、global MMD 和未调参 stage LMMD。
5. 报告中应严谨表述为：完整迁移模型在主要整体预测指标上最优；末期高精度命中率可通过 late-stage 加权进一步提高，但会引入整体误差取舍。

## 5. 锁定配置

当前第二迁移主配置：

```text
configs/nasa_to_satellite_battery_stage_lmmd_targetsup_src0p02_w0p0005_50e.yaml
```

输出目录：

```text
outputs/second_transfer_50e/nasa_to_satellite_battery_stage_lmmd_targetsup_src0p02_w0p0005_50e
```

建议作为补充消融配置：

```text
configs/nasa_to_satellite_battery_stage_lmmd_targetsup_src0p05_w0p0005_late1p5_50e.yaml
configs/satellite_battery_sim_psa_mcd_late_50e.yaml
```

## 6. 不建议继续投入的方向

- 不建议继续大网格扫 `lmmd_weight`。目前 0.0005 优于 0.0002，且 0.003 已经明显偏大。
- 不建议把主模型改成 late-stage 配置。它改善末期，但主指标下降。
- 不建议继续跑多 seed，除非最终报告需要统计显著性表述。当前可以按固定 seed=202608 的可复现实验结果写。
