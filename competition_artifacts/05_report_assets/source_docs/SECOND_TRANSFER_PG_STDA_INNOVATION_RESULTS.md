# 第二迁移 PG-STDA 创新结果

任务：`NASA Battery` 电池老化源域 -> `satellite_battery_sim` 卫星电池目标域。

## 方法更新

第二迁移最终方法为：

```text
Physics-Guided Stage-aware Transfer Domain Adaptation with Stage Auxiliary Calibration,
Reliability-weighted Stage Prototype Alignment, and Validation-only Time-aware Output Calibration
(PG-STDA-SAC-RSPA-TC)
```

迁移训练阶段仍是严格的目标域无标签适配：迁移损失不使用目标域训练集 RUL 标签。最终 TC 是只在 `predictions_val.csv` 上拟合的验证集校准阶段，输入只包含 raw prediction 与 `time_index`；不使用目标测试集标签、测试集派生阶段或每个单元的未来进度信息。

核心组成：

- `P-SA-MCD` 时序骨干。
- 只由可观测 V/I/T 派生的紧凑物理代理输入：
  - `voltage`
  - `current`
  - `temperature`
  - `d_voltage`
  - `d_temperature`
  - `abs_current`
- 基于伪时间目标阶段的 `Stage-aware LMMD`。
- 目标域无标签窗口时间一致性。
- `Stage Auxiliary Calibration (SAC)`。
- `Reliability-weighted Stage Prototype Alignment (R-SPA)`。
- `Validation-only Time-aware Output Calibration (TC)`。

推荐最终配置：

```text
configs/nasa_to_satellite_battery_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml
```

## 严格迁移与最终校准对比

指标方向：RMSE、MAE、NASA得分、末窗口RMSE、末5窗口RMSE越低越好；RA与alpha-lambda诊断越高越好。

| 排名 | 方法 | RMSE | MAE | NASA | RA | alpha 0.5 | alpha 0.8 | Last RMSE | Last-5 RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PG-STDA-SAC-RSPA-TC, final | 0.1525 | 0.1161 | 29.5495 | 0.6706 | 0.5351 | 0.2838 | 0.2052 | 0.2108 |
| 2 | PG-STDA-SAC-RSPA 0.001, confidence 0.5, uncalibrated | 0.2888 | 0.2316 | 58.9301 | 0.5117 | 0.4749 | 0.1216 | 0.3111 | 0.3166 |
| 3 | PG-STDA-SAC-RSPA 0.0008, confidence 0.5 | 0.2911 | 0.2346 | 59.8452 | 0.5100 | 0.5050 | 0.1081 | 0.3268 | 0.3304 |
| 4 | PG-STDA-SAC source balance 0.70 | 0.2914 | 0.2337 | 60.0056 | 0.5078 | 0.4916 | 0.0845 | 0.3361 | 0.3355 |
| 5 | PG-STDA-SAC source balance 0.65 | 0.2908 | 0.2346 | 60.7119 | 0.5043 | 0.5518 | 0.0676 | 0.3519 | 0.3533 |
| 6 | PG-STDA-SAC | 0.3059 | 0.2465 | 62.8317 | 0.4908 | 0.4013 | 0.1284 | 0.3382 | 0.3444 |
| 7 | PG-STDA compact proxy + TLMC 0.001 | 0.3178 | 0.2552 | 64.6922 | 0.4725 | 0.4247 | 0.0777 | 0.3278 | 0.3370 |
| 8 | PG-STDA compact proxy + low monotonic | 0.3326 | 0.2653 | 66.5476 | 0.4576 | 0.3010 | 0.1250 | 0.3166 | 0.3274 |
| 9 | AD-TCN-MSC-DIM stage LMMD + pseudo-time | 0.3407 | 0.2789 | 71.7953 | 0.4564 | 0.4783 | 0.1149 | 0.4225 | 0.4202 |
| 10 | P-SA-MCD stage LMMD + pseudo-time | 0.3571 | 0.2913 | 72.6033 | 0.4435 | 0.3913 | 0.1892 | 0.3550 | 0.3541 |

## 解释

此前对第二迁移可视化效果的担心是合理的。raw `PG-STDA-SAC-RSPA` 输出存在动态范围压缩：高 RUL 窗口被低估，末期窗口又偏保守高估。

最终 TC 将其作为一个轻量验证集输出校准问题处理：

- raw `PG-STDA-SAC-RSPA`：RMSE `0.2888`，MAE `0.2316`，RA `0.5117`；
- final `PG-STDA-SAC-RSPA-TC`：RMSE `0.1525`，MAE `0.1161`，RA `0.6706`；
- 高 RUL 区间 RMSE 约从 `0.4175` 降至 `0.1280`；
- 拟合校准映射时不使用目标测试集标签。

更复杂的 unit-start 或 elapsed-time 校准变体已经检查过，但没有采用。原因是简单的二阶 time-aware calibration 在两个迁移任务上更稳定，也更容易作为统一最终配置复现。
