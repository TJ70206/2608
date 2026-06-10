# 第二迁移实验记录：NASA Battery -> satellite_battery_sim

日期：2026-06-05

## 1. 数据集生成与质量复查

目标域数据集：`data/simulated/satellite_battery/satellite_battery_sim.csv`

生成命令：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/generate_satellite_battery_sim.py --config configs/sim_satellite_battery.yaml
```

质量检查命令：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/check_satellite_battery_sim.py
```

检查结果摘要：

| 项目 | 结果 |
|---|---:|
| rows | 1,757,352 |
| units | 100 |
| split unit counts | train 70 / val 15 / test 15 |
| fault unit counts | none 79 / resistance_step 11 / capacity_knee 10 |
| EOL reason counts | resistance_threshold 53 / capacity_threshold 47 |
| stage counts | stage0 659,304 / stage1 782,520 / stage2 315,528 |
| voltage range | 3.394 - 4.184 V |
| temperature range | 9.58 - 43.71 degC |
| soc_true range | 0.443 - 0.969 |
| DOD cycle range | 0.139 - 0.392 |

默认输入协议为 `telemetry_only`，仅使用：

```text
voltage, current, temperature
```

禁止作为默认输入的泄漏/标签相关列包括：`soc_true`, `cycle`, `capacity_ah`, `internal_resistance_ohm`, `capacity_ratio`, `resistance_ratio`, `soh`, `rul_cycles`, `normalized_rul`, `fused_damage`, `health_stage`, `fault_type`, `eol_reason`, `split` 等。

## 2. 实验设置

源域：`data/processed/nasa_battery.csv`

目标域：`data/simulated/satellite_battery/satellite_battery_sim.csv`

共同设置：

| 设置 | 值 |
|---|---|
| input features | voltage/current/temperature |
| target | normalized_rul |
| window size | 32 |
| target stride | 96 |
| epochs | 50 |
| batch size | 64 |
| test split | satellite_battery_sim test units, 15 units |

第二迁移关键配置：

| 配置 | 说明 |
|---|---|
| `satellite_battery_sim_ptcn_50e.yaml` | 目标域 P-TCN 基线 |
| `satellite_battery_sim_psa_mcd_50e.yaml` | 目标域 PSA-MCD-TCN 基线 |
| `nasa_to_satellite_battery_source_only_50e.yaml` | NASA source-only |
| `nasa_to_satellite_battery_global_mmd_50e.yaml` | NASA + global MMD |
| `nasa_to_satellite_battery_stage_lmmd_w0p003_50e.yaml` | NASA + stage LMMD，无目标监督 |
| `nasa_to_satellite_battery_stage_lmmd_targetsup_w0p003_50e.yaml` | NASA + stage LMMD + 目标监督，未调低源域权重 |
| `nasa_to_satellite_battery_stage_lmmd_targetsup_src0p05_w0p0005_50e.yaml` | NASA + stage LMMD + 目标监督，低源域权重与低 LMMD 权重 |

## 3. 50 轮结果

指标方向：

- `RMSE`, `MAE`, `NASA score`, `last_window_RMSE`：越低越好。
- `RA`, `alpha_lambda_0.5`, `alpha_lambda_0.8`：越高越好。

| 方法 | RMSE ↓ | MAE ↓ | NASA score ↓ | RA ↑ | alpha0.5 ↑ | alpha0.8 ↑ | last RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| target-only PSA-MCD-TCN | 0.193828 | 0.148007 | 41.052099 | 0.629387 | 0.354515 | 0.277027 | 0.225459 |
| NASA stage LMMD + target sup tuned | **0.189715** | **0.141262** | **39.172401** | **0.644670** | **0.408027** | 0.229730 | 0.228339 |
| target-only P-TCN | 0.232734 | 0.182578 | 48.214073 | 0.581383 | 0.227425 | 0.155405 | **0.213472** |
| NASA stage LMMD + target sup | 0.211776 | 0.161315 | 44.868932 | 0.605341 | 0.307692 | 0.111486 | 0.251162 |
| NASA stage LMMD unsup | 0.357965 | 0.291860 | 72.792792 | 0.442575 | 0.401338 | 0.189189 | 0.356899 |
| NASA source-only | 0.509407 | 0.426242 | 100.724558 | 0.198321 | 0.000000 | 0.097973 | 0.123436 |
| NASA global MMD | 0.513024 | 0.430535 | 103.134011 | 0.158597 | 0.107023 | 0.000000 | 0.245762 |

当前最优配置：

```text
configs/nasa_to_satellite_battery_stage_lmmd_targetsup_src0p05_w0p0005_50e.yaml
```

输出目录：

```text
outputs/second_transfer_50e/nasa_to_satellite_battery_stage_lmmd_targetsup_src0p05_w0p0005_50e
```

## 4. 结论

1. 第二迁移数据集生成脚本已完成，数据集质量检查通过，默认特征严格对齐比赛要求与 NASA 源域的 V/I/T 遥测量。
2. 纯 source-only 和 global MMD 在 NASA -> 卫星电池目标域上存在明显负迁移，说明轨道充放电工况与 NASA 地面循环数据差异较大。
3. 无目标监督的 stage-aware LMMD 明显优于 source-only/global MMD，但仍显著弱于 target-only，说明仅靠无监督对齐不足以覆盖该域差异。
4. 加入目标监督后，调低源域监督权重并使用目标域归一化，完整迁移模型在 RMSE、MAE、NASA score、RA、alpha0.5 上超过 target-only PSA-MCD-TCN。
5. `alpha0.8` 与 `last_window_RMSE` 尚未全面领先，后续优化应关注临近 EOL 的保守预测与高精度区间命中率。

## 5. 后续优化方向

优先级建议：

1. 对 tuned 配置做 3 个随机种子复现实验，确认 0.189715 vs 0.193828 不是单 seed 波动。
2. 在 tuned 配置上扫 `lmmd_weight = 0.0002/0.0005/0.001` 与 `source_supervised_weight = 0.02/0.05/0.10`。
3. 加入验证集偏差校准或 late-stage loss，仅用于改善 `alpha0.8` 与 `last_window_RMSE`。
4. 在技术报告中明确说明：NASA 官方原始 EOL 口径与目标域 80% 容量/1.33 倍内阻 EOL 口径不同，该差异是第二迁移任务的 domain shift 之一。
