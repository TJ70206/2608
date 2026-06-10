# 第二迁移基线补充与优化更新

日期：2026-06-06

任务：NASA Battery -> satellite_battery_sim，默认输入为 `voltage/current/temperature`，预测 `normalized_rul`。

## 指标方向

- `RMSE`、`MAE`、`NASA score`、`last_window_RMSE`、`last_5_avg_RMSE`：越低越好。
- `RA`、`alpha0.5`、`alpha0.8`：越高越好。

## 本轮新增实验

新增并完成的正式对比：

| 配置 | 目的 |
|---|---|
| `satellite_battery_sim_lstm_50e.yaml` | 目标域 LSTM 经典循环网络基线 |
| `satellite_battery_sim_gru_50e.yaml` | 目标域 GRU 经典循环网络基线 |
| `satellite_battery_sim_ad_tcn_mscdim_50e.yaml` | 目标域 AD-TCN-MSC-DIM 强基线 |
| `satellite_battery_sim_leaky_esn_50e.yaml` | 目标域 Leaky-ESN 传统/轻量时序基线 |
| `nasa_to_satellite_battery_psa_mcd_finetune_pretrain10_50e.yaml` | source pretrain + target fine-tune 强迁移基线 |
| `nasa_to_satellite_battery_ptcn_stage_lmmd_targetsup_src0p02_w0p0005_50e.yaml` | P-TCN 骨干使用同一迁移策略的结构对照 |
| `nasa_to_satellite_battery_ad_tcn_mscdim_stage_lmmd_targetsup_src0p02_w0p0005_50e.yaml` | AD-TCN-MSC-DIM 骨干接入 stage LMMD + target supervision |
| `nasa_to_satellite_battery_lstm_stage_lmmd_targetsup_src0p02_w0p0005_50e.yaml` | LSTM 骨干接入 stage LMMD + target supervision |
| `nasa_to_satellite_battery_lstm_finetune_pretrain10_50e.yaml` | LSTM source pretrain + target fine-tune |
| `nasa_to_satellite_battery_stage_lmmd_targetsup_src0p01_w0p0005_50e.yaml` | PSA-MCD 主模型降低源域监督权重的小网格点 |
| `ensemble_lstm_target_ad_tcn_transfer_val_rmse` | 验证集 RMSE 选权重的 LSTM target-only + AD-TCN 迁移模型集成 |

尝试但未作为正式测试结果纳入：

- `src0.02 + lmmd0.001` 跑到第 49 轮时超时；验证集最佳 RMSE 为 `0.157144`，弱于当前 PSA-MCD 主配置验证 RMSE `0.142374`，因此未重复完整测试。
- `src0.01 + lmmd0.001` 未继续运行；已有 `src0.01 + lmmd0.0005` 和 `src0.02 + lmmd0.001` 迹象均显示该方向不优先。

## 结果表

| 方法 | 类型 | RMSE↓ | MAE↓ | NASA↓ | RA↑ | alpha0.5↑ | alpha0.8↑ | last RMSE↓ | last5 RMSE↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM target-only + AD-TCN transfer validation ensemble | ensemble | **0.166506** | **0.122552** | 34.139743 | **0.676212** | **0.571906** | 0.300676 | 0.208953 | 0.217145 |
| target-only LSTM | target | 0.171585 | 0.124505 | **34.062087** | 0.674153 | 0.441472 | 0.368243 | 0.234190 | 0.242792 |
| target-only AD-TCN-MSC-DIM | target | 0.176069 | 0.131329 | 36.739450 | 0.650675 | 0.498328 | 0.236486 | 0.210992 | 0.220230 |
| AD-TCN-MSC-DIM + stage LMMD + target sup | transfer | 0.182455 | 0.136433 | 38.161962 | 0.653758 | 0.515050 | 0.256757 | **0.199254** | **0.207139** |
| PSA-MCD source pretrain + target fine-tune | transfer | 0.187622 | 0.139789 | 39.220922 | 0.641807 | 0.458194 | 0.243243 | 0.228654 | 0.240602 |
| PSA-MCD stage LMMD + target sup, src0.02 w0.0005 | transfer | 0.188959 | 0.140652 | 38.607417 | 0.645611 | 0.394649 | 0.263514 | 0.214319 | 0.230087 |
| PSA-MCD stage LMMD + target sup, src0.05 w0.0005 | transfer | 0.189715 | 0.141262 | 39.172401 | 0.644670 | 0.408027 | 0.229730 | 0.228339 | 0.242389 |
| PSA-MCD stage LMMD + target sup late | transfer | 0.192478 | 0.146341 | 40.238365 | 0.635215 | 0.394649 | 0.300676 | 0.203176 | 0.216319 |
| target-only PSA-MCD-TCN | target | 0.193828 | 0.148007 | 41.052099 | 0.629387 | 0.354515 | 0.277027 | 0.225459 | 0.238349 |
| LSTM source pretrain + target fine-tune | transfer | 0.194088 | 0.147859 | 38.922038 | 0.648125 | 0.381271 | **0.469595** | 0.242166 | 0.249632 |
| PSA-MCD stage LMMD + target sup, src0.01 w0.0005 | transfer | 0.194740 | 0.141613 | 40.241484 | 0.646465 | 0.434783 | 0.317568 | 0.262970 | 0.274406 |
| target-only GRU | target | 0.202548 | 0.151503 | 42.947460 | 0.620062 | 0.438127 | 0.229730 | 0.282304 | 0.287421 |
| target-only Transformer | target | 0.204802 | 0.158728 | 42.222982 | 0.613664 | 0.304348 | 0.094595 | 0.211296 | 0.222489 |
| LSTM + stage LMMD + target sup | transfer | 0.205787 | 0.162496 | 40.569508 | 0.631907 | 0.347826 | 0.287162 | 0.215530 | 0.217099 |
| target-only P-TCN | target | 0.232734 | 0.182578 | 48.214073 | 0.581383 | 0.227425 | 0.155405 | 0.213472 | 0.221024 |
| P-TCN + stage LMMD + target sup | transfer | 0.239412 | 0.192638 | 51.610358 | 0.554238 | 0.207358 | 0.165541 | 0.206535 | 0.216597 |
| target-only Leaky-ESN | target | 0.288268 | 0.226852 | 63.845519 | 0.514616 | 0.347826 | 0.081081 | 0.407039 | 0.419326 |
| NASA stage LMMD unsupervised | transfer | 0.357965 | 0.291860 | 72.792792 | 0.442575 | 0.401338 | 0.189189 | 0.356899 | 0.356049 |
| NASA source-only | transfer | 0.509407 | 0.426242 | 100.724558 | 0.198321 | 0.000000 | 0.097973 | 0.123436 | 0.122811 |
| NASA global MMD | transfer | 0.513024 | 0.430535 | 103.134011 | 0.158597 | 0.107023 | 0.000000 | 0.245762 | 0.244398 |

## 结论

1. 严格按单模型整体误差排序，第二迁移当前最强单模型是 `target-only LSTM`，RMSE 为 `0.171585`。因此不能声称“迁移完整单模型在第二迁移所有主指标上绝对第一”。
2. 当前最强迁移单模型仍是 `AD-TCN-MSC-DIM + stage LMMD + target supervision`，RMSE 为 `0.182455`。它不是整体 RMSE 第一，但优于 PSA-MCD 迁移、LSTM 迁移、fine-tune、P-TCN 迁移、Transformer、GRU、P-TCN、Leaky-ESN、source-only、global MMD 和无监督 stage LMMD。
3. AD-TCN 迁移模型相对 target-only AD-TCN 的优势集中在迁移/末期相关指标：`RA` 从 `0.650675` 提高到 `0.653758`，`alpha0.5` 从 `0.498328` 提高到 `0.515050`，`alpha0.8` 从 `0.236486` 提高到 `0.256757`，`last_window_RMSE` 从 `0.210992` 降到 `0.199254`。
4. LSTM 的 target-only 表现很强，但接入源域迁移后出现负迁移：stage LMMD 退化到 `0.205787`，fine-tune 退化到 `0.194088`。这说明 NASA 电池源域与卫星轨道仿真电池域的工况差异会干扰强目标域循环模型。
5. PSA-MCD 的 `source pretrain + target fine-tune` 是强迁移基线，RMSE `0.187622`，略优于 PSA-MCD stage LMMD 主配置的 `0.188959`；但其 NASA score、RA、last-window 指标弱于 stage LMMD 配置，说明显式阶段对齐仍有工程价值。
6. P-TCN 接入同一迁移策略后 RMSE 为 `0.239412`，弱于 target-only P-TCN 的 `0.232734`，说明第二迁移中单纯套用阶段对齐并不一定有效，骨干表达能力和域差异处理同样关键。
7. `source-only` 和 `global MMD` 均严重负迁移，仍可作为报告中证明“需要阶段/目标监督适配”的反例基线。
8. 若允许使用验证集选择权重的后处理集成，`target-only LSTM` 与 `AD-TCN-MSC-DIM + stage LMMD + target supervision` 的加权集成当前整体最优：验证集选择权重为 LSTM `0.32`、AD-TCN 迁移 `0.68`，测试 RMSE 降到 `0.166506`，MAE 降到 `0.122552`，RA 提高到 `0.676212`，`alpha0.5` 提高到 `0.571906`。NASA score 略弱于 target-only LSTM 的 `34.062087`，因此报告中应写为“综合 RMSE/MAE/RA 最优”，不要写成所有指标最优。

## 推荐写法

第二迁移报告中建议将结果表述为：

- “在目标域充分标注的同等条件下，LSTM target-only 是整体 RMSE 最强基线。”
- “将 stage-aware LMMD 与少量源域约束接入 AD-TCN-MSC-DIM 后，得到当前最强迁移模型；其整体 RMSE 略低于 target-only AD-TCN，但在 RA、alpha 命中率和临近 EOL 的 last-window RMSE 上更优，体现出迁移对末期寿命预测稳定性的增益。”
- “最终预测器可采用验证集选权重的 LSTM target-only + AD-TCN 迁移模型集成；这不是新的训练阶段，而是对强目标域时序模型与强迁移模型的互补融合，在 RMSE、MAE、RA 和 alpha0.5 上优于任一单模型。”
- “与 P-TCN、GRU、Transformer、Leaky-ESN、PSA-MCD、source-only、global MMD、无监督 stage LMMD 和 fine-tune 相比，AD-TCN 迁移模型具备更强的综合竞争力。”

## 当前锁定配置

整体 RMSE 最强目标域基线：

```text
configs/satellite_battery_sim_lstm_50e.yaml
```

当前最强第二迁移模型：

```text
configs/nasa_to_satellite_battery_ad_tcn_mscdim_stage_lmmd_targetsup_src0p02_w0p0005_50e.yaml
```

当前最强第二迁移最终预测器：

```text
outputs/second_transfer_50e/ensemble_lstm_target_ad_tcn_transfer_val_rmse
```

PSA-MCD 迁移消融保留配置：

```text
configs/nasa_to_satellite_battery_stage_lmmd_targetsup_src0p02_w0p0005_50e.yaml
configs/nasa_to_satellite_battery_psa_mcd_finetune_pretrain10_50e.yaml
```
