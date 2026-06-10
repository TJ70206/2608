# 第一/第二迁移严格无监督同口径对比表

生成时间：2026-06-06

本表只保留严格无监督迁移结果：源域有标签监督训练，目标域训练集只用于无标签分布对齐，不使用目标域标签训练；不包含 `target-only`、`source pretrain + target fine-tune`、`target-sup`、监督上界、集成模型和调参变体。第二迁移的 stage LMMD 已改为 `target_stage_source: time_progress`，即目标域阶段伪标签来自 unit 内时间进度，而不是目标 RUL 标签。

指标方向：`RMSE`、`MAE`、`NASA`、`LastRMSE`、`Last5RMSE` 越低越好；`RA`、`a0.5`、`a0.8` 越高越好。两张表的方法顺序完全一致，便于逐行对照。

## 第一迁移：XJTU-SY 轴承源域 -> reaction_wheel_sim 反作用飞轮目标域

| Method | RMSE ↓ | MAE ↓ | NASA ↓ | RA ↑ | a0.5 ↑ | a0.8 ↑ | LastRMSE ↓ | Last5RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-SA-MCD source-only | 0.2831 | 0.2412 | 16.9461 | 0.4870 | 0.6463 | 0.0000 | 0.4981 | 0.4530 |
| P-SA-MCD global MMD | 0.2799 | 0.2391 | 17.1005 | 0.4897 | 0.7927 | 0.0000 | 0.4854 | 0.4807 |
| **P-SA-MCD stage LMMD + pseudo-time** | **0.1943** | **0.1535** | **10.6761** | **0.6024** | 0.2561 | **0.2000** | **0.2134** | **0.2079** |
| P-TCN source-only | 0.3597 | 0.2911 | 20.6859 | 0.4428 | 0.3902 | 0.0471 | 0.5262 | 0.5165 |
| P-TCN global MMD | 0.2779 | 0.2375 | 16.9778 | 0.4934 | 0.7683 | 0.0118 | 0.4688 | 0.4723 |
| P-TCN stage LMMD + pseudo-time | 0.2792 | 0.2361 | 16.8536 | 0.4941 | 0.5366 | 0.0118 | 0.4682 | 0.4705 |
| LSTM stage LMMD + pseudo-time | 0.2656 | 0.2289 | 16.4421 | 0.4985 | 0.7561 | 0.0000 | 0.4684 | 0.4691 |
| GRU stage LMMD + pseudo-time | 0.2733 | 0.2300 | 16.4724 | 0.5056 | 0.6951 | 0.0118 | 0.4708 | 0.4720 |
| Transformer stage LMMD + pseudo-time | 0.2729 | 0.2345 | 16.7379 | 0.4957 | **0.9268** | 0.0000 | 0.4815 | 0.4756 |
| AD-TCN-MSC-DIM stage LMMD + pseudo-time | 0.2527 | 0.1998 | 14.5448 | 0.5327 | 0.2561 | 0.1294 | 0.3669 | 0.3843 |

结论：第一迁移严格无监督口径下，`P-SA-MCD stage LMMD + pseudo-time` 是当前综合最强结果，RMSE、MAE、NASA、RA、LastRMSE、Last5RMSE 均优于同类 source-only/global MMD 和其他基础序列模型。

## 第二迁移：NASA Battery 源域 -> satellite_battery_sim 卫星电池目标域

| Method | RMSE ↓ | MAE ↓ | NASA ↓ | RA ↑ | a0.5 ↑ | a0.8 ↑ | LastRMSE ↓ | Last5RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-SA-MCD source-only | 0.5094 | 0.4262 | 100.7246 | 0.1983 | 0.0000 | 0.0980 | 0.1234 | 0.1228 |
| P-SA-MCD global MMD | 0.5130 | 0.4305 | 103.1340 | 0.1586 | 0.1070 | 0.0000 | 0.2458 | 0.2444 |
| **P-SA-MCD stage LMMD + pseudo-time** | 0.3571 | 0.2913 | 72.6033 | 0.4435 | 0.3913 | **0.1892** | 0.3550 | 0.3541 |
| P-TCN source-only | 0.5842 | 0.5072 | 119.6561 | 0.0665 | 0.0000 | 0.0000 | **0.0772** | **0.0781** |
| P-TCN global MMD | 0.4569 | 0.3748 | 89.0250 | 0.3177 | 0.0000 | 0.2905 | 0.1781 | 0.1783 |
| P-TCN stage LMMD + pseudo-time | 0.3877 | 0.3144 | 77.2155 | 0.3963 | 0.3211 | 0.1588 | 0.3177 | 0.3165 |
| LSTM stage LMMD + pseudo-time | 0.3513 | 0.2863 | 72.2647 | 0.4475 | 0.4548 | 0.1149 | 0.3686 | 0.3680 |
| GRU stage LMMD + pseudo-time | 0.3986 | 0.3228 | 82.1169 | 0.3700 | 0.1906 | 0.0811 | 0.3845 | 0.3847 |
| Transformer stage LMMD + pseudo-time | 0.3604 | 0.2914 | 77.2884 | 0.4341 | 0.2843 | 0.1453 | 0.4731 | 0.4775 |
| AD-TCN-MSC-DIM stage LMMD + pseudo-time | **0.3407** | **0.2789** | **71.7953** | **0.4564** | **0.4783** | 0.1149 | 0.4225 | 0.4202 |

结论：第二迁移严格无监督口径下，`P-SA-MCD stage LMMD + pseudo-time` 明显优于 P-SA-MCD source-only/global MMD，但当前最低 RMSE 来自 `AD-TCN-MSC-DIM stage LMMD + pseudo-time`。因此第二迁移不能写成 P-SA-MCD 完整模型全指标最优；更严谨的表述是：P-SA-MCD 是统一主方法且在第一迁移最优、第二迁移显著提升同架构迁移基线；AD-TCN-MSC-DIM 是第二迁移上的强竞争基线。

## 复现实验文件

第一迁移结果文件：

| Method | Metrics file |
|---|---|
| P-SA-MCD source-only | `outputs/first_transfer_50e/xjtu_to_reaction_wheel_source_only_50e/transfer_metrics.json` |
| P-SA-MCD global MMD | `outputs/first_transfer_50e/xjtu_to_reaction_wheel_global_mmd_50e/transfer_metrics.json` |
| P-SA-MCD stage LMMD + pseudo-time | `outputs/first_transfer_strict_50e/psa_mcd_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| P-TCN source-only | `outputs/first_transfer_strict_50e/ptcn_source_only_50e/transfer_metrics.json` |
| P-TCN global MMD | `outputs/first_transfer_strict_50e/ptcn_global_mmd_50e/transfer_metrics.json` |
| P-TCN stage LMMD + pseudo-time | `outputs/first_transfer_strict_50e/ptcn_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| LSTM stage LMMD + pseudo-time | `outputs/first_transfer_strict_50e/lstm_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| GRU stage LMMD + pseudo-time | `outputs/first_transfer_strict_50e/gru_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| Transformer stage LMMD + pseudo-time | `outputs/first_transfer_strict_50e/transformer_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| AD-TCN-MSC-DIM stage LMMD + pseudo-time | `outputs/first_transfer_strict_50e/ad_tcn_mscdim_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |

第二迁移结果文件：

| Method | Metrics file |
|---|---|
| P-SA-MCD source-only | `outputs/second_transfer_50e/nasa_to_satellite_battery_source_only_50e/transfer_metrics.json` |
| P-SA-MCD global MMD | `outputs/second_transfer_50e/nasa_to_satellite_battery_global_mmd_50e/transfer_metrics.json` |
| P-SA-MCD stage LMMD + pseudo-time | `outputs/second_transfer_strict_50e/psa_mcd_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| P-TCN source-only | `outputs/second_transfer_strict_50e/ptcn_source_only_50e/transfer_metrics.json` |
| P-TCN global MMD | `outputs/second_transfer_strict_50e/ptcn_global_mmd_50e/transfer_metrics.json` |
| P-TCN stage LMMD + pseudo-time | `outputs/second_transfer_strict_50e/ptcn_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| LSTM stage LMMD + pseudo-time | `outputs/second_transfer_strict_50e/lstm_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| GRU stage LMMD + pseudo-time | `outputs/second_transfer_strict_50e/gru_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| Transformer stage LMMD + pseudo-time | `outputs/second_transfer_strict_50e/transformer_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
| AD-TCN-MSC-DIM stage LMMD + pseudo-time | `outputs/second_transfer_strict_50e/ad_tcn_mscdim_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json` |
