# 第一迁移优化实验记录

日期：2026-06-05

## 目的

在已经完成的严格第一迁移主配置基础上，测试三个轻量优化方向是否能进一步提升：

1. source pretrain；
2. LMMD warm-up；
3. stage-weighted LMMD。

本轮只优化第一迁移 `XJTU-SY Bearing -> reaction_wheel_sim`，不进入第二迁移。

## 新增代码能力

新增能力均为可选配置，默认关闭，不影响旧实验复现：

- `weighted_mmd_loss`：支持带样本权重的 MMD；
- `stage_weights`：支持对不同退化阶段的 LMMD 损失加权；
- `lmmd_warmup_epochs`：支持迁移对齐权重按 epoch 线性 warm-up；
- `source_pretrain_epochs`：支持正式迁移前只用源域 RUL 监督预训练。

对应代码：

- `src/xa202608/losses.py`
- `scripts/train_transfer.py`

对应测试：

- `tests/test_first_transfer_experiment.py`

## 单 seed 消融结果

固定 seed = 202608、50 epoch、严格 pseudo-time target stage。

| 配置 | RMSE | MAE | RA | alpha@0.5 | alpha@0.8 | last RMSE | last-5 RMSE | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 当前主配置：stage LMMD pseudo-time | 0.194274 | 0.153517 | 0.602411 | 0.256098 | 0.200000 | 0.213379 | 0.207936 | 48 |
| source pretrain 10 + warm-up 15 + stage weights [0.5,1.0,2.0] | 0.244391 | 0.188190 | 0.549226 | 0.134146 | 0.223529 | 0.236141 | 0.249526 | 49 |
| warm-up 10 only | 0.214753 | 0.165649 | 0.588186 | 0.304878 | 0.270588 | 0.256102 | 0.234826 | 50 |
| stage weights [0.75,1.0,1.5] only | 0.231712 | 0.179787 | 0.558617 | 0.256098 | 0.188235 | 0.234608 | 0.226466 | 48 |
| stage weights [0.75,1.0,1.5] + warm-up 10 | 0.219089 | 0.170526 | 0.576334 | 0.219512 | 0.223529 | 0.277118 | 0.251776 | 37 |

## 判断

当前主配置仍然是第一迁移的最优主结果。

本轮优化中，`warm-up 10 only` 的 alpha 指标较高，但主指标 RMSE/MAE、RA、临近失效 RMSE 都没有超过当前主配置。由于第一迁移主指标优先级是 RMSE/MAE 和临近失效误差，因此不能把 warm-up 版本作为主模型。

`source pretrain 10` 明显不适合作为当前默认策略。源域预训练阶段 target validation RMSE 后期上升，说明模型被源域退化模式拉偏，存在负迁移风险。

简单 `stage_weights` 也没有带来收益。当前 target pseudo-stage 来自 time progress，直接增加晚期阶段权重会削弱整体分布对齐，导致整体 RMSE 和临近失效指标都变差。

## 当前锁定方案

第一迁移当前应锁定：

- 配置：`configs/xjtu_to_reaction_wheel_psa_mcd_stage_lmmd_pseudo_time_w0p003_50e.yaml`
- 方法：`P-SA-MCD-TCN + stage-aware LMMD + pseudo-time target stage`
- 输出：`outputs/first_transfer_strict_50e/psa_mcd_stage_lmmd_pseudo_time_w0p003_50e`

主报告应使用该配置及其 3 seed 结果，不采用本轮新增消融作为主结果。

## 后续建议

第一迁移已经可以进入“整理报告与稳定性确认”阶段。若继续优化，不建议再做简单 warm-up 或手工 stage weighting；更有价值的方向是：

1. 基于 source-pretrained pseudo-RUL 的动态 target stage，而不是固定 time-progress stage；
2. 真正实现 shared/private feature separation，使 `SA-FS-LMMD` 与规划一致；
3. 启用 Conformal Prediction 做预测区间，不以降低 RMSE 为目标；
4. 扩展 5 seed 或 10 seed 稳定性评估，确认当前主配置均值优势。

