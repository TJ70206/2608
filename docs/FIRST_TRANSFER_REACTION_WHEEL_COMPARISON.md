# 第一迁移对照实验：XJTU-SY Bearing -> reaction_wheel_sim

日期：2026-06-05

## 目的

当前只验证第一个迁移任务：从公开轴承退化数据 XJTU-SY Bearing 迁移到自建航天器反作用飞轮仿真数据 `reaction_wheel_sim`。第二个迁移任务暂不展开。

本轮实验用于确认：

- 目标域 HI 数据链路是否可训练；
- 源域直接迁移是否存在明显域差异；
- 当前工程完整版 v1，即 `P-SA-MCD-TCN + stage-aware LMMD`，相对基础迁移方法是否有效；
- 50 epoch 训练下，`stage-aware LMMD` 的 `lmmd_weight` 是否需要调整。

## 指标方向

| 指标 | 方向 | 作用 |
| --- | --- | --- |
| RMSE | 越低越好 | 主指标，衡量整体 RUL 回归误差 |
| MAE | 越低越好 | 主指标，衡量平均绝对误差 |
| RA | 越高越好 | 相对精度，越接近 1 越好 |
| alpha@0.5 / alpha@0.8 | 越高越好 | 误差落入容忍带的比例 |
| last-window RMSE | 越低越好 | 最临近失效窗口的误差 |
| last-5 RMSE | 越低越好 | 临近失效前 5 个窗口的平均误差 |
| NASA score | 越低越好 | 非对称惩罚分数；当前 RUL 已归一化，主要作辅助参考 |

第一迁移方法内部优先看 RMSE/MAE，其次看 last-window 与 last-5 RMSE。`alpha` 指标受归一化 RUL 接近 0 时的相对误差影响较大，因此只作为辅助指标。

## 5 epoch 快速筛查结果

| 方法 | RMSE | MAE | RA | alpha@0.5 | alpha@0.8 | last RMSE | last-5 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HI P-TCN target-only | 0.061178 | 0.048538 | 0.810528 | 0.902439 | 0.341176 | 0.054627 | 0.079435 |
| HI P-SA-MCD target-only | 0.058116 | 0.045311 | 0.822568 | 0.914634 | 0.494118 | 0.072517 | 0.094413 |
| XJTU source-only | 0.391747 | 0.316812 | 0.434003 | 0.207317 | 0.000000 | 0.740696 | 0.693150 |
| XJTU global MMD | 0.296638 | 0.248033 | 0.478760 | 0.573171 | 0.000000 | 0.524677 | 0.493750 |
| XJTU stage-aware LMMD | 0.264148 | 0.220596 | 0.512430 | 0.524390 | 0.058824 | 0.488055 | 0.469576 |

5 epoch 只证明链路可跑通、迁移方向有效，不能作为最终效果。

## 50 epoch 主对照结果

| 方法 | RMSE | MAE | RA | alpha@0.5 | alpha@0.8 | last RMSE | last-5 RMSE | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HI P-TCN target-only | 0.052831 | 0.040171 | 0.834473 | 0.951220 | 0.400000 | 0.047032 | 0.076435 | - |
| HI P-SA-MCD target-only | 0.052078 | 0.040334 | 0.830106 | 0.975610 | 0.435294 | 0.063350 | 0.083458 | - |
| XJTU source-only | 0.283068 | 0.241177 | 0.487003 | 0.646341 | 0.000000 | 0.498073 | 0.453036 | 50 |
| XJTU global MMD | 0.279885 | 0.239052 | 0.489683 | 0.792683 | 0.000000 | 0.485398 | 0.480720 | 1 |
| XJTU stage-aware LMMD, weight=0.01 | 0.204792 | 0.162765 | 0.584080 | 0.292683 | 0.117647 | 0.269843 | 0.242943 | 27 |

输出目录：`outputs/first_transfer_50e/`。

判断：

1. 50 epoch 明显优于 5 epoch，之前的 5 epoch 迁移实验未充分收敛。
2. 目标域监督训练仍是上限，RMSE 约 0.052。
3. `source-only` 和 `global MMD` 都在 0.28 左右，直接迁移和全局对齐仍不足。
4. `stage-aware LMMD` 在迁移组中明显最好，RMSE 降到 0.205，说明当前完整版 v1 的阶段对齐有实际收益。
5. `global MMD` 的 best epoch 为 1，说明当前 `lmmd_weight=0.01` 下全局 MMD 长训练稳定性较差。

## stage-aware LMMD 权重网格

固定 50 epoch、seed=202608，只调整 `lmmd_weight`：

| lmmd_weight | RMSE | MAE | RA | alpha@0.5 | alpha@0.8 | last RMSE | last-5 RMSE | best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0.230200 | 0.183228 | 0.553316 | 0.207317 | 0.176471 | 0.218460 | 0.245175 | 45 |
| 0.003 | 0.204628 | 0.160294 | 0.595049 | 0.207317 | 0.223529 | 0.187951 | 0.183832 | 46 |
| 0.010 | 0.204792 | 0.162765 | 0.584080 | 0.292683 | 0.117647 | 0.269843 | 0.242943 | 27 |
| 0.030 | 0.216327 | 0.168580 | 0.576247 | 0.195122 | 0.141176 | 0.210513 | 0.207444 | 46 |
| 0.100 | 0.220565 | 0.174415 | 0.557634 | 0.341463 | 0.058824 | 0.237312 | 0.240821 | 46 |

当前最优候选为 `lmmd_weight=0.003`：

- 整体 RMSE 最低：0.204628；
- MAE 最低：0.160294；
- RA 最高：0.595049；
- last-window RMSE 最低：0.187951；
- last-5 RMSE 最低：0.183832。

正式复现实验配置：`configs/xjtu_to_reaction_wheel_stage_lmmd_w0p003_50e.yaml`。

## 严格迁移对照：pseudo-time target stage

为避免把目标域真实 RUL 分段用于 target stage 对齐，新增 `target_stage_source: time_progress`。该设置只使用目标域 HI 序列的时间进度生成 pseudo-stage，不使用目标域 RUL 标签参与迁移对齐。

严格版完整模型配置：`configs/xjtu_to_reaction_wheel_psa_mcd_stage_lmmd_pseudo_time_w0p003_50e.yaml`。

50 epoch、seed=202608 的严格迁移对照如下：

| 方法 | RMSE | MAE | RA | alpha@0.5 | alpha@0.8 | last RMSE | last-5 RMSE | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P-SA-MCD source-only | 0.283068 | 0.241177 | 0.487003 | 0.646341 | 0.000000 | 0.498073 | 0.453036 | 50 |
| P-SA-MCD global MMD | 0.279885 | 0.239052 | 0.489683 | 0.792683 | 0.000000 | 0.485398 | 0.480720 | 1 |
| P-SA-MCD stage LMMD pseudo-time | 0.194274 | 0.153517 | 0.602411 | 0.256098 | 0.200000 | 0.213379 | 0.207936 | 48 |
| P-TCN source-only | 0.359724 | 0.291127 | 0.442814 | 0.390244 | 0.047059 | 0.526221 | 0.516549 | 25 |
| P-TCN global MMD | 0.277892 | 0.237463 | 0.493362 | 0.768293 | 0.011765 | 0.468840 | 0.472334 | 4 |
| P-TCN stage LMMD pseudo-time | 0.279218 | 0.236112 | 0.494074 | 0.536585 | 0.011765 | 0.468216 | 0.470518 | 5 |

严格单 seed 结论：当前完整模型 `P-SA-MCD + stage-aware LMMD + pseudo-time target stage` 在主指标 RMSE/MAE、RA、last-window RMSE、last-5 RMSE 上均为迁移组最佳。`alpha@0.5` 不是最高，但该指标在归一化 RUL 接近 0 时受相对误差影响较大，作为辅助指标，不作为第一主判据。

## 3 seed 稳定性检查

对最强完整模型、最强 P-TCN global MMD 基线、同结构 P-SA-MCD global MMD 基线补跑 3 seeds：202608、202609、202610。

| 方法 | RMSE mean | RMSE std | MAE mean | MAE std | RA mean | last RMSE mean | last-5 RMSE mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full P-SA-MCD stage LMMD pseudo-time | 0.227394 | 0.040405 | 0.179558 | 0.034513 | 0.562000 | 0.223145 | 0.220934 |
| P-TCN global MMD | 0.284899 | 0.009955 | 0.241882 | 0.005527 | 0.492014 | 0.481417 | 0.462290 |
| P-SA-MCD global MMD | 0.276324 | 0.003097 | 0.236780 | 0.002193 | 0.494515 | 0.468144 | 0.466123 |

3 seed 结论：完整模型的均值 RMSE、MAE、RA、临近失效 RMSE 均优于两个强基线。完整模型方差更大，说明后续还应做训练稳定性优化，但目前平均效果仍明显最好。

## 严谨性备注

早期 `stage-aware LMMD` 诊断实验使用 `WindowedTimeSeriesDataset` 生成的阶段标签；该阶段标签来自 RUL 分段。因此，早期诊断版不是严格无监督设置，更准确地说是“带目标域阶段信息的迁移诊断”或半监督阶段对齐。

严格迁移主结论应使用 `target_stage_source: time_progress` 的 pseudo-stage 版本。该版本不依赖目标域真实 RUL 生成 target stage，更适合作为比赛报告中的第一迁移主结果。后续若要进一步增强 UDA 严谨性，可改进为基于 HI 单调趋势或模型伪标签的 target pseudo-stage。

## 下一步

第一迁移下一步不建议立即进入第二迁移，应先完成少标签目标域评估：

1. 构造 target labeled units = 3/5/10/15 的 few-label 目标域训练配置；
2. 保留目标域 train units 的无标签特征用于 MMD/LMMD 对齐；
3. 用 `target_stage_source: time_progress` 与 `lmmd_weight=0.003` 作为当前完整版主配置；
4. 至少运行 3 个 seeds，报告均值和标准差；
5. 继续优化训练稳定性，重点降低完整模型的跨 seed 方差。
