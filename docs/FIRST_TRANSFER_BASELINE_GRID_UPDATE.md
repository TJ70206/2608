# 第一迁移必要基线与小网格补充

日期：2026-06-05

## 实验目的

本轮只补第一迁移 `XJTU-SY Bearing -> reaction_wheel_sim` 的必要证据：

- 补充 `source pretrain -> target fine-tune` 基线，回答普通监督微调能达到什么水平；
- 补充 `LSTM/GRU` 经典循环网络基线，使第一迁移与第二迁移的基线矩阵对齐；
- 对严格 UDA 主模型做少量关键参数搜索，避免把 `lmmd_weight=0.003` 固定得过早；
- 保持 50 epoch、同一 split、同一 P-SA-MCD-TCN 主干，避免不公平比较。

## 新增配置

| 配置 | 作用 |
| --- | --- |
| `configs/xjtu_to_reaction_wheel_psa_mcd_finetune_pretrain10_50e.yaml` | 源域预训练10轮，然后只用目标域标签微调50轮；不做 MMD/LMMD |
| `configs/xjtu_to_reaction_wheel_psa_mcd_stage_lmmd_pseudo_time_w0p002_50e.yaml` | 严格 UDA，`lmmd_weight=0.002` |
| `configs/xjtu_to_reaction_wheel_psa_mcd_stage_lmmd_pseudo_time_w0p005_50e.yaml` | 严格 UDA，`lmmd_weight=0.005` |
| `configs/xjtu_to_reaction_wheel_psa_mcd_stage_lmmd_pseudo_time_src0p5_w0p003_50e.yaml` | 严格 UDA，`source_supervised_weight=0.5` |
| `configs/reaction_wheel_hi_lstm_50e.yaml` | 目标域监督 LSTM 经典循环网络基线 |
| `configs/reaction_wheel_hi_gru_50e.yaml` | 目标域监督 GRU 经典循环网络基线 |
| `configs/xjtu_to_reaction_wheel_lstm_stage_lmmd_pseudo_time_w0p003_50e.yaml` | LSTM 骨干接入 strict stage LMMD pseudo-time |
| `configs/xjtu_to_reaction_wheel_gru_stage_lmmd_pseudo_time_w0p003_50e.yaml` | GRU 骨干接入 strict stage LMMD pseudo-time |

## 指标方向

RMSE、MAE、NASA score、last-window RMSE、last-5 RMSE 越低越好。
RA、alpha@0.5、alpha@0.8 越高越好。
第一主指标仍然是 RMSE/MAE，其次看临近失效的 last-window 与 last-5 RMSE。

## 结果总表

| Method | Class | RMSE | MAE | NASA score | RA | alpha@0.5 | alpha@0.8 | last RMSE | last-5 RMSE | best epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target-only GRU | supervised upper | **0.051842** | 0.040561 | 2.763724 | 0.832283 | **0.975610** | 0.400000 | 0.060358 | 0.084758 | - |
| target-only P-SA-MCD | supervised upper | 0.052078 | 0.040334 | 2.793270 | 0.830106 | 0.975610 | 0.435294 | 0.063350 | 0.083458 | - |
| target-only LSTM | supervised upper | 0.052559 | 0.041349 | 2.776801 | 0.832691 | 0.939024 | **0.470588** | 0.071593 | 0.091078 | - |
| target-only P-TCN | supervised upper | 0.052831 | 0.040171 | 2.746080 | 0.834473 | 0.951220 | 0.400000 | 0.047032 | 0.076435 | - |
| P-SA-MCD source-only | strict UDA baseline | 0.283068 | 0.241177 | 16.946109 | 0.487003 | 0.646341 | 0.000000 | 0.498073 | 0.453036 | 50 |
| P-SA-MCD global MMD | strict UDA baseline | 0.279885 | 0.239052 | 17.100530 | 0.489683 | 0.792683 | 0.000000 | 0.485398 | 0.480720 | 1 |
| P-TCN source-only | strict UDA baseline | 0.359724 | 0.291127 | 20.685917 | 0.442814 | 0.390244 | 0.047059 | 0.526221 | 0.516549 | 25 |
| P-TCN global MMD | strict UDA baseline | 0.277892 | 0.237463 | 16.977822 | 0.493362 | 0.768293 | 0.011765 | 0.468840 | 0.472334 | 4 |
| P-TCN stage LMMD pseudo-time | strict UDA baseline | 0.279218 | 0.236112 | 16.853601 | 0.494074 | 0.536585 | 0.011765 | 0.468216 | 0.470518 | 5 |
| P-SA-MCD stage LMMD pseudo-time w0.003 | strict UDA main | **0.194274** | **0.153517** | **10.676146** | **0.602411** | 0.256098 | 0.200000 | **0.213379** | **0.207936** | 48 |
| LSTM stage LMMD pseudo-time w0.003 | strict UDA recurrent | 0.265569 | 0.228936 | 16.442128 | 0.498477 | 0.756098 | 0.000000 | 0.468351 | 0.469099 | 1 |
| GRU stage LMMD pseudo-time w0.003 | strict UDA recurrent | 0.273260 | 0.229979 | 16.472413 | 0.505569 | 0.695122 | 0.011765 | 0.470795 | 0.472007 | 3 |
| P-SA-MCD stage pseudo w0.002 | strict UDA grid | 0.215424 | 0.164266 | 11.607231 | 0.582353 | 0.182927 | **0.258824** | 0.252221 | 0.216478 | 48 |
| P-SA-MCD stage pseudo w0.005 | strict UDA grid | 0.206362 | 0.163244 | 11.499034 | 0.588845 | 0.304878 | 0.117647 | 0.312755 | 0.272310 | 35 |
| P-SA-MCD stage pseudo src0.5 w0.003 | strict UDA grid | 0.210080 | 0.166877 | 11.494576 | 0.579590 | 0.256098 | 0.129412 | 0.277747 | 0.236447 | 27 |
| P-SA-MCD finetune pretrain10 | supervised adapt | 0.063204 | 0.050920 | 3.416338 | 0.808582 | 0.951220 | 0.447059 | 0.076160 | 0.099821 | 40 |

## 结论

严格无目标标签迁移设置下，当前主模型仍然应固定为：

`P-SA-MCD-TCN + stage-aware LMMD + pseudo-time target stage + lmmd_weight=0.003`

理由：

- 它在 strict UDA 组里取得最低 RMSE、MAE、NASA score、last-window RMSE、last-5 RMSE，以及最高 RA；
- 新增的 LSTM/GRU strict stage LMMD 迁移对照未超过主模型：LSTM RMSE 为 0.265569，GRU RMSE 为 0.273260，说明经典循环网络在该跨域无标签适配设定下容易被源域退化形态拉偏；
- `lmmd_weight=0.002` 的 alpha@0.8 更高，但主误差和临近失效误差更差，不适合作主配置；
- `lmmd_weight=0.005` 与 `source_supervised_weight=0.5` 都没有超过 `w0.003`；
- 旧的 warmup、stage weighting、source pretrain 等优化实验也没有超过 strict pseudo-time 主模型。

`source pretrain -> target fine-tune` 是必要强基线，但它使用目标域标签监督，不能和 strict UDA 主模型混为同一设定。它的 RMSE 为 0.063204，接近但仍弱于 target-only GRU/P-SA-MCD/P-TCN/LSTM 约 0.052 的监督上界，说明在目标域标签充足时，直接目标域监督仍是上界；而在无目标标签或弱标签迁移设置下，stage-aware LMMD 的价值更明确。

## 报告写法建议

第一迁移建议分两层汇报：

1. 严格 UDA 对比：source-only、global MMD、P-TCN variants、LSTM/GRU stage LMMD、P-SA-MCD stage LMMD pseudo-time。该层用来证明跨域无标签适配效果。
2. 监督适配参考：target-only P-SA-MCD/P-TCN/LSTM/GRU 与 source-pretrain target-fine-tune。该层作为上界与工程可用方案，不作为 strict UDA 的直接竞品。

最终不要声称 strict UDA 已接近 target-only 上界。更严谨的表述是：完整模型在无目标标签迁移组里显著优于 source-only/global MMD/P-TCN 对照，但与目标域充分监督上界仍有差距。
