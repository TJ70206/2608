# C-MAPSS FD001 基准结果与参考论文对比

## 当前实验

配置：`configs/cmapss_fd001_ptcn.yaml`

| 项目 | 当前设置 |
|---|---:|
| 模型 | P-TCN baseline |
| 数据集 | C-MAPSS FD001 |
| 窗口长度 | 30 |
| RUL 上限 | 125 |
| 输入特征数 | 17 |
| Drop sensors | S1, S5, S6, S10, S16, S18, S19 |
| 归一化 | train-only z-score |
| Epoch | 30 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Optimizer | AdamW |
| 参数量 | 128,897 |

当前 P-TCN baseline 结果：

| 口径 | RMSE | MAE | NASA Score | RA |
|---|---:|---:|---:|---:|
| 所有测试窗口 | 16.79 | 12.57 | 84532.35 | 0.8546 |
| 每台发动机最后窗口 | 16.47 | 12.28 | 572.50 | 0.7981 |

说明：C-MAPSS 论文表格通常报告每台测试发动机最后检测点指标，因此横向对比应优先使用“最后窗口”口径。所有测试窗口指标适合分析曲线质量和全生命周期误差，不适合直接和多数论文表格的 Score 对比。

## 参考论文指标

### DVGTformer（参考论文 1）

| 方法 | FD001 RMSE | FD001 Score/SF | 备注 |
|---|---:|---:|---|
| SVGTformer | 10.79 | 153.44 | 单视角空间结构依赖，FD001/FD003 更强 |
| DVGTformer | 11.33 | 179.75 | 双视图图 Transformer |
| HAGCN | 11.93 | 222.3 | 图卷积对比方法 |
| DAST | 11.43 | 203.15 | Transformer 类方法 |
| STGCN | 12.76 | N/A | 时空图卷积 |

DVGTformer 关键参数：

| 参数 | 论文设置 |
|---|---:|
| 输入尺寸 | 30 × 17 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Dropout | 0.1 / regressor dropout 0.5 |
| 结构 | 6 个 DVGTformer block，4-head attention |

与我们相比：窗口长度、输入特征数、学习率、dropout 基本对齐；batch size 我们更大；模型复杂度和结构表达能力明显弱于 DVGTformer。

### 预处理驱动 TCN（参考论文 4）

| 方法 | FD001 RMSE | FD001 Score |
|---|---:|---:|
| 普通 TCN | 13.51 | 266 |
| ATCN | 11.48 | 194.25 |
| 论文 Ours mean | 9.26 | 376.67 |

关键实验设置：

| 参数 | 论文设置 |
|---|---:|
| Batch size | 8 |
| Learning rate | 0.01 |
| Epoch | 1000 |
| Early stop patience | 40 |
| 重复次数 | 10 independent runs |
| 归一化 | z-score |
| RUL 上限 | 125 |

与我们相比：RUL clip、z-score、dense supervision 思路一致；但论文使用 1000 epoch + early stopping + 10 次重复，训练预算远高于我们当前 30 epoch 快速验证。因此当前不能期望达到其最优 RMSE。

2026-05-26 更新：已将参考论文 4 的完整序列 dense supervision 思路加入 `P-SA-MCD-TCN`，并额外加入末端 5 点加权以对齐 C-MAPSS last-window 评价口径。纯 full-sequence dense training 明显改善 all-window RMSE，但 last-window RMSE 变差；加入 tail weighting 后同时获得更好的 dense 曲线拟合与 capped last-window 结果。

| 我们的配置 | 口径 | FD001 RMSE | FD001 Score | 备注 |
|---|---|---:|---:|---|
| `cmapss_fd001_psa_mcd_aligned_pseudo_match` | capped last-window | 12.7011 | 293.23 | 历史 capped 最佳 |
| `cmapss_fd001_psa_mcd_paper4_fullseq_capped_99e` | capped last-window | 17.0678 | 371.59 | 纯 dense/full-sequence，不采用为最终 |
| `cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e` | all-window dense | 9.3132 | 19082.12 | 接近论文 4 的 dense RMSE 9.26，但 Score 不宜与 last-window Score 混比 |
| `cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e` | capped last-window | 12.3669 | 202.71 | 当前最佳 paper-style capped 结果 |
| `cmapss_fd001_psa_mcd_paper_strict_80e_raw` | raw last-window | 13.8777 | 321.17 | 当前最佳 strict raw RMSE |
| `cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e` raw re-eval | raw last-window | 14.1133 | 260.87 | raw Score 更优，但 raw RMSE 未超过旧 best |

因此，正式报告应明确区分 `all-window dense`、`capped last-window`、`raw last-window` 三个口径。参考论文 4 的 `9.26` 更可能对应 dense/full-sequence 曲线拟合口径或至少没有清楚声明 last-window 口径；我们可以报告 `all-window RMSE=9.3132` 作为曲线拟合能力，但主表横向对比仍优先使用 `capped last-window RMSE=12.3669 / Score=202.71`。

### MSCDIM/动态 dilation 类方法（参考论文 3）

| 方法 | FD001 RMSE | FD001 Score |
|---|---:|---:|
| SA-MSTCN-GFA | 10.62 | 165.00 |
| ATCN | 11.18 | 194.25 |
| 论文 Ours | 11.39 | 170.52 |
| MSIDSN | 11.74 | 205.00 |

该论文强调动态 dilation 与 MSCDIM 模块可在 baseline TCN 上带来明显提升。我们的 P-SA-MCD-TCN 是受其启发的轻量化工程版本，但当前 FD001 只跑了 P-TCN baseline，还没跑 P-SA-MCD-TCN 正式对比。

### SFTN-CM/迁移学习（参考论文 5）

关键设置：

| 参数 | C-MAPSS | Bearing |
|---|---:|---:|
| Window length | 30 | 30 |
| Learning rate | 0.0005 | 0.001 |
| Epoch | 200 | 100 |
| Batch size | 32 | 64 |
| Optimizer | SGD | SGD |
| RUL 上限 | 125 | - |

与我们相比：窗口长度和 RUL 上限一致；我们当前用 AdamW、batch size 128、epoch 30，更偏快速验证。迁移学习阶段后续可以参考其 200 epoch 但比赛开发阶段可先保持 30 epoch。

## 是否能达到论文效果

当前 P-TCN baseline 的最后窗口 RMSE 为 16.47，尚未达到参考论文普通 TCN 的 13.51，也未达到 SOTA 的 10-12 区间。主要原因包括：

1. 当前训练预算仅 30 epoch，而多数论文使用 200-1000 epoch 或 early stopping。
2. 当前 baseline 结构较简单，尚未启用 P-SA-MCD-TCN 的多尺度跨维交互优势。
3. 当前 batch size 128 偏大，可能降低小数据 FD001 上的泛化细节。
4. 当前只做了一次随机划分，未进行多 seed 平均或超参数搜索。
5. 当前训练策略没有学习率调度、warmup、早停、模型集成等增强。

## 下一步建议

在不增加训练轮数、仍保持 30 epoch 的前提下，建议按顺序做：

1. 跑 `P-SA-MCD-TCN` 的 FD001 对比，验证我们的主模型是否优于 P-TCN。
2. 把 C-MAPSS 官方 last-window 指标写入每次实验的 `metrics.json`。
3. 新增轻量调参配置：batch size 64、learning rate 0.0005、hidden 64/96、dropout 0.1/0.2。
4. 后续正式报告再选择 100-200 epoch 做最终表格，当前开发阶段继续保持 30 epoch。

## 30 epoch 主模型对比结果

| 模型 | All-window RMSE | All-window MAE | Last-window RMSE | Last-window MAE | Last-window Score | 参数量 | 延迟 ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| P-TCN | 16.79 | 12.57 | 16.47 | 12.28 | 572.50 | 128,897 | 0.1202 |
| P-SA-MCD-TCN | 15.37 | 11.98 | 14.90 | 10.95 | 390.10 | 158,281 | 0.1204 |
| P-SA-MCD-TCN full | 14.66 | 10.69 | 14.52 | 10.77 | 340.27 | 267,738 | 0.2691 |

30 epoch 快速实验下，P-SA-MCD-TCN full 相比 P-TCN：last-window RMSE 降低约 11.84%，last-window Score 降低约 40.56%，all-window RMSE 降低约 12.68%。参数量增加到 267,738，单样本 CPU 推理延迟约 0.2691 ms。该版本是当前 C-MAPSS FD001 主模型入口。
