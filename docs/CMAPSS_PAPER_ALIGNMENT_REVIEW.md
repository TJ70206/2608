# C-MAPSS FD001 论文对齐复查与下一步路线

## 当前判断

当前 `P-SA-MCD-TCN full` 没有发现明显代码性错误，训练、预测、last-window 指标和 C-MAPSS RUL 计算逻辑整体正确。与论文仍有差距，主要原因更可能是：

1. 训练预算只有 30 epoch，而论文常见 200-1000 epoch 或 early stopping。
2. 当前验证集来自完整 run-to-failure 训练发动机，验证 last-window RUL 接近 0；官方测试 last-window RUL 来自截断测试序列，分布不同。
3. FD001 是单工况，但当前 full 配置使用了 3 个 operating settings，其中 `setting_3` 为常数，`setting_1/2` 方差极小，归一化后可能变成噪声。
4. 当前模型是轻量工程模型，参数量约 26.8 万；论文 SOTA 如 DVGTformer/注意力图网络通常结构更强、训练更充分。
5. 当前只跑单 seed，没有多 seed 平均、超参数搜索或 100-200 epoch 正式结果。

补充复查：2026 arXiv 预处理驱动 TCN 的核心策略此前没有完整使用。旧版本只用了标准化和 RUL 截断，仍是滑动窗口训练；没有做到完整序列训练、随机尾部裁剪和每个时间点 dense supervision。

## 已复查项目

| 项目 | 结论 |
|---|---|
| C-MAPSS train RUL | 正确，`max_cycle - cycle` |
| C-MAPSS test RUL | 正确，`final_rul + max_cycle - cycle` |
| Last-window 指标 | 正确，按每台测试发动机最大 `time_index` 选最后窗口 |
| NASA Score 方向 | 正确，晚预测 `y_pred > y_true` 惩罚更重 |
| Normalization | 正确，normalizer 只用训练 unit 拟合 |
| Unit split | 正确，训练/验证按 unit 划分，避免窗口泄漏 |
| `resolved_config.json` | 已修正，现在会在 `input_channels` 自动更新后保存 |
| 模型前向/反向 | full 与 sensors-only 配置均通过 smoke test |

## 关键诊断结果

当前 full 配置特征列：

```text
['setting_1', 'setting_2', 'setting_3', 's_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
```

FD001 operating settings 训练集标准差：

| 特征 | std |
|---|---:|
| setting_1 | 0.0021873134 |
| setting_2 | 0.0002930621 |
| setting_3 | 0.0 |

这说明 FD001 的 setting 基本不提供工况区分信息。下一步优先验证 sensors-only 配置是否更接近论文结果。

## 当前结果

| 模型 | All-window RMSE | Last-window RMSE | Last-window Score |
|---|---:|---:|---:|
| P-TCN | 16.79 | 16.47 | 572.50 |
| P-SA-MCD-TCN v1 | 15.37 | 14.90 | 390.10 |
| P-SA-MCD-TCN full | 14.66 | 14.52 | 340.27 |
| P-SA-MCD-TCN full sensors-only | 13.89 | 15.00 | 489.32 |
| P-TCN paper-preproc quick | 12.51 | 15.61 | 452.30 |
| P-TCN paper-preproc long | 12.13 | 15.08 | 378.24 |
| P-SA-MCD paper-preproc 100e | 14.13 | 21.89 | 626.92 |
| P-TCN paper-preproc pseudo-val | 22.27 | 19.68 | 526.12 |

解释：

- **sensors-only**：all-window RMSE 更好，但 last-window 变差，不适合作为当前 FD001 官方指标主模型。
- **paper-preproc quick**：30 epoch 下 all-window RMSE 明显改善，说明完整序列训练方向正确；但 last-window 尚未改善，需要按论文长训练。
- **paper-preproc long**：1000 epoch + early stopping 实际运行 90 epoch，最佳验证 all-window RMSE 为 11.56，最终测试 all-window RMSE 为 12.13；但官方 last-window RMSE 为 15.08。

## 正在运行的消融实验

配置：

```text
configs/cmapss_fd001_psa_mcd_full_sensors_only.yaml
```

目的：去掉 FD001 单工况下可能为噪声的 `setting_1/2/3`，只使用 14 个有效传感器。

运行命令：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/train_baseline.py --config configs/cmapss_fd001_psa_mcd_full_sensors_only.yaml
```

输出目录：

```text
outputs/cmapss_fd001_psa_mcd_full_sensors_only/
```

## 2026 arXiv 预处理策略复现

已新增完整序列训练数据模式：

- `FullSequenceTimeSeriesDataset`
- `collate_sequence_batch`
- `training_mode: full_sequence_random_trim`
- `random_end_trim: true`
- `trim_min: 10`
- `trim_max: 75`
- dense per-timestep RUL loss

已新增配置：

```text
configs/cmapss_fd001_ptcn_paper_preproc_quick.yaml
configs/cmapss_fd001_ptcn_paper_preproc_long.yaml
```

`quick` 是 30 epoch 方向验证；`long` 是接近论文的 1000 epoch + early stopping patience 40。

论文关键设置对齐情况：

| 论文策略 | 当前状态 |
|---|---|
| 完整序列训练 | 已实现 |
| 随机尾部裁剪 10-75 | 已实现 |
| z-score 标准化 | 已实现 |
| RUL 截断 125 | 已实现 |
| 每个时间点预测 RUL | 已实现 |
| 标准 TCN hidden=200 | 已实现 |
| dilation `[1,2,4,8,16]` | 已实现 |
| kernel size 3 | 已实现 |
| dropout 0.3 | 已实现 |
| BatchNorm | 已实现 |
| batch size 8 | 已实现 |
| lr 0.01 | 已实现 |
| 1000 epoch + early stop 40 | `long` 正在运行 |

`long` 训练结果：

| 项目 | 值 |
|---|---:|
| 实际运行 epoch | 90 |
| 最佳 `val_rmse` | 11.56 @ epoch 50 |
| 最佳 `val_last_window_rmse` | 14.38 @ epoch 81 |
| Test all-window RMSE / MAE | 12.13 / 8.82 |
| Test last-window RMSE / Score | 15.08 / 378.24 |
| Test last-window bias | -1.28 |

该结果说明完整序列随机裁剪显著改善了全序列 RUL 曲线拟合，但在当前 checkpoint 选择和验证集构造下，尚未直接转化为更好的官方 last-window RMSE。

## 指标口径说明

训练过程中不能只看 `val_rmse`：

- **`val_rmse`**：验证发动机所有时间点的 dense RMSE，适合观察完整 RUL 曲线拟合能力。
- **`val_last_window_rmse`**：验证发动机最后观测点 RMSE，更接近 C-MAPSS 官方 test 口径。
- **`test last_window_rmse / last_window_nasa_score`**：最终与多数论文表格对比时应优先报告的官方口径。

因此，`P-SA-MCD paper seq` 中出现的 `val_rmse=8.68` 说明模型在验证完整序列上拟合很强，但不能直接等同于论文测试集 RMSE。

补充最终结果：

| 排名口径 | 当前最好版本 | 指标 |
|---|---|---:|
| Test all-window RMSE | `P-TCN paper-preproc long` | 12.13 |
| Test last-window RMSE | `P-SA-MCD full window 30e` | 14.52 |
| Test last-window Score | `P-SA-MCD full window 30e` | 340.27 |

与论文 FD001 `Ours(mean)` 的 RMSE 9.26 相比，当前最接近的同类全序列 RMSE 是 12.13，差距约 2.87。需要注意论文结果是 10 次独立运行均值，且其具体验证/测试保存策略可能与当前实现不同。

`P-SA-MCD paper-preproc 100e` 出现 `best_val_rmse=7.53`，但 test all-window RMSE 为 14.13、last-window RMSE 为 21.89，属于验证集过拟合或验证分布不匹配，不能作为最终好结果。

为解决验证分布不匹配问题，已新增 pseudo-test validation 配置：

```text
configs/cmapss_fd001_ptcn_paper_preproc_pseudo_val.yaml
```

该配置让验证集也采用固定随机尾部裁剪，并按 `val_last_window_rmse` 保存 checkpoint。当前后台训练 ID 为 `1417`。

## 对迁移学习的意义

该策略对后续跨域迁移是有用的，甚至比单纯换模型结构更关键：

1. **缓解源域 final-failure bias**：随机尾部裁剪让源域训练不总是看到完整失效结尾，更接近目标域不完整退化序列。
2. **提供 dense stage supervision**：每个时间点都有 RUL 和 stage，可用于 SA-FS-LMMD 的阶段对齐，而不是只对齐窗口末端。
3. **学习全生命周期形状**：完整序列训练让编码器学习平台期、退化起点和线性下降过程，适合作为迁移特征提取器。
4. **适配 XJTU/NASA**：XJTU 转低频 HI 后可按轴承完整寿命序列训练；NASA 电池可按 cycle-level SOH/RUL 完整序列训练。

下一步应把 `training_mode: full_sequence_random_trim` 作为 P-SA-MCD-TCN 和 transfer 训练的可选默认模式。

## 下一步参数路线

如果 sensors-only 有提升：

1. 将 sensors-only 作为 FD001/FD003 单工况默认配置。
2. FD002/FD004 多工况仍保留 operating settings。
3. 再做 100 epoch sensors-only 正式训练。

如果 sensors-only 没提升：

1. 保留 full 配置。
2. 优先调 `stage_loss_weight`：`0.0 / 0.02 / 0.05`。
3. 调 `late_prediction_weight`：`1.0 / 1.05 / 1.15`。
4. 调 `learning_rate`：`0.0005 / 0.001`。
5. 再考虑 hidden channels `96`。

## 进入下一阶段的条件

建议满足以下任一条件后进入 XJTU/NASA 阶段：

1. FD001 30 epoch last-window RMSE 进入 13.x。
2. 或确认 30 epoch 下已经到当前工程上限，并启动 100 epoch 正式训练作为后台长任务。
3. 或 sensors-only 与 full 差别不大，说明差距主要来自训练预算和论文模型复杂度，而不是明显 bug。
