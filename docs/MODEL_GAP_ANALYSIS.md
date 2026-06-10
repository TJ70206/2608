# P-SA-MCD-TCN 当前效果差距诊断

## 当前结论

当前 `P-SA-MCD-TCN` 在 FD001 30 epoch 下相比 `P-TCN` 是有效的，但还不是论文级“终极模型”。

| 模型 | Last-window RMSE | Last-window Score | 说明 |
|---|---:|---:|---|
| P-TCN | 16.47 | 572.50 | baseline |
| P-SA-MCD-TCN | 14.90 | 390.10 | 有提升，但未到论文 10-12 RMSE 区间 |
| P-SA-MCD-TCN v2 | 14.52 | 340.55 | 加入真实 SA、stage auxiliary loss、后期加权 |
| P-SA-MCD-TCN full | 14.52 | 340.27 | 正式完整模型配置入口，综合表现当前最好 |

主要原因不是数据放错，而是当前实现缺少多项论文关键措施，且训练预算只有 30 epoch。

## 已经做到的措施

| 措施 | 状态 | 位置 |
|---|---|---|
| 固定窗口长度 30 | 已做 | `configs/cmapss_fd001_psa_mcd.yaml` |
| RUL 截断为 125 | 已做 | `configs/cmapss_fd001_psa_mcd.yaml` / `WindowedTimeSeriesDataset` |
| 删除常用无效传感器 S1/S5/S6/S10/S16/S18/S19 | 已做 | `configs/cmapss_fd001_psa_mcd.yaml` |
| 训练集 z-score 归一化 | 已做 | `build_cmapss_loaders` |
| 按 unit 划分训练/验证，避免窗口泄漏 | 已做 | `split_unit_ids` |
| TCN 固定 dilation bank `[1,2,4,8,16]` | 已做 | `configs/cmapss_fd001_psa_mcd.yaml` |
| 多尺度特征融合雏形 | 已做 | `MultiScaleCrossDimensionBlock` |
| SmoothL1 回归损失 | 已做 | `fit_baseline` |
| Conformal 区间 | 已做 | `SplitConformalRegressor` |
| C-MAPSS last-window 官方口径指标 | 已做 | `last_window_metrics` |

## 还没有真正做到的措施

| 论文/规划措施 | 当前状态 | 影响 |
|---|---|---|
| Self-Attention / SA | v1 未实现，v2 已实现 | v2 在 TCN 输出后加入 `MultiheadAttention` |
| Stage-aware auxiliary learning | v1 未实现，v2 已实现 | v2 让 `stage_head` 输出并加入 stage classification loss |
| Stage-aware loss weighting | v1 未实现，v2 已实现 | v2 对后期退化样本提高回归损失权重 |
| Adaptive / dynamic dilation | 未实现 | dilation 是固定 `[1,2,4,8,16]`，没有按样本动态调节 |
| 真正 MSCDIM 跨时间-通道交互 | 部分实现 | 当前只对各 dilation block 的最后时间步做 gate 融合，跨维交互较弱 |
| Last-window-aware checkpoint | 未实现 | 现在按 `val_rmse` 保存最佳模型，而不是按 `val_last_window_rmse` 或 Score |
| Learning rate scheduler | 未实现 | 30 epoch 下收敛不充分时无法细调后期学习率 |
| 多 seed 平均 | 未做 | 单次 seed 可能波动较大 |
| 论文级训练预算 | 未做 | 多数论文 200-1000 epoch 或 early stopping，我们当前固定 30 epoch |

## 为什么 P-SA-MCD-TCN 有提升但仍不够好

1. **模型名与实现不完全匹配**

   当前 `P-SA-MCD-TCN` 没有真正的 Self-Attention；`stage_head` 也没参与训练。因此它更准确地说是“带多尺度门控融合的 TCN”，不是完整 stage-aware/self-attention 模型。

2. **验证模型选择目标不匹配论文指标**

   当前训练保存最佳模型依据是 `val_rmse`，这是所有验证窗口 RMSE。论文 C-MAPSS 表格通常看每台发动机最后检测点的 RMSE/Score。二者目标不完全一致。

3. **后期退化样本没有额外权重**

   RUL 预测真正关心的是接近失效阶段，但当前所有窗口等权训练，模型容易被大量早中期样本主导。

4. **30 epoch 训练预算偏低**

   论文常见设置是 200 epoch、1000 epoch + early stopping，或多次重复取均值。我们当前 30 epoch 主要适合快速验证模型趋势，不适合直接追 SOTA。

5. **batch size 偏大**

   FD001 数据量不大，当前 batch size 128。参考论文常用 8/32/64，大 batch 可能降低泛化细节，也会减少参数更新次数。

## 低成本改进优先级

在继续保持 30 epoch 的前提下，建议优先做这些改动：

1. **先改 checkpoint 选择策略**

   增加配置项，让 C-MAPSS 按 `val_last_window_rmse` 或 `val_last_window_nasa_score` 保存最佳模型。

2. **启用 stage-aware auxiliary loss**

   让 `stage_head` 输出三阶段分类结果，并在训练时加入 `CrossEntropyLoss`。这是当前代码里已经埋了头但没用的措施，性价比最高。

3. **加入退化后期样本加权**

   对 stage=2 或低 RUL 样本提高回归 loss 权重，让模型更关注最后窗口指标。

4. **补一个轻量 Self-Attention block**

   在 TCN 输出后加入 `nn.MultiheadAttention` 或 channel attention，使模型名里的 SA 真实落地。

5. **做 30 epoch 轻量调参**

   优先尝试：

   - batch size: 64
   - learning rate: 0.0005 / 0.001
   - hidden channels: 64 / 96
   - dropout: 0.1 / 0.2
   - dilation bank: `[1,2,4,8,16,32]`

## 建议下一步

不要急着继续 XJTU。先把当前 C-MAPSS 主模型补成真正的 `P-SA-MCD-TCN`：

1. 实现 stage auxiliary loss。
2. 实现 last-window-aware checkpoint。
3. 可选加入 lightweight attention。
4. 再跑 30 epoch 对比。

如果这一步 FD001 last-window RMSE 能从 14.90 进一步降到 13.x，说明主模型方向成立；后面再接 XJTU/NASA 迁移会更稳。

## v2 快速验证结果

已新增 `configs/cmapss_fd001_psa_mcd_v2.yaml`，包含以下改动：
- **真实 Self-Attention**：在 TCN 输出序列后加入轻量 `MultiheadAttention`。
- **Stage auxiliary loss**：`stage_head` 参与三阶段分类监督。
- **后期样本加权**：`stage >= 2` 的回归损失权重设为 1.5。
- **更长 dilation bank**：`[1,2,4,8,16,32]`。
- **小 batch**：batch size 从 128 改为 64。

30 epoch 对比：

| 模型 | All-window RMSE | Last-window RMSE | Last-window Score | 参数量 | 延迟 ms/sample |
|---|---:|---:|---:|---:|---:|
| P-TCN | 16.79 | 16.47 | 572.50 | 128,897 | 0.1202 |
| P-SA-MCD-TCN v1 | 15.37 | 14.90 | 390.10 | 158,281 | 0.1204 |
| P-SA-MCD-TCN v2 | 15.81 | 14.52 | 340.55 | 216,330 | 0.2353 |
| P-SA-MCD-TCN full | 14.66 | 14.52 | 340.27 | 267,738 | 0.2691 |

结论：v2 对官方 last-window 指标继续有效，Score 降低明显；full 在保持 last-window 指标基本相同的同时，把 all-window RMSE 从 v2 的 15.81 降到 14.66，是当前更均衡的主模型版本。

## 正式完整模型

正式完整模型配置为：

```text
configs/cmapss_fd001_psa_mcd_full.yaml
```

完整模型包含：

- **多尺度 TCN dilation bank**：`[1,2,4,8,16,32]`。
- **Temporal self-attention**：通过 `use_temporal_attention: true` 开启。
- **跨尺度统计融合**：通过 `use_temporal_descriptors: true` 同时使用 last/mean/max 描述符做 scale gate。
- **通道门控**：通过 `use_channel_attention: true` 开启 channel gate。
- **Stage-aware auxiliary head**：通过 `use_stage_head: true` 和 `stage_loss_weight` 加入阶段分类辅助监督。
- **后期退化样本加权**：通过 `late_stage_weight` 和 `late_stage_threshold` 控制。
- **晚预测惩罚加权**：通过 `late_prediction_weight` 对 `y_pred > y_true` 的样本提高回归损失。
- **官方 C-MAPSS last-window 指标**：训练输出中保留 `last_window_rmse`、`last_window_nasa_score` 等字段。

已完成验证与训练：

| 检查项 | 结果 |
|---|---|
| `python -m compileall src scripts` | 通过 |
| full 配置模型构建 | 通过 |
| full 模型前向输出 | `pred=[B,1]`，`features=[B,64]`，`stage_logits=[B,3]` |
| 单 batch 训练反向传播 | 通过 |
| FD001 30 epoch 完整训练 | 通过 |
| Last-window RMSE / Score | 14.52 / 340.27 |
| All-window RMSE / MAE | 14.66 / 10.69 |
| 参数量 / 延迟 | 267,738 / 0.2691 ms/sample |

运行完整 30 epoch：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/train_baseline.py --config configs/cmapss_fd001_psa_mcd_full.yaml
```
