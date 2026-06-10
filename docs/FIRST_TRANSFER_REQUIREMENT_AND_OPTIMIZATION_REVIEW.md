# 第一迁移完成度与优化审查

日期：2026-06-05

## 结论

第一迁移 `XJTU-SY Bearing -> reaction_wheel_sim` 已经达到阶段性完成：数据生成链路、严格 pseudo-time 目标阶段对齐、50 epoch 主实验、P-TCN / P-SA-MCD / LSTM / GRU 基线、以及 3 seed 稳定性检查均已完成。

但它还不是比赛整体完成。比赛要求至少覆盖两个典型航天仿真场景，并完成公开退化数据训练到航天仿真目标域适配/验证。当前只完成了反作用飞轮方向，卫星电池方向仍未完成。

## 当前第一迁移效果

严格迁移主配置：

- `configs/xjtu_to_reaction_wheel_psa_mcd_stage_lmmd_pseudo_time_w0p003_50e.yaml`
- 模型：`P-SA-MCD-TCN`
- 迁移损失：`stage-aware LMMD`
- 目标域阶段：`target_stage_source: time_progress`
- 训练轮数：50 epoch

单 seed 严格对照中，完整模型 RMSE = 0.194274，优于：

- P-SA-MCD global MMD：RMSE = 0.279885
- P-TCN global MMD：RMSE = 0.277892
- P-TCN source-only：RMSE = 0.359724
- LSTM stage LMMD pseudo-time：RMSE = 0.265569
- GRU stage LMMD pseudo-time：RMSE = 0.273260

3 seed 均值对照中，完整模型仍然最好：

| 方法 | RMSE mean | RMSE std | MAE mean | RA mean | last RMSE mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full P-SA-MCD stage LMMD pseudo-time | 0.227394 | 0.040405 | 0.179558 | 0.562000 | 0.223145 |
| P-SA-MCD global MMD | 0.276324 | 0.003097 | 0.236780 | 0.494515 | 0.468144 |
| P-TCN global MMD | 0.284899 | 0.009955 | 0.241882 | 0.492014 | 0.481417 |

按 3 seed 均值计算，完整模型相对 P-SA-MCD global MMD 的 RMSE 降低约 17.7%，相对 P-TCN global MMD 降低约 20.2%。因此可以说：在当前严格第一迁移对照内，完整模型效果明显最好。

需要保守表述的是，target-only 监督上限 RMSE 约 0.052，其中新增 GRU target-only 的 RMSE 为 0.051842，是当前监督上界中最低的整体误差。完整迁移模型与全监督目标域训练仍有明显差距，这属于无标签/弱标签目标域迁移的正常难度，不能把当前结果表述为接近全监督上限。

## 对比赛要求的覆盖

已覆盖：

- 反作用飞轮摩擦/润滑退化仿真数据集；
- 使用公开退化数据 XJTU-SY Bearing 作为源域；
- 完成公开源域到航天仿真目标域的迁移适配；
- PyTorch 训练、迁移、预测产物已跑通；
- 至少已有 source-only、global MMD、P-TCN、LSTM、GRU 等基础对照。

尚未覆盖：

- 第二个典型航天场景 `satellite_battery_sim`；
- NASA Battery -> satellite_battery_sim 的第二迁移；
- 最终 Docker / 一键复现说明；
- 最终技术报告中的数据集生成方法、迁移流程、对照实验和工程部署说明；
- Conformal Prediction 在第一迁移主配置中尚未启用；
- 原规划中的 `SA-FS-LMMD` 还没有完整落地，目前代码实现是 `stage-aware LMMD`，不是 feature-separated LMMD。

## 论文交叉核验

EviAdapt 指出，不完整目标域下全局对齐容易把源域晚期退化与目标域早期退化误对齐，因此提出 stage-wise alignment 和 evidential uncertainty alignment。我们当前使用 pseudo-time stage LMMD，方向与 stage-wise alignment 一致，但没有实现 evidential uncertainty alignment。

HIWSAN 指出三类关键问题：退化多尺度信息不足、只做全局对齐、忽略时间权重。我们当前 P-SA-MCD-TCN 已覆盖一部分多尺度特征，stage LMMD 覆盖局部阶段对齐，但没有 HI temporal weights 和 HI-based subdomain division。

Feature Separation Transfer Network 指出仅对齐共享特征不够，还要分离 shared features 与 domain-private features，并对共享特征做 global/local alignment。我们规划文件写的是 `SA-FS-LMMD`，但当前代码没有 shared/private feature separation 和正交/去相关约束。

TACDA 通过 target reconstruction 保留目标域特异信息，并做 consistent degradation alignment。这个方向可以作为后续增强，但实现复杂度高，当前不建议先做。

VMD-SSA-PatchTST 主要针对锂电池 RUL，适合作为第二迁移的电池方向参考，不是第一迁移的优先优化项。

## 第一迁移下一步优化优先级

1. 先做稳定性优化。完整模型均值最好，但 RMSE std = 0.040405，高于 global MMD 基线。优先尝试学习率下调、alignment warm-up、source pretrain 后再 adaptation、以及固定 best checkpoint 策略。

2. 补齐 HI-weighted LMMD。把当前按阶段均值的 LMMD 扩展为带 stage/sample weights 的 LMMD，使晚期退化窗口在对齐中权重更高，直接对应 HIWSAN 的核心贡献。

3. 补齐 feature separation。为 P-SA-MCD 特征添加 shared/private 分支和正交或相关性惩罚，再只对 shared feature 做 global + stage LMMD。这一步能让项目名义上的 `SA-FS-LMMD` 与代码一致。

4. 改进 target pseudo-stage。当前 pseudo-time 不用目标 RUL，是严格的；但它假设时间进度和退化阶段一致。后续可用 source-pretrained model 生成目标伪 RUL/HI，再做动态 stage segmentation，更接近 EviAdapt。

5. 启用 Conformal Prediction 做不确定性输出。它不会直接降低 RMSE，但对比赛工程完整性和应用展示有价值。

## 建议

第一迁移可以先作为阶段性完成结果保留，不建议立刻替换主路线。下一步应先在第一迁移上做一个小闭环优化：`source pretrain + alignment warm-up + HI/stage weighted LMMD`，对比当前主配置和两个 global MMD 基线，仍用 50 epoch、至少 3 seeds 报告均值和标准差。
