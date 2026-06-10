# 比赛要求覆盖矩阵

| 比赛要求 | 对应证据 | 状态 |
|---|---|---|
| 至少覆盖两个典型寿命预测场景 | 反作用轮与卫星电池数据集设计文档；两个最终迁移实验结果 | 已覆盖 |
| 公开退化源域到航天仿真目标域的迁移 | `XJTU-SY -> reaction_wheel_sim` 与 `NASA Battery -> satellite_battery_sim` 的最终配置和输出 | 已覆盖 |
| 长期渐进退化预测，而非瞬时故障分类 | 两个自建数据集设计文档、预测轨迹图和末期误差分析 | 已覆盖 |
| 明确退化机理假设及可观测量 | `05_report_assets/source_docs/*DATASET_DESIGN_REVIEW.md` | 已覆盖 |
| 数据生成与数据集可复现 | `01_datasets/dataset_inventory.md`；项目源码中的生成脚本 | 已覆盖 |
| PyTorch 训练、迁移和预测流程 | `02_experiments` 下的最终配置与输出证据 | 已覆盖 |
| 同数据条件下与基础模型/方法对比 | `03_results/strict_unsupervised_comparison.md` 与监督参考表 | 已覆盖 |
| 消融实验 | `03_results/ablation_table.md` 与 `03_results/tc_ablation/tc_ablation_summary.md` | 已覆盖 |
| 参数敏感性分析 | `03_results/parameter_sensitivity.md` 与参数敏感性补充图 | 已覆盖 |
| 误差、稳定性和临近失效分析 | RMSE/MAE/NASA/RA/alpha/末窗口指标表与阶段误差图 | 已覆盖 |
| 加分项：直观退化过程与工程应用可视化 | Nature-style 主图、流程图、机理-遥测映射图和 PHM 应用看板 | 已覆盖 |
