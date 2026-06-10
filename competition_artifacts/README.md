# XA-202608 比赛证据包

本目录是双迁移寿命预测方案的整理后证据包，由 `scripts/prepare_competition_artifacts.py` 从已有指标、预测文件、配置文件和设计文档自动生成。
脚本每次运行都会先清空并重建本目录，因此这里不手工存放临时文件。

## 目录顺序

| 目录 | 用途 |
|---|---|
| `00_requirement_mapping` | 比赛硬要求与证据文件的对应关系。 |
| `01_datasets` | 数据集清单、数据角色和存在性检查。 |
| `02_experiments` | 最终配置文件、最终预测文件和最终指标输出。 |
| `03_results` | 总指标 CSV、公平对比表、监督参考表、消融表、TC 校准消融和参数敏感性表。 |
| `04_figures` | 可直接放入报告/答辩材料的 PNG/SVG/PDF 图件。 |
| `05_report_assets` | 数据集设计、方法创新和实验更新等报告素材文档。 |
| `99_cleanup` | 产物包清理说明和最终检查点保留策略。 |

## 报告口径边界

- `strict_unsupervised_comparison.md` 是主公平对比表，只比较未经过最终输出校准的 raw 迁移输出。
- `supervised_reference_comparison.md` 是监督参考/上界类对比，不与严格无监督迁移主张混为一类。
- `ablation_table.md` 和 `parameter_sensitivity.md` 用于支撑方法模块有效性和鲁棒性讨论。
- `03_results/tc_ablation/tc_ablation_summary.md` 单独说明 TC 的 `raw`、`y_pred-only`、`time-only`、`y_pred+time` 消融，避免把输出校准误说成纯迁移训练能力。
- `figure_1_strict_transfer_performance.png` 展示 raw 迁移输出的公平对比。
- 最终预测诊断图展示 `PG-STDA-SAC-RSPA-TC` 的验证集校准后输出；TC 只使用目标域验证集，不使用测试集标签。
- `figure_4_aerospace_workflow_summary.png`、`figure_5_mechanism_observable_map.png`、`figure_6_phm_application_dashboard.png` 用于加分项中的工程流程和可视化展示。
- `04_figures` 下每张主图均同时保留 PNG、SVG、PDF，便于报告排版和后续编辑。
- `05_report_assets/demo_payload.json` 是后续 dashboard/demo 的推荐主入口，包含指标卡、代表性预测曲线和 TC 消融。
- `05_report_assets/demo_input_manifest.json` 与 `demo_input_manifest.md` 锁定后续 dashboard/demo 的只读输入契约。
