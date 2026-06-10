# 动态 Demo 输入清单

本文件锁定后续动态 demo 或静态 dashboard 可以读取的结果文件。Demo 应只读这些文件，不现场训练模型，不重新拟合 TC，不使用测试标签做任何校准。

校验入口：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/check_demo_inputs.py
```

## 预测输入

| 名称 | 路径 | 用途 |
|---|---|---|
| `first_transfer_test` | `competition_artifacts/02_experiments/final_outputs/first_transfer/predictions_test.csv` | final TC test predictions for reaction wheel dashboard panels |
| `first_transfer_val` | `competition_artifacts/02_experiments/final_outputs/first_transfer/predictions_val.csv` | validation predictions used by TC calibration |
| `second_transfer_test` | `competition_artifacts/02_experiments/final_outputs/second_transfer/predictions_test.csv` | final TC test predictions for satellite battery dashboard panels |
| `second_transfer_val` | `competition_artifacts/02_experiments/final_outputs/second_transfer/predictions_val.csv` | validation predictions used by TC calibration |

## 指标与表格

| 名称 | 路径 | 用途 |
|---|---|---|
| `first_transfer_metrics` | `competition_artifacts/02_experiments/final_outputs/first_transfer/transfer_metrics.json` | final first-transfer metrics |
| `second_transfer_metrics` | `competition_artifacts/02_experiments/final_outputs/second_transfer/transfer_metrics.json` | final second-transfer metrics |
| `metrics_master` | `competition_artifacts/03_results/metrics_master.csv` | all collected experiment metrics |
| `tc_ablation_summary` | `competition_artifacts/03_results/tc_ablation/tc_ablation_summary.csv` | TC boundary and time-prior ablation |

## 图件与解释文件

| 名称 | 路径 | 用途 |
|---|---|---|
| `strict_raw_comparison` | `competition_artifacts/03_results/strict_unsupervised_comparison.md` | main fair raw transfer comparison |
| `tc_ablation_markdown` | `competition_artifacts/03_results/tc_ablation/tc_ablation_summary.md` | human-readable TC ablation interpretation |
| `final_recommendation` | `competition_artifacts/03_results/final_recommendation.md` | final method and reporting boundary |
| `mechanism_observable_figure` | `competition_artifacts/04_figures/figure_5_mechanism_observable_map.png` | mechanism-observable explanatory figure |
| `phm_dashboard_figure` | `competition_artifacts/04_figures/figure_6_phm_application_dashboard.png` | static PHM dashboard figure |
| `representative_prediction_figure` | `competition_artifacts/04_figures/figure_3_representative_predictions.png` | representative final prediction trajectories |

## 边界

- `strict_raw_comparison` 用于公平 raw 迁移对比，不含最终 TC。
- `final_outputs` 用于 `PG-STDA-SAC-RSPA-TC` 最终工程管线展示。
- `tc_ablation_summary` 用于说明 TC 与时间先验边界，不能隐藏第二迁移 `time-only TC` 很强这一事实。
- AI 生成概念图只能作为 conceptual schematic，不能替代代码生成的实验图和指标。
