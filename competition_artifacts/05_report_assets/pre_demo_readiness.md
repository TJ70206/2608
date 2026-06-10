# 动态 Demo 前就绪检查

本报告由 `scripts/check_pre_demo_readiness.py` 生成，用于确认 Docker 收尾和动态 HTML demo 开发前，当前证据包、payload、manifest 和报告口径边界处于可用状态。

| 检查项 | 状态 | 文件/入口 | 说明 |
|---|---|---|---|
| check_demo_inputs | 通过 | `scripts/check_demo_inputs.py` | demo input check ok: 15 manifest entries |
| check_html_demo | 通过 | `scripts/check_html_demo.py` | html demo check ok |
| demo_input_manifest | 通过 | `competition_artifacts/05_report_assets/demo_input_manifest.json` | size_bytes=5674 |
| demo_payload | 通过 | `competition_artifacts/05_report_assets/demo_payload.json` | size_bytes=59850 |
| strict_raw_comparison | 通过 | `competition_artifacts/03_results/strict_unsupervised_comparison.md` | required_text_present |
| tc_ablation_boundary | 通过 | `competition_artifacts/03_results/tc_ablation/tc_ablation_summary.md` | required_text_present |
| final_recommendation_boundary | 通过 | `competition_artifacts/03_results/final_recommendation.md` | required_text_present |
| pre_demo_boundary_audit | 通过 | `competition_artifacts/05_report_assets/source_docs/PRE_DEMO_BOUNDARY_AUDIT.md` | required_text_present |
| pre_demo_hardening | 通过 | `competition_artifacts/05_report_assets/source_docs/PRE_DEMO_SUBMISSION_HARDENING.md` | required_text_present |
| html_demo_design_spec | 通过 | `competition_artifacts/05_report_assets/source_docs/HTML_DEMO_DESIGN_SPEC.md` | required_text_present |
| html_demo_index | 通过 | `competition_artifacts/06_html_demo/index.html` | size_bytes=2476 |
| html_demo_data | 通过 | `competition_artifacts/06_html_demo/assets/demo-data.js` | size_bytes=59884 |
| docker_reproducibility | 通过 | `competition_artifacts/05_report_assets/source_docs/DOCKER_REPRODUCIBILITY.md` | required_text_present |

## 边界

- 本检查不运行 Docker，也不启动动态 HTML demo。
- 本检查不训练模型，不重新拟合 TC，不读取目标测试标签做校准。
- 后续 demo 应优先读取 `competition_artifacts/05_report_assets/demo_payload.json`；其他文件只作为追溯证据。
- `PG-STDA-SAC-RSPA` 是 strict raw UDA 迁移模型；`PG-STDA-SAC-RSPA-TC` 是 validation-only calibrated final pipeline。
