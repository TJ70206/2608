# 清理与保留清单

本证据包面向报告和答辩整理，原始数据、源代码和完整实验输出仍保留在项目对应目录中。

可安全清理的临时内容：

- Python `__pycache__` 目录。
- 过期 `.pid` 进程标记文件。
- 在保留 `PG-STDA-SAC-RSPA-TC` 最终检查点后，清理重复的非最终 `.pt` 检查点。

必须保留的最终检查点：

- `outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0005_c0p5_srcsup0p7_50e/transfer_model.pt`
- `outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e/transfer_model.pt`

校准后的最终输出目录复制同一检查点，并额外包含只基于验证集 TC 的预测和指标：

- `outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e`
- `outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e`
