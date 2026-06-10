# 开源代码参考边界

本项目没有直接复制下载仓库的整套实现，而是参考其任务组织方式和数据处理经验，自行实现符合 XA-202608 方案的干净代码。

## 已参考的本地仓库

- `开源代码/EviAdapt-main`：参考跨域训练入口、source/target 场景组织、evidential 作为二阶段参考。
- `开源代码/PyTorch-LSTM-for-RUL-Prediction-master`：参考 C-MAPSS 基本字段、RUL 标签构造、窗口序列任务形式。
- `开源代码/PyTorch-Transformer-for-RUL-Prediction-master`：参考 C-MAPSS 传感器选择和 RUL clipping 口径。
- `开源代码/rul-datasets-master`：参考 XJTU-SY 作为轴承 RUL 数据集的处理入口。

## 当前代码原则

- 不直接迁移原始高频轴承波形到反作用轮遥测。
- XJTU-SY 先经 `extract_low_freq_hi(signal_window)` 转为低频退化健康指标。
- C-MAPSS 按 `unit_id` 切分，归一化只使用训练单元拟合。
- Conformal Prediction 的校准集与训练/测试按单元独立。
