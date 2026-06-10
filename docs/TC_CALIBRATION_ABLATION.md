# Validation-only TC 消融结果

本文件用于回答最终 `PG-STDA-SAC-RSPA-TC` 中 `TC` 是否只是依赖时间先验的问题。消融只基于已经训练好的 raw `PG-STDA-SAC-RSPA` 输出，不重新训练模型。

## 实验口径

- `raw`：未校准的 `PG-STDA-SAC-RSPA` 预测。
- `y_pred-only TC`：只用验证集 raw `y_pred` 拟合 ridge 校准。
- `time-only TC`：只用验证集 `time_index` 拟合 ridge 校准，用作时间先验负控。
- `y_pred+time TC`：使用验证集 raw `y_pred` 与 `time_index`，对应最终 `PG-STDA-SAC-RSPA-TC`。
- 所有 TC 变体只使用目标验证集标签拟合校准映射，不使用测试标签，不重新训练模型。

指标方向：RMSE、MAE、NASA、Last RMSE 越低越好；RA 越高越好。

## 结果表

| 任务 | 消融口径 | 特征 | RMSE | MAE | NASA | RA | Last RMSE | RMSE较raw降低 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 第一迁移 | raw | none | 0.1274 | 0.1004 | 7.0249 | 0.6902 | 0.1732 | - |
| 第一迁移 | y_pred-only TC | y_pred | 0.1268 | 0.1000 | 6.9121 | 0.6945 | 0.1686 | 0.4% |
| 第一迁移 | time-only TC | time | 0.1037 | 0.0822 | 5.7250 | 0.7198 | 0.2105 | 18.6% |
| 第一迁移 | y_pred+time TC | y_pred,time_index | 0.0891 | 0.0691 | 4.8414 | 0.7418 | 0.1400 | 30.0% |
| 第二迁移 | raw | none | 0.2888 | 0.2316 | 58.9301 | 0.5117 | 0.3111 | - |
| 第二迁移 | y_pred-only TC | y_pred | 0.2650 | 0.2243 | 59.4954 | 0.5264 | 0.4133 | 8.2% |
| 第二迁移 | time-only TC | time | 0.1463 | 0.1094 | 27.8927 | 0.6835 | 0.2238 | 49.3% |
| 第二迁移 | y_pred+time TC | y_pred,time_index | 0.1525 | 0.1161 | 29.5495 | 0.6706 | 0.2052 | 47.2% |

## 结论

第一迁移中，`y_pred+time TC` 明显优于 `time-only TC`，说明 raw 迁移预测与时间先验具有互补性，最终 TC 主要是在 raw 模型输出基础上修正跨域尺度偏差。

第二迁移中，`time-only TC` 的 RMSE 略低于 `y_pred+time TC`，说明卫星电池仿真目标域具有较强的轨道周期/生命周期时间先验。报告中应据此保持谨慎：第二迁移的 strict raw 结果才是迁移表征能力的主证据，TC 后结果应定位为 validation-only 工程校准管线，而不是纯无监督迁移训练能力。

本消融不改变主结论边界：

- strict raw 迁移能力仍以未校准 `PG-STDA-SAC-RSPA` 为准；
- `PG-STDA-SAC-RSPA-TC` 是 validation-only calibrated final engineering pipeline；
- TC 不应被表述为严格无目标标签的训练模块。
