# XA-202608 航天器关键组件寿命预测代码说明

本目录是 XA-202608“基于跨领域退化数据迁移的航天器关键组件寿命预测方法研究”的工程代码。当前工程已经从早期调试阶段推进到双迁移实验、最终方法整理和比赛证据包生成阶段。

最终统一方法短名为 `PG-STDA-SAC-RSPA-TC`，完整名称为：

```text
Physics-Guided Stage-aware Transfer Domain Adaptation with Stage Auxiliary Calibration,
Reliability-weighted Stage Prototype Alignment, and Validation-only Time-aware Output Calibration
```

## 当前任务

本工程覆盖两个航天关键组件寿命预测迁移场景：

| 迁移任务 | 源域 | 目标域 | 预测目标 |
|---|---|---|---|
| 第一迁移 | `XJTU-SY Bearing` 轴承退化数据 | `reaction_wheel_sim` 反作用轮仿真数据 | 归一化 RUL |
| 第二迁移 | `NASA Battery` 电池老化数据 | `satellite_battery_sim` 卫星电池仿真数据 | 归一化 RUL |

两个目标域均为自建航天仿真数据集，分别对应反作用轮摩擦/热退化和电池容量衰减/内阻增长。数据集设计文档在 `docs/REACTION_WHEEL_SIM_DATASET_DESIGN_REVIEW.md` 与 `docs/SATELLITE_BATTERY_SIM_DATASET_DESIGN_REVIEW.md`。

## 目录结构

```text
configs/                  实验配置文件
data/                     公开源域、处理后数据和自建仿真数据
docs/                     数据集设计、方法创新、实验结果和审阅记录
outputs/                  训练、迁移、预测和校准输出
scripts/                  数据准备、训练、评估、集成和证据包生成入口
src/xa202608/             核心源码：模型、损失、指标、训练评估和工具函数
tests/                    单元测试和关键功能回归测试
competition_artifacts/    自动生成的比赛证据包
```

## 环境

当前主要使用本地 Conda 环境 `jiebang`：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' --version
```

依赖环境可用以下命令更新：

```powershell
& 'D:\Anaconda3\condabin\conda.bat' env update -n jiebang -f environment.yml
```

如需快速检查工程结构：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/check_project.py
```

## 数据目录约定

公开数据和自建数据放置在以下目录：

```text
data/raw/xjtu_sy/                 XJTU-SY Bearing 原始数据
data/raw/nasa_battery/            NASA Battery 原始数据
data/processed/xjtu_sy_hi.npz     XJTU-SY 低频 HI 处理结果
data/simulated/reaction_wheel/    反作用轮仿真目标域
data/simulated/satellite_battery/ 卫星电池仿真目标域
```

检查数据是否就位：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/check_data.py --all
```

## 数据生成与预处理

XJTU-SY 低频 HI 提取：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/prepare_xjtu_hi.py --root data/raw/xjtu_sy --output data/processed/xjtu_sy_hi.npz
```

NASA Battery 预处理：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/prepare_nasa_battery.py --source data/raw/nasa_battery
```

反作用轮仿真数据生成：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/generate_reaction_wheel_sim.py --config configs/sim_reaction_wheel.yaml
```

卫星电池仿真数据生成：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/generate_satellite_battery_sim.py --config configs/sim_satellite_battery.yaml
```

## 训练与迁移

通用监督基线训练入口：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/train_baseline.py --config <配置文件>
```

通用迁移训练入口：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/train_transfer.py --config <配置文件>
```

当前最终推荐配置：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/train_transfer.py --config configs/xjtu_to_reaction_wheel_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml
& 'D:\anaconda\envs\jiebang\python.exe' scripts/train_transfer.py --config configs/nasa_to_satellite_battery_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml
```

说明：最终 `TC` 是验证集输出校准环节。严格无监督主对比表使用未校准的 raw 迁移输出；最终模型展示和应用看板使用 `PG-STDA-SAC-RSPA-TC`。

## 输出文件

训练或迁移输出通常包含：

```text
metrics.json / transfer_metrics.json   测试指标与训练摘要
predictions_test.csv                   测试集逐窗口预测
predictions_val.csv                    验证集逐窗口预测
resolved_config.json                   解析后的配置
env_info.json                          Python / PyTorch / 平台信息
model.pt / transfer_model.pt           模型权重
figures/                               单实验预测图
```

主要评价指标：

| 指标 | 方向 | 含义 |
|---|---|---|
| `RMSE` | 越低越好 | RUL 预测均方根误差 |
| `MAE` | 越低越好 | RUL 平均绝对误差 |
| `NASA score` | 越低越好 | 非对称寿命预测误差惩罚 |
| `RA` | 越高越好 | 相对精度 |
| `alpha@0.5`、`alpha@0.8` | 越高越好 | alpha-lambda 命中诊断 |
| `last_window_RMSE`、`last_5_avg_RMSE` | 越低越好 | 临近失效窗口误差 |

## 比赛证据包

生成整理后的比赛证据包：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/prepare_competition_artifacts.py
```

生成目录为：

```text
competition_artifacts/
```

该脚本会先清空并重建 `competition_artifacts`，再整理：

- 比赛要求覆盖矩阵；
- 数据集清单；
- 最终配置和最终输出；
- 严格无监督对比、监督参考、消融和参数敏感性表；
- PNG/SVG/PDF 图件；
- 报告素材文档；
- 清理和检查点保留说明；
- 各子目录中文 `README.md`。

## 报告口径

需要严格区分三类结果：

- 主公平对比：`strict_unsupervised_comparison.md`，只使用未校准 raw 迁移输出，不含 `TC`。
- 最终方案展示：`PG-STDA-SAC-RSPA-TC`，TC 只使用目标域验证集，不使用测试集标签。
- 监督参考：`target-only`、`target-supervised`、`fine-tune`、集成模型等，只作为参考或上界类结果，不与严格无监督迁移主张混用。

## 测试

关键测试可运行：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' -m unittest tests.test_train_transfer_options -v
```

如只需检查证据包脚本语法：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' -m py_compile scripts/prepare_competition_artifacts.py
```

## Docker 说明

仓库包含基础 Docker 相关文件，后续正式提交前需要完成 build 和运行验证。当前本地主要复现路径仍以 Conda 环境 `jiebang` 为准。
