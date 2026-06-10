# 动态 Demo 前边界审计

日期：2026-06-10

用途：在 Docker 收尾和动态 demo 开发之前，固定当前方案的口径边界、可引用证据和不应夸大的表述，避免后续报告、答辩和 dashboard 叙事冲突。

## 1. 已锁定的主口径

- `PG-STDA-SAC-RSPA` 是 strict raw 迁移模型，用于公平比较源域监督训练后在目标域上的无校准迁移能力。
- `PG-STDA-SAC-RSPA-TC` 是 validation-only calibrated final engineering pipeline，只用目标验证集做输出校准，不使用目标测试集标签。
- `TC` 不是严格无监督训练模块，不应被写成“完全无目标标签迁移训练”的核心方法。
- `Conformal Prediction` 目前不是最终主方法核心模块，只能作为可选风险区间表达或附录内容。
- `target-only`、`target-supervised`、`source pretrain + target fine-tune` 仅作为监督参考/上界，不与 strict raw 迁移主张混为一类。

## 2. 自建仿真的正确表述

比赛方案允许“利用卫星仿真或模拟软件构建组件长期运行、故障注入及退化演化场景”，并要求“数据生成过程清晰、可复现”。当前更稳妥的表述是：

> 基于 Python 的机理约束航天组件遥测退化仿真引擎 / simulation testbed

这意味着：

- 可以是自研 Python 仿真，不必强制使用 STK、Simulink、AMESim、FreeFlyer 之类商业软件；
- 但不能写成随机合成数据；
- 也不能冒充真实在轨数据；
- 应明确退化机理假设、可观测量、故障注入方式和随机种子。

## 3. 已吸收的关键建议

- TC 消融已保留，并作为最终校准边界证据。
- 第一迁移与第二迁移都使用同一最终框架，但报告口径必须分清 raw 证据和 TC 管线。
- 第二迁移中 `time-only TC` 很强，说明电池仿真域存在明显时间/生命周期先验，因此不能把 TC 后提升全部归因为迁移表征学习。
- 不再优先引入 Mamba、Diffusion、Evidential、复杂 Transformer 作为新主干。

## 4. 还不能说的话

- 不能说 `PG-STDA-SAC-RSPA-TC` 是完全无监督训练模型。
- 不能说第二迁移 TC 后提升全部来自无标签迁移表征。
- 不能说 Python 自研仿真等同高保真整星仿真。
- 不能说 AI 生成图本身就是实验结果证据。

## 5. 动态 demo 前应只读的证据

- 第一迁移最终预测与指标。
- 第二迁移最终预测与指标。
- 严格无监督迁移对比表。
- TC 消融表。
- 机理-遥测映射图。
- PHM 风险看板图。

## 6. 当前阶段的收尾优先级

1. 保持主模型冻结，不再大改算法主线。
2. 维持 strict raw 与 final TC 的分层叙事。
3. 完成 Docker 前的文档、证据包和口径一致性审计。
4. 后续再做静态或动态 demo。
