# 动态 HTML Demo 设计规格

日期：2026-06-10

用途：固定后续动态 HTML demo 的设计边界、数据输入、页面结构和验证口径。该 demo 面向比赛展示与答辩，不承担训练、调参或重新校准职责。

## 1. 定位

动态 HTML demo 应做成离线可打开的 PHM 监控台，用于把当前实验产物转化为可交互展示：

- 展示两个迁移任务：`XJTU-SY Bearing -> reaction_wheel_sim` 与 `NASA Battery -> satellite_battery_sim`。
- 展示最终工程管线：`PG-STDA-SAC-RSPA-TC`。
- 展示 strict raw UDA 与 validation-only TC 的边界。
- 展示代表性预测曲线、关键指标、TC 消融、退化阶段和风险建议。

该 demo 不应做成训练平台，不提供上传新数据后现场训练模型的入口。

## 2. 推荐实现形态

建议放置在：

```text
competition_artifacts/06_html_demo/
```

建议文件结构：

```text
competition_artifacts/06_html_demo/
  index.html
  README.md
  assets/
    demo.css
    demo.js
    demo-data.js
```

其中 `demo-data.js` 由脚本从 `competition_artifacts/05_report_assets/demo_payload.json` 转换生成，例如：

```javascript
window.XA202608_DEMO_PAYLOAD = {...};
```

采用该方式可以避免浏览器直接用 `file://` 打开 HTML 时因安全策略无法 `fetch` 本地 JSON，保证评委双击 `index.html` 也能看到完整内容。

## 3. 只读数据输入

主入口：

```text
competition_artifacts/05_report_assets/demo_payload.json
```

该文件已经包含：

- 两个迁移任务的最终指标；
- 两条代表性预测曲线；
- TC 消融 8 行结果；
- 图件路径；
- reporting boundary。

追溯证据：

```text
competition_artifacts/05_report_assets/demo_input_manifest.json
competition_artifacts/05_report_assets/demo_input_manifest.md
competition_artifacts/03_results/strict_unsupervised_comparison.md
competition_artifacts/03_results/tc_ablation/tc_ablation_summary.csv
competition_artifacts/02_experiments/final_outputs/*/predictions_test.csv
competition_artifacts/02_experiments/final_outputs/*/transfer_metrics.json
```

demo 不现场训练模型，不重新拟合 TC，不使用目标测试标签做任何校准。

## 4. 页面结构

参考项目根目录 `演示demo/1.png` 至 `演示demo/4.png` 的布局方向，但最终 HTML 必须以真实实验产物为准。

### 4.1 总览页

用途：让评委第一眼看到项目闭环和两个任务结果。

应展示：

- 公开源域退化数据；
- 阶段感知迁移；
- 航天仿真目标域；
- RUL/SOH 预测；
- 两个任务的核心指标卡；
- 数据集导入状态；
- raw UDA、validation-calibrated pipeline、conformal interval 三个边界标签。

注意：Conformal 只能作为可选风险表达标签或扩展位，不写成当前最终核心模块。

### 4.2 数据导入与复现页

用途：证明 demo 读取的是已有实验产物，帮助支撑工程可复现性。

应展示：

- 数据集清单；
- 字段角色：输入特征、隐藏状态、标签、元数据；
- 质量检查：缺失率、字段完整性、单位划分、TC 边界；
- 随机种子与配置文件；
- 只读输入 manifest。

### 4.3 反作用轮详情页

用途：展示第一迁移任务的预测与工程解释。

应展示：

- `reaction_wheel_sim` 代表性 unit；
- true RUL 与 final TC prediction 曲线；
- 当前时间滑块；
- 当前 RUL、阶段、风险等级；
- RMSE、MAE、RA、NASA Score、last window RMSE；
- 机理解释：摩擦/润滑退化如何映射到电流、温度、振动代理等遥测量。

### 4.4 卫星电池详情页

用途：展示第二迁移任务的预测与工程解释。

应展示：

- `satellite_battery_sim` 代表性 unit；
- true SOH/RUL 与 final TC prediction 曲线；
- 当前 SOH、内阻风险、退化阶段；
- RMSE、MAE、RA、NASA Score、last window RMSE；
- TC 消融提示：第二迁移中 `time-only TC` 很强，最终报告需说明时间先验边界。

### 4.5 迁移方法与消融页

用途：把方法创新和公平对比讲清楚。

应展示：

- `PG-STDA-SAC-RSPA` strict raw UDA 结果；
- `PG-STDA-SAC-RSPA-TC` final engineering pipeline 结果；
- TC 消融：raw、y_pred-only、time-only、y_pred+time；
- 监督参考只作为上界，不与 strict raw 主结论混排。

## 5. 视觉原则

风格建议采用深色航天控制台，而不是论文静态图：

- 左侧固定导航；
- 中央为曲线与流程；
- 右侧为指标、风险和边界说明；
- 使用蓝/青作为主色，绿色表示健康，黄色表示关注，红色表示临近失效；
- 文字不重叠，模型名称保留英文，解释性文字用中文；
- 图表优先用 Canvas/SVG 原生绘制，避免把整张图当背景导致不可读。

## 6. 交互范围

第一版只做低风险交互：

- 任务切换；
- 页面切换；
- 时间滑块；
- 指标卡随时间更新；
- TC 消融表切换；
- 曲线 hover 提示。

不做：

- 现场训练；
- 文件上传后重新计算；
- 在线调参；
- 调用外网资源；
- 使用测试标签重新校准。

## 7. 验证要求

正式实现后至少验证：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/check_demo_inputs.py
& 'D:\anaconda\envs\jiebang\python.exe' scripts/check_pre_demo_readiness.py
```

并人工打开：

```text
competition_artifacts/06_html_demo/index.html
```

检查：

- 页面非空；
- 两个迁移任务都能切换；
- 曲线来自 `demo_payload.json`；
- 指标与 `transfer_metrics.json` 一致；
- TC 边界表述包含 validation-only；
- 不出现“完全无监督 TC”“真实在轨数据”“AI 图证明实验结果”等越界表述。

