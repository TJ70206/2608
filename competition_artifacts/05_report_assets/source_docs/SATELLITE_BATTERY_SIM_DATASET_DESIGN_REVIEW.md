# satellite_battery_sim 自建数据集最终设计稿

> 当前状态：交叉审阅后修订版。  
> 本文件用于在正式编写 `satellite_battery_sim` 生成脚本之前固定数据集设计。本文不等同于最终脚本说明，也不包含代码实现。

## 0. 本文件要回答的问题

第二个迁移实验需要一个与第一迁移 `reaction_wheel_sim` 不同的目标域自建数据集。第一迁移对应反作用飞轮退化；第二迁移建议对应卫星锂离子电池退化，即：

```text
公共源域数据集：NASA Battery Aging / 可选 CALCE Battery
目标域自建数据集：satellite_battery_sim
迁移任务：由地面电池退化数据迁移到卫星在轨电池退化仿真数据
预测对象：SOH / RUL，主评价建议以 RUL 回归为核心，SOH/容量/内阻作为辅助诊断量
```

本设计稿的目标是先把数据集物理假设、变量、标签、失效阈值、泄漏边界、验证规则写清楚。交叉审阅后的最终决策如下，后续生成脚本必须以这些决策为准。

## 0.1 交叉审阅后的最终决策

| 问题 | 最终决策 |
|---|---|
| 第二迁移数据集 | 保持 `NASA Battery Aging -> satellite_battery_sim`，CALCE 只作为后续扩展或动态工况参考 |
| 建模尺度 | 保持等效单体/等效电池单元，不做完整电池包拓扑 |
| 目标轨道场景 | 第一版采用 LEO-like 工况：约 90 min 轨道、约 55 min 光照充电、约 35 min 阴影放电 |
| DOD/SOC 窗口 | 主体 DOD 20%-40%，SOC 真值范围通常约 0.55-0.96；不再使用 0.25-0.95 作为默认窗口 |
| 默认输入 | 严格使用 `voltage/current/temperature` 三类遥测量；`soc_true` 不作为输入 |
| SOC 处理 | 生成 `soc_true` 用于仿真端电压和校验；可选生成含噪 `soc_est`，但只用于 target-rich 消融，不进入主实验 |
| `cycle` 输入 | 不作为默认输入；只用于排序、窗口切片和标签计算 |
| 数据划分 | 第一版使用 unit-level `70/15/15`，即 train/val/test；如后续做不确定性量化，再从 val 中划出 calib |
| EOL 阈值 | 目标域使用容量 80% 或内阻 1.33 倍 BOL；同时明确 NASA 源域官方停机标准是 30% capacity fade |
| Thevenin 时间尺度 | 若每周期只输出 24/32 点，内部仿真必须使用细步长或设置宏观等效时间常数，避免极化项退化为静态欧姆压降 |
| right-censored 样本 | 第一版训练/测试不引入删失单元；可在附加鲁棒性实验中生成少量截断样本 |

## 1. 比赛要求对应关系

比赛方案要求提交内容中包含公共数据集和自主构建的空间环境仿真数据集，并说明仿真数据集构建过程，覆盖运行、故障、长期退化等场景；代码需要支持从公共退化数据训练到空间仿真数据适配、微调、预测的完整流程。

本数据集在比赛要求中的定位如下：

| 比赛要求 | satellite_battery_sim 的对应设计 |
|---|---|
| 公共数据集与自建数据集结合 | 源域使用 NASA Battery Aging；目标域构建卫星电池仿真数据 |
| 跨领域退化数据迁移 | 地面标准充放电循环 -> 卫星轨道光照/阴影周期、温度循环、载荷脉冲 |
| 长期退化场景 | 容量衰减、内阻增长、容量膝点加速、热应力加速 |
| 故障场景 | 容量膝点故障、内阻阶跃故障、可选热控异常故障 |
| 可复现实验 | 固定随机种子、配置文件、数据字典、生成日志、统计校验 |
| 预测任务 | RUL、SOH、容量、内阻等标签；主任务建议 RUL |
| 对比实验 | target-only、source-only、fine-tune、MMD/LMMD、完整模型等 |

结论：从比赛文本看，第二迁移不能只复用第一迁移数据，也不宜只做 NASA 电池数据内部划分；应明确构建一个空间运行条件下的电池目标域数据集。

## 2. 已阅读文献与作用分工

本次核查覆盖了用户指定的 `参考论文/数据集论文/4.md` 至 `12.md`。其中 4、5 与反作用飞轮相关，不作为电池物理模型的核心依据；6-12 是本数据集的主要依据。

| 文件 | 主题 | 对本数据集的作用 |
|---|---|---|
| `参考论文/数据集论文/4.md` | ITHACO 反作用飞轮退化数据 | 第一迁移依据，不混入电池仿真；仅用于区分两个目标域 |
| `参考论文/数据集论文/5.md` | 反作用飞轮摩擦/轴承退化 | 第一迁移依据，不作为电池物理约束 |
| `参考论文/数据集论文/6.md` | 电池老化经验/半经验模型综述 | 支撑容量衰减、内阻增长、温度、C-rate、SoC、DoD、日历老化与循环老化 |
| `参考论文/数据集论文/7.md` | 半经验 SOH 估计模型 | 支撑周期数、温度、放电电流共同影响 SOH；给出 `SOH = Qmax_aged / Qmax_fresh` 思路 |
| `参考论文/数据集论文/8.md` | NASA/CALCE 电池数据与半经验电化学电压模型 | 支撑公共源域选择、NASA 电池工况和电压/电流/温度字段 |
| `参考论文/数据集论文/9.md` | 多变量容量退化模型 | 支撑温度、C-rate、循环次数、运行时间共同驱动容量损失 |
| `参考论文/数据集论文/10.md` | 容量与内阻融合 RUL 预测 | 支撑 80% 容量 EOL、1.33 倍内阻 EOL、容量+内阻融合健康指标 |
| `参考论文/数据集论文/11.md` | 容量与欧姆内阻在线 SOH | 支撑容量衰减和内阻增长高度相关、内阻在线估计、SoC 80%-30% 稳定区间 |
| `参考论文/数据集论文/12.md` | 考虑温度的改进 Thevenin 模型 | 支撑 `OCV(SOC,T)`、欧姆内阻、极化支路和端电压生成 |

重要边界：本文不把反作用飞轮文献中的物理量或故障机制迁移到电池数据集。两个目标域应保持机制、变量、标签定义独立，只在迁移学习算法层共享框架。

## 3. 关键文献证据与设计解释

### 3.1 公共源域为什么选 NASA Battery Aging

`8.md` 描述了 NASA Open Data Portal 的 Li-ion Battery Aging 数据集，包含商用 18650 锂离子电池，标称容量约 2Ah，充放电、阻抗等实验过程，记录电压、电流、温度等信号，并通过容量衰退得到寿命标签。NASA Open Data Portal 对该数据集的说明中也明确列出了电压、电流、温度、容量和阻抗字段，且实验停止条件为额定容量 30% 衰减，即 2Ah 到 1.4Ah。该数据集与比赛中的跨域退化迁移目标匹配度高：

- 源域是真实电池退化数据，而非纯模拟。
- 信号字段与目标域可对齐：电压、电流、温度、容量、循环数。
- 源域工况偏地面标准循环，目标域可设计为卫星轨道运行条件，从而形成合理 domain shift。

设计解释：NASA 数据适合作为第二迁移的公共源域。`satellite_battery_sim` 不应简单复制 NASA 的恒定充放电循环，而应引入轨道周期、热循环、载荷脉冲和长期容量/内阻退化，使目标域具备航天器运行特征。

重要区分：

```text
NASA 源域官方停止标准：30% capacity fade，约 2Ah -> 1.4Ah。
satellite_battery_sim 目标域 EOL：capacity <= 0.8 Q_BOL 或 internal_resistance >= 1.33 R_BOL。
```

这两个阈值不强行统一。源域和目标域 EOL 定义差异本身是 domain shift 的一部分。迁移训练时可统一到 `normalized_rul` 或 `normalized_soh` 标签空间，但技术报告必须说明源域原始 EOL 和目标域仿真 EOL 的差别。

### 3.2 容量衰减与内阻增长必须同时建模

`10.md` 指出容量衰减和内阻增长都可作为 RUL 指标。该文献在实验中把容量 EOL 设为 BOL 容量的 80%，并指出当容量下降到 80% BOL 时，内阻约增加 33%，因此给出了以下参考阈值：

```text
Q_EOL = 0.8 * Q_BOL
R_EOL = 1.33 * R_BOL
```

`11.md` 进一步说明容量衰减和欧姆内阻增长之间存在强相关性，可用于在线 SOH 估计。

设计解释：只生成容量衰减是不够严谨的。卫星电池退化数据应至少同时包含：

- `capacity_ah`：剩余可用容量。
- `internal_resistance_ohm`：欧姆内阻或等效直流内阻。
- `capacity_ratio`：`capacity_ah / Q_BOL`。
- `resistance_ratio`：`internal_resistance_ohm / R_BOL`。
- `soh`：综合健康状态。
- `rul_cycles`：到失效阈值的剩余循环数。

### 3.3 温度、C-rate、DoD、运行时间是必须的退化驱动因素

`6.md` 与 `9.md` 都强调电池老化受温度、充放电倍率、循环次数、SoC/DoD、运行时间影响。`9.md` 的多变量退化模型把容量损失拆为循环项与运行时间项；`6.md` 指出日历老化和循环老化分别受不同应力因素影响。

设计解释：目标域数据不能只有 `cycle -> capacity` 的单变量曲线。至少需要生成：

- 轨道内动态 `current`。
- 轨道内动态 `temperature`。
- `soc_true` 与 `dod_cycle`，其中 `soc_true` 只用于仿真和校验。
- 名义 `c_rate_nominal`。
- `elapsed_days` 或累计运行时间。
- 单元级参数差异，用来模拟不同电芯初始状态和寿命差异。

### 3.4 Thevenin 端电压模型用于保证信号形态合理

`12.md` 给出了考虑温度影响的改进 Thevenin 模型，核心思想是端电压由 `OCV(SOC,T)`、欧姆压降和极化压降共同决定。

本文建议使用下列结构生成端电压：

```text
v_p[k+1] = a * v_p[k] + R_p * (1 - a) * current[k]
a = exp(-dt / (R_p * C_p))
voltage[k] = OCV(soc[k], temperature[k]) - current[k] * R0[k] - v_p[k] + noise
```

其中本数据集约定：

```text
current > 0 表示放电
current < 0 表示充电
```

注意：`12.md` 中的原始 OCV 多项式系数来自特定电池与实验条件。第一版数据集不建议直接照抄系数，而应使用一个受物理范围约束的 `OCV(SOC,T)` 代理函数，保证端电压落在合理范围内，再在技术报告中说明这是仿真近似。

## 4. 目标域物理场景设计

### 4.1 目标对象

目标对象为卫星能源系统中的锂离子蓄电池单体或等效电池单元。第一版不模拟完整电池包均衡和热网络，而模拟等效单元的长期退化。这样可以控制复杂度，并保证与 NASA 源域字段可对齐。

建议第一版配置：

| 项目 | 建议值 | 说明 |
|---|---:|---|
| 虚拟单元数 | 100 | 与第一迁移数据规模接近，便于对照 |
| 单元寿命 | 300-1200 accelerated cycles | 每个 cycle 表示一个轨道充放电周期或等效加速老化任务周期 |
| 每周期采样点 | 24 或 32 | 覆盖光照、阴影和载荷脉冲 |
| 标称容量 `Q_BOL` | 2.0 Ah 附近随机扰动 | 与 NASA 18650 2Ah 量级对齐 |
| 初始内阻 `R_BOL` | 0.035-0.080 ohm | 作为等效欧姆内阻，按单元扰动 |
| 轨道周期 | 90 min | LEO-like 第一版；55 min 光照、35 min 阴影 |
| DOD 范围 | 主体 20%-40%，少量压力样本可到 50% | 与 LEO 电池测试更一致 |
| SOC 真值范围 | 通常约 0.55-0.96 | 仅用于仿真和校验，不作为默认输入 |
| 温度范围 | 常规 0-40 degC，压力场景可到 45 degC | 与 LEO 电池和文献温度工况衔接 |
| 失效阈值 | 容量 80% 或内阻 1.33 倍 | 与 `10.md`、`11.md` 对齐 |

说明：真实 LEO 电池可运行数万次轨道循环。为了让数据集规模适合本项目训练，第一版采用加速老化仿真，将退化系数校准到 300-1200 个等效周期内触发 EOL。技术报告中应明确这是 accelerated simulation，而不是声称真实 LEO 电池只运行 300-1200 个轨道即失效。

### 4.2 轨道运行周期

每个 cycle 表示一个 LEO-like 轨道周期或等效加速充放电任务周期。第一版固定为 90 min 周期，其中约 55 min 为光照充电，约 35 min 为阴影放电。一个周期内分为：

```text
sunlight phase：太阳照射，太阳翼供电，电池以小到中等倍率充电，允许 CC + taper 形态
eclipse phase：地影期，电池放电支撑平台载荷，主体 DOD 20%-40%
pulse load：短时高功耗载荷或姿控/通信任务脉冲
```

建议生成字段：

```text
orbit_phase        # 0-1，周期内相位
sunlight           # 1 表示光照，0 表示阴影
load_mode          # normal / pulse / recovery，可选
current            # A，放电为正，充电为负
temperature        # degC
soc_true           # 0-1，仿真真值，不作为默认输入
dod_cycle          # 周期内放电深度
```

轨道周期的 domain shift 相比 NASA 源域主要体现在：

- NASA 多为受控 CC/CV 充电和 CC 放电。
- 目标域存在交替光照/阴影。
- 目标域电流不是单一恒定倍率，而是低倍率背景 + 载荷脉冲。
- 目标域温度存在轨道周期变化和长期热控差异。

### 4.3 温度模型

第一版采用可解释的轨道热循环模型：

```text
temperature =
    unit_base_temp
  + orbit_temp_amp * sin(2*pi*orbit_phase + phase_offset)
  + current_heating_coeff * abs(current)
  + seasonal_drift
  + thermal_noise
```

并设置少量热应力单元：

```text
thermal_stress = normal / high_temp / poor_thermal_control
```

建议约束：

- 正常单元大部分时间在 15-35 degC。
- 高温或热控异常单元可短时到 40-45 degC。
- 温度不作为标签，而作为输入应力变量。
- 温度范围和异常比例写入 metadata。

## 5. 退化模型设计

### 5.1 单元级随机参数

每个虚拟单元需要独立采样一组退化参数，以形成个体差异：

```text
Q_BOL
R_BOL
cycle_life_target
capacity_decay_rate
resistance_growth_rate
thermal_sensitivity
c_rate_sensitivity
dod_sensitivity
knee_cycle
knee_strength
fault_type
fault_cycle
```

设计原则：

- 同一 split 内和不同 split 间都要有分布差异，但不能让 test split 完全脱离训练分布。
- fault unit 比例建议第一版控制在 20%-30%。
- 故障不能直接泄漏到默认输入特征；`fault_type` 仅用于分析和分层评价。

### 5.2 容量衰减

建议用半经验损伤累积形式，而不是直接用固定直线下降。核心原因是它能自然接入温度、倍率、DoD 和运行时间。

可采用如下设计：

```text
capacity_damage[k+1] =
    capacity_damage[k]
  + cycle_damage_increment[k]
  + calendar_damage_increment[k]
  + knee_damage_increment[k]

capacity_ah[k] = Q_BOL * (1 - capacity_damage[k])
```

循环损伤项：

```text
cycle_damage_increment =
    k_cyc
  * f_temp(temperature)
  * f_c_rate(c_rate_nominal)
  * f_dod(dod)
  * throughput_ah
```

日历或运行时间损伤项：

```text
calendar_damage_increment =
    k_cal
  * f_temp(temperature)
  * f_soc(soc)
  * delta_days
```

容量膝点项：

```text
knee_multiplier = 1 + knee_strength * sigmoid((cycle - knee_cycle) / knee_width)
knee_damage_increment = base_increment * (knee_multiplier - 1)
```

设计解释：

- `6.md` 和 `9.md` 支持温度、倍率、DoD、时间共同影响容量衰减。
- `7.md` 支持用半经验模型从周期数、温度、放电电流估计 SOH。
- 膝点不是每个单元都有；只在部分 fault 或 accelerated aging 单元中明显出现。

### 5.3 内阻增长

内阻增长应与容量衰减相关，但不能完全由容量决定。建议：

```text
resistance_damage[k+1] =
    resistance_damage[k]
  + rho_corr * capacity_damage_increment[k]
  + rho_independent * resistance_stress_increment[k]
  + resistance_step_fault[k]

internal_resistance_ohm[k] =
    R_BOL * (1 + 0.33 * resistance_damage_scaled[k])
```

其中 `resistance_damage_scaled = 1` 附近对应 `R_EOL = 1.33 * R_BOL`。

设计解释：

- `10.md` 支持内阻达到 1.33 倍 BOL 作为 EOL 阈值。
- `11.md` 支持容量衰减和内阻增长之间存在强相关，但仍建议保留独立噪声和故障项，以形成真实的个体差异。

### 5.4 故障类型

第一版建议只保留少量可解释故障，避免脚本复杂度过高：

| fault_type | 机制 | 数据表现 | 是否建议第一版加入 |
|---|---|---|---|
| `none` | 正常老化 | 平滑容量衰减和内阻增长 | 是 |
| `capacity_knee` | 容量膝点加速 | 中后期容量衰减斜率突然变大 | 是 |
| `resistance_step` | 内阻阶跃增长 | 某周期后内阻永久上移，端电压压降增大 | 是 |
| `thermal_stress` | 热控异常 | 温度偏高，容量和内阻老化加速 | 可作为第二版或少量第一版 |
| `mixed` | 多机制叠加 | 难度更高 | 暂不建议第一版大量使用 |

第一版推荐比例：

```text
none:             70%-80%
capacity_knee:    10%-15%
resistance_step:  10%-15%
thermal_stress:    0%-5%  # 可选
```

## 6. 标签、阈值与健康指标

### 6.1 EOL 定义

每个单元的失效周期定义为以下条件首次满足的周期：

```text
capacity_ah <= 0.8 * Q_BOL
or
internal_resistance_ohm >= 1.33 * R_BOL
```

如果一个单元在仿真截止时仍未达到阈值，应继续生成到阈值附近，或者标记为 right-censored。为了第一版训练和评价简洁，建议所有单元都生成到明确 EOL，不引入删失样本。

### 6.2 RUL 标签

主标签：

```text
rul_cycles = eol_cycle - cycle
normalized_rul = rul_cycles / eol_cycle
```

注意：

- `rul_cycles` 与 `normalized_rul` 不应作为输入。
- 训练时可预测 `normalized_rul`，评价时还原为 `rul_cycles`。
- 如果使用滑动窗口，窗口标签建议取窗口最后一个时刻对应的 RUL。

### 6.3 SOH 标签

容量 SOH：

```text
soh_capacity = capacity_ah / Q_BOL
```

内阻 SOH：

```text
soh_resistance =
    clip((R_EOL - internal_resistance_ohm) / (R_EOL - R_BOL), 0, 1)
```

综合 SOH 建议第一版使用可解释加权：

```text
soh = alpha * clip((soh_capacity - 0.8) / 0.2, 0, 1)
    + (1 - alpha) * soh_resistance

alpha = 0.5
```

这里的 `soh` 是从 BOL 到 EOL 的归一化健康状态，BOL 附近约为 1，EOL 附近约为 0。

### 6.4 融合健康指标

为对齐 `10.md` 中容量和内阻融合思想，可额外输出一个不作为默认输入的辅助分析量：

```text
capacity_damage =
    clip((Q_BOL - capacity_ah) / (Q_BOL - Q_EOL), 0, 1)

resistance_damage =
    clip((internal_resistance_ohm - R_BOL) / (R_EOL - R_BOL), 0, 1)

fused_damage =
    alpha * capacity_damage + (1 - alpha) * resistance_damage
```

说明：

- `fused_damage` 越大表示越接近失效。
- 第一版用线性归一化融合更稳定，避免指数式融合在单位尺度上产生歧义。
- 技术报告中可以说明其思想来自容量-内阻融合 RUL 文献，但具体公式为本仿真数据的可解释实现。

## 7. 端电压与可观测信号

### 7.1 SOC 更新

以库仑计数更新仿真真值 `soc_true`：

```text
soc_true[k+1] =
    clip(soc_true[k] - current[k] * dt_hours / capacity_ah[k], soc_min, soc_max)
```

其中：

- `current > 0`：放电，SOC 下降。
- `current < 0`：充电，SOC 上升。
- `capacity_ah[k]` 是隐含真实容量，不作为默认输入。
- `soc_true` 是仿真内部状态，不作为默认输入。

可选生成一个含噪估计量 `soc_est`，用于 target-rich 消融实验：

```text
soc_est[k+1] =
    clip(soc_est[k] - current_measured[k] * dt_hours / Q_BOL_nominal
         + bias_drift[k] + eps_soc[k], 0, 1)
```

`soc_est` 必须带初始偏差、积分漂移和测量噪声，不能直接复制 `soc_true`。主实验不使用 `soc_est`，因为 NASA 源域没有直接可靠的 SOC 序列，比赛描述也强调基于电压、电流、温度等遥测量。

### 7.2 OCV 代理函数

建议使用单调且受限的 OCV 代理函数：

```text
ocv_base = 3.0 + 1.20 * soc_true - 0.08 * sin(pi * soc_true)
ocv_temp = ocv_base + k_ocv_temp * (temperature - 25)
ocv = clip(ocv_temp, 2.8, 4.20)
```

理由：

- 物理范围清晰，端电压不容易失真。
- 保留 `OCV(SOC,T)` 的文献结构。
- 不直接照搬特定电芯系数，避免由于源论文电芯型号差异导致不合理电压。
- `k_ocv_temp` 第一版建议取约 `-0.0005` 到 `-0.0010 V/degC` 的小量级系数，并通过电压范围检查兜底。

### 7.3 Thevenin 极化支路

端电压由 OCV、欧姆压降、极化压降和噪声组成：

```text
v_p[k+1] = exp(-dt / (R_p * C_p)) * v_p[k]
         + R_p * (1 - exp(-dt / (R_p * C_p))) * current[k]

voltage[k] = ocv[k] - current[k] * internal_resistance_ohm[k] - v_p[k] + eps_v
```

建议约束：

```text
2.5 V <= voltage <= 4.25 V
eps_v ~ N(0, 0.005^2) 到 N(0, 0.02^2)
```

可选遥测量化：

```text
voltage_measured = round(voltage / adc_step_v) * adc_step_v
adc_step_v = 0.001 到 0.01 V
```

第一版可默认开启较小量化步长，例如 `0.001 V` 或 `0.005 V`，模拟遥测采样分辨率；但必须同时保留未量化的仿真中间量用于质量检查。

时间尺度约束：

- 如果直接用每周期 24/32 个输出点更新 Thevenin 极化支路，则 `dt` 约为 170-225 s。
- 若 `R_p*C_p` 只有几十秒，`exp(-dt/(R_p*C_p))` 会接近 0，极化项会退化成近似静态项。
- 因此生成脚本必须二选一：
  - 内部用更细步长积分，例如 5-10 s，再下采样到 24/32 点。
  - 或明确使用宏观等效 `R_p*C_p`，使 `tau = R_p*C_p` 与输出步长同量级，保留轨道内电压过渡。

第一版建议采用“细步长内部积分 + 下采样输出”，物理解释更稳。

### 7.4 默认可观测输入与禁止输入

默认模型输入建议：

```text
voltage
current
temperature
```

这是 strict shared-feature mode，与 NASA 源域字段直接对齐。

可选 target-rich 消融输入：

```text
voltage
current
temperature
soc_est
c_rate_nominal
orbit_phase_sin
orbit_phase_cos
sunlight
```

其中 `c_rate_nominal = abs(current) / Q_BOL_nominal`。target-rich 模式只能作为消融或增强实验，不能替代主对比结果。

默认禁止作为模型输入的列：

```text
capacity_ah
capacity_ratio
internal_resistance_ohm
resistance_ratio
soh
soh_capacity
soh_resistance
fused_damage
rul_cycles
normalized_rul
health_stage
fault_type
fault_active
eol_cycle
split
cycle
time_step
soc_true
```

说明：

- `capacity_ah` 与 `internal_resistance_ohm` 可作为辅助监督或分析标签，但默认不作为输入，否则容易直接泄漏健康状态。
- `cycle` 和 `time_step` 可保留在文件中用于排序和窗口切片；主实验不把绝对时间索引作为输入，避免模型只学寿命进度而不是退化信号。
- `soc_true` 是内部状态真值，禁止输入。`soc_est` 只有在 target-rich 消融中允许使用。

## 8. 输出数据格式

建议输出目录：

```text
data/satellite_battery_sim/
  metadata.json
  units.csv
  timeseries.csv
  arrays.npz
  splits.json
  README.md
```

CSV 是审阅和报告友好的主格式；`arrays.npz` 是训练友好的缓存格式，建议由同一生成脚本同步输出，保证二者来自同一随机种子和同一配置。

### 8.1 `units.csv`

每个虚拟单元一行：

```text
unit_id
split
Q_BOL
R_BOL
eol_cycle
fault_type
fault_cycle
thermal_stress
capacity_decay_rate
resistance_growth_rate
knee_cycle
knee_strength
seed
```

### 8.2 `timeseries.csv`

每个采样时刻一行：

```text
unit_id
split
cycle
time_step
elapsed_days
orbit_phase
orbit_phase_sin
orbit_phase_cos
sunlight
current
voltage
temperature
soc_true
soc_est
c_rate_nominal
dod_cycle
ocv
polarization_voltage
capacity_ah
capacity_ratio
internal_resistance_ohm
resistance_ratio
soh_capacity
soh_resistance
soh
fused_damage
rul_cycles
normalized_rul
health_stage
fault_type
fault_active
```

### 8.3 `metadata.json`

必须记录：

```text
dataset_name
version
random_seed
num_units
steps_per_cycle
split_rule
feature_columns_default
target_columns
forbidden_input_columns
eol_definition
generation_equations
parameter_ranges
literature_basis
script_commit_or_hash
created_at
```

## 9. 数据划分与泄漏控制

### 9.1 单元级划分

必须按 `unit_id` 划分，而不是按行随机划分。

推荐第一版：

```text
train: 70 units
val:   15 units
test:  15 units
```

如果后续需要 conformal calibration 或其他不确定性量化，再从 `val` 中固定划出 `calib`，不要在第一版数据层强制增加第四个 split。

### 9.2 禁止泄漏规则

生成脚本和数据加载器必须检查：

- 同一个 `unit_id` 只能出现在一个 split 中。
- 默认输入列不得包含容量、内阻、SOH、RUL、故障类型、EOL 周期等标签或派生标签。
- 标准化器只能在 train split 上 fit，然后应用到 val/test。
- 滑动窗口不能跨越 unit 边界。
- test split 的参数不能从 train 统计中反推出 EOL。

### 9.3 与 NASA 源域对齐

为了迁移实验能顺利进行，目标域默认输入应尽量与 NASA 源域保持交集：

```text
voltage
current
temperature
```

如果目标域包含 `soc_est`、`sunlight`、`orbit_phase` 等 NASA 源域没有的字段，迁移模型需要两种配置：

```text
shared_feature_mode:
  只使用源域和目标域共同字段

target_rich_mode:
  目标域使用额外轨道字段，源域缺失字段用 mask 或零填充，仅做消融
```

第一版对比实验建议先使用 `shared_feature_mode`，避免因为输入维度不一致引入额外不确定性。后续再测试 `target_rich_mode` 是否提高目标域效果。

## 10. 质量检查清单

生成脚本完成后，必须自动运行以下检查，检查结果写入 `metadata.json` 或单独 `validation_report.json`。

### 10.1 物理范围检查

```text
0 <= soc_true <= 1
0 <= soc_est <= 1
2.5 <= voltage <= 4.25
-10 <= temperature <= 50
capacity_ah > 0
internal_resistance_ohm > 0
0 <= normalized_rul <= 1
```

### 10.2 单调退化检查

对每个 unit：

```text
capacity_ah 的周期级真实值总体非增
internal_resistance_ohm 的周期级真实值总体非减
rul_cycles 随 cycle 严格递减或非增
eol_cycle 与阈值首次触发周期一致
```

允许 `voltage`、`temperature`、`current`、`soc_true` 由于运行工况产生周期内波动。

### 10.3 统计覆盖检查

必须输出：

```text
split counts
fault_type counts
eol_cycle min/median/max
capacity_ratio at EOL
resistance_ratio at EOL
temperature min/median/max
current min/median/max
voltage min/median/max
soc_true min/median/max
dod_cycle min/median/max
```

建议目标：

- test split 中至少包含 `none`、`capacity_knee`、`resistance_step` 三类。
- train/test 的 EOL 分布有重叠，但 test 可略偏难。
- 故障样本不能过少，否则分层指标无意义。

### 10.4 可视化检查

生成后至少保存以下图：

```text
capacity_vs_cycle_examples.png
resistance_vs_cycle_examples.png
voltage_current_temperature_one_unit.png
soc_orbit_profile_one_cycle.png
eol_distribution_by_split.png
fault_type_distribution.png
```

人工审查重点：

- 容量是否有合理缓慢衰减和少量膝点。
- 内阻是否随退化上升，阶跃故障是否清晰但不过度夸张。
- 轨道周期内电流、`soc_true`、温度、端电压是否一致。
- 电压是否存在明显不合理截断或长时间贴边。

## 11. 第一版不建议做的事情

为了保证第二迁移先稳定完成，以下内容不建议第一版加入：

- 完整电池包均衡、电芯串并联拓扑。
- 高保真电化学 P2D/SPM 模型。
- 复杂热网络或轨道姿态热模型。
- 多化学体系混合，例如 NMC/LFP/LCO 全部混入。
- 大量 fault_type 叠加，导致标签解释困难。
- 把容量或内阻作为默认输入特征。
- 先生成 right-censored 未失效样本，增加 RUL 评价复杂度。

这些内容可以作为后续增强版，但不应阻塞第一版迁移实验。

## 12. 第二迁移实验建议

第一版数据集通过交叉审阅和质量检查后，第二迁移可按以下顺序进行：

```text
Step 1: NASA Battery 源域预训练
Step 2: satellite_battery_sim target-only 基线
Step 3: source-only 直接测试目标域
Step 4: source pretrain + target fine-tune
Step 5: MMD / CORAL 全局域对齐
Step 6: stage-aware 或 fault-aware LMMD 对齐
Step 7: 完整模型与所有基线同等 epoch、同等输入、同等 split 对比
```

主指标建议：

```text
RUL RMSE      越低越好
RUL MAE       越低越好
RUL R2        越高越好
SOH MAE       越低越好
EOL timing error 绝对值越低越好
late-stage RMSE 越低越好
```

其中比赛更关心迁移后在目标域上的寿命预测效果，所以最终报告应优先看 `satellite_battery_sim` test split 上的 RUL 指标，而不是源域 NASA 的训练误差。

## 13. 交叉审阅处理结果

外部审阅中合理的意见已经吸收到最终方案中，处理结果如下：

| 审阅意见 | 处理结果 |
|---|---|
| 默认输入不要包含 SOC 真值 | 采纳。主实验默认输入固定为 `voltage/current/temperature`，`soc_true` 禁止输入 |
| 可选 SOC 应为估计值而非真值 | 采纳。允许生成含噪 `soc_est`，但只用于 target-rich 消融 |
| LEO DOD 不宜默认 75% | 采纳。主 DOD 改为 20%-40%，少量压力样本可到 50% |
| 明确 LEO 轨道周期 | 采纳。第一版采用约 90 min 轨道、55 min 光照、35 min 阴影 |
| NASA 源域 EOL 是 30% fade，不是 80% | 采纳并澄清。NASA 原始停机标准与目标域仿真 EOL 分开说明 |
| 1.33 倍内阻阈值依据不足 | 部分反驳。外部泛化依据不强，但本地 `10.md` 明确给出该阈值；本方案把它作为目标域仿真规则，不声称是 NASA 官方阈值 |
| Thevenin 极化时间常数可能与 24/32 点采样不匹配 | 采纳。脚本必须内部细步长积分或设置宏观等效时间常数 |
| 容量膝点是唯象模型 | 采纳。技术报告中应表述为半经验宏观唯象模型，不声称第一性原理 |
| split 改为 70/15/15 | 采纳。第一版不强制 calib split |
| Test 加入 right-censored 未失效单元 | 暂不采纳为第一版。删失样本增加 RUL 标签和评价复杂度，可做附加鲁棒性实验 |
| CSV + NPZ 双格式 | 采纳为实现建议。CSV 便于审阅，NPZ 便于 PyTorch 训练 |
| 两个自建数据集接口统一 | 采纳。生成脚本、配置和 Dataset 加载接口应与 `reaction_wheel_sim` 对齐 |

## 14. 当前结论

本方案在设计层面满足第二迁移数据集的基本要求：

- 与第一迁移反作用飞轮目标域明确区分。
- 与公共 NASA Battery 源域有字段和任务对齐关系。
- 覆盖卫星运行、长期退化、故障三类比赛要求场景。
- 文献依据覆盖容量衰减、内阻增长、温度/C-rate/DoD 应力、Thevenin 端电压和 RUL/SOH 标签。
- 明确了默认输入、禁止输入、split 规则和质量检查，能降低标签泄漏风险。

本最终方案可进入实现阶段：

```text
scripts/generate_satellite_battery_sim.py
configs/dataset/satellite_battery_sim.yaml
data/satellite_battery_sim/README.md
scripts/validate_satellite_battery_sim.py
```
