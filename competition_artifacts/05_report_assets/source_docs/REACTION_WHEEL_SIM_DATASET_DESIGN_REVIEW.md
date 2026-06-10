# reaction_wheel_sim 自建数据集设计审阅稿

本文档用于在正式编码生成 `reaction_wheel_sim` 之前，供其他模型/成员交叉审阅。目标是保证自建反作用轮退化数据集的**文献依据明确、工程假设透明、生成过程可复现、字段定义无泄漏、符合 XA-202608 大赛要求**。

> 当前状态：本文件是数据集设计审查稿，不是最终生成脚本说明。本文已吸收第一轮交叉审阅反馈，重点修正参数单位、工程假设、代理变量、迁移边界和评估口径。待交叉审阅通过后，再据此实现 `scripts/generate_reaction_wheel_sim.py`、配置文件、数据 README 和验证脚本。

---

## 1. 设计目标与比赛合规口径

### 1.1 数据集目标

`reaction_wheel_sim` 是本项目第一个自建航天目标域数据集，用于模拟小卫星/航天器反作用轮在长期运行中的摩擦、润滑和电机健康退化，并支持 RUL 预测与跨域迁移实验。

主实验链路：

```text
XJTU-SY Bearing low-frequency HI  ->  reaction_wheel_sim
```

其中 XJTU-SY 只作为公开源域退化数据，不能直接把高频振动信号迁移到反作用轮。源域和目标域应先统一到低频健康指标/遥测统计层面，再做阶段感知 LMMD 或 HI 加权子域对齐。

### 1.2 合规口径

XA-202608 要求“跨领域退化数据训练 + 航天仿真场景适配”，并要求自建航天仿真数据集的构建方法清晰、可复现。本数据集采用：

```text
Python 自研机理约束遥测退化模拟软件 / 仿真引擎
```

不声称替代 STK、Simulink、AMESim 或高保真整星仿真软件；其定位是：

- **机理约束**：关键退化公式和参数来自公开反作用轮/轴承润滑文献。
- **遥测可观测**：输入字段围绕转速、电流、电压、温度、摩擦/振动代理等可获取量。
- **可复现**：固定随机种子、显式配置、CSV/NPZ 输出、metadata、README、一键命令。
- **可迁移**：与 XJTU-SY 源域在低频 HI 层面建立映射，而非原始信号硬迁移。

---

## 2. 已阅读文献与作用分工

本设计重点参考 `参考论文/数据集论文/1.md` 到 `5.md`，其中 `1.md`、`2.md`、`3.md`、`4.md`、`5.md` 均与反作用轮/动量轮/航天轴承摩擦润滑退化直接相关。

| 本地文件 | 论文主题 | 对数据集设计的作用 |
|---|---|---|
| `1.md` | 反作用轮摩擦参数估计 | 提供 Coulomb、viscous、static/Stribeck 摩擦模型，以及电流-力矩关系 |
| `2.md` | 基于润滑衰减的反作用轮 RUL 估计 | 提供 ITHACO Type-A 参数、润滑温度、润滑损耗模型、两类润滑故障场景 |
| `3.md` | 卫星动量/反作用轮轴承健康监测与寿命延长 | 提供润滑不足的物理阶段、cage instability、温度/振动/力矩异常机理 |
| `4.md` | ITHACO Type-A 反作用轮三阶段 LSTM RUL 预测 | 提供 `kt` 指数退化、30% nominal 失效阈值、噪声/缺失/插值、不同寿命跨度设置 |
| `5.md` | 航天反作用轮精密轴承摩擦力矩预测 | 提供速度、温度、润滑剂类型对摩擦力矩和 Stribeck 曲线的影响，强调低速/过零区 |

---

## 3. 关键文献依据摘录与解释

### 3.1 反作用轮可观测量和健康指标

`4.md` 明确指出，反作用轮可直接获取的传感器量包括：

```text
wheel speed ω_m
motor current I_m
command voltage V_comm
```

而关键健康状态如：

```text
motor torque constant k_t
bus voltage V_bus
```

通常不能直接由传感器读取，需要通过模型或数据驱动方式估计。该论文将 `k_t` 作为健康指标，并通过 `k_t` 与阈值相交来确定 RUL。

`2.md` 则强调润滑温度 `T_lub` 和润滑剩余量/累计损耗同样可作为反作用轮轴承退化的重要健康状态，但 `T_lub` 通常也不易直接测量，需要通过 `ω_m` 和 `I_m` 等遥测估计。

因此本数据集第一版应同时包含：

- **输入遥测**：`wheel_speed`、`motor_current`、`command_voltage`、`temperature`、`friction_torque_proxy`、`vibration_proxy`。
- **辅助健康状态**：`kt`、`kt_ratio`、`lubricant_loss`、`lubricant_hi`、`friction_torque`。
- **主监督标签**：`rul`、`normalized_rul`。
- **迁移对齐标签**：`health_stage`、`degradation_progress`。

训练模型时要避免标签泄漏：`kt`、`lubricant_hi`、`degradation_progress` 可用于数据说明、辅助任务或消融，但默认主输入不应直接包含由未来失效时间计算得到的标签量。

---

### 3.2 ITHACO Type-A / Bialke 模型参数依据

`4.md` Table 1 和 `2.md` Table 2 均使用 ITHACO Type-A / Bialke 风格反作用轮模型。关键参数包括：

| 参数 | 符号 | 文献值 | 用途 |
|---|---|---:|---|
| Motor Torque Constant | `k_t0` | `0.029 Nm/A` | 电机健康指标初始值 |
| Back-EMF | `k_e` | `0.029 V/(rad/s)` | 电机电气反馈 |
| Coulomb friction | `tau_c` | `0.002 Nm` | 干摩擦基线 |
| Flywheel inertia | `J` | `0.0077 N m s^2` | 角加速度计算 |
| Drive gain | `G_d` | `0.19 A/V` | 电压到电流增益 |
| Driver bandwidth | `omega_d` | `9 rad/s` | 电流响应低通 |
| Speed limiter threshold | `omega_s` | `680 rad/s` | 超速限制 |
| Speed limiter gain | `K_s` | `95 V/(rad/s)` | 超速负反馈 |
| Motor poles | `N` | `36` | 转矩扰动/齿槽项 |
| Periodic disturbance amplitude | `B` | `0.22 Nm` | 齿槽/转矩纹波等周期性扰动代理 |
| Input resistance | `R_IN` | `2 Ohm` | 电气模型参数 |

参数边界说明：

- `omega_s` 在 `4.md` 的 ITHACO Type-A 参数表中为 `680 rad/s`，而 `2.md` 的同类模型表中为 `690 rad/s`。第一版实现固定采用 `680 rad/s`，与 `4.md` 的 `k_t` 退化合成数据主依据保持一致；`690 rad/s` 作为文献差异记录在 metadata 中，不在代码中混用。
- `K_s` 不是无量纲常数，应保留文献单位 `V/(rad/s)`。
- `B = 0.22 Nm` 在不同 ITHACO/Bialke 表述中可能被称为 `cogging torque amplitude` 或 `ripple torque`。第一版不区分两者的电机学细节，仅将其作为可配置的周期性低频扰动幅值，不作为核心退化失效机理。

第一版不必完整复刻 Bialke 高保真模型全部模块，但应保留以下可解释结构：

```text
command_voltage -> motor_current -> motor_torque = k_t * motor_current
motor_torque - friction_torque - disturbance -> wheel_speed evolution
```

理由：项目目标是构建可复现退化数据集，而不是复现某一 MATLAB 高保真模型的所有细节。

---

### 3.3 `k_t` 指数退化和失效阈值

`4.md` 将 `k_t` 设置为指数退化：

```text
k_t(t) = k_t0 / exp(a * t)
       = k_t0 * exp(-a * t)
```

并给出三种时间跨度的退化系数：

| 时间跨度 | 采样/时长 | `k_t` 退化形式 |
|---|---|---|
| hours span | 15,000 s | `k_t = k_t0 * exp(-1e-4 * t)` |
| days span | 1,400,000 s | `k_t = k_t0 * exp(-1e-6 * t)` |
| months span | 13,200,000 s | `k_t = k_t0 * exp(-1e-7 * t)` |

上述 `1e-4 / 1e-6 / 1e-7` 分别来自 `4.md` Table 8 / Table 9 / Table 10，不是本文档自行外推。

同一论文说明：在约 `30% nominal` 的 `k_t` 附近，反作用轮无法正常工作。结合 `k_t0 = 0.029 Nm/A`，第一版失效阈值采用：

```text
kt_eol = 0.3 * kt0 = 0.0087 Nm/A
```

这是本数据集最核心、最硬的文献依据之一。

---

### 3.4 摩擦模型依据

`1.md` 给出反作用轮摩擦模型：

```text
T_w = J_w * dω/dt + b * ω + sign(ω) * [c + d * exp(-ω^2 / ω_s^2)]
```

其中：

```text
d = T_s - c
```

并给出电流-力矩关系：

```text
T_w = k_m * I
```

文献参数示例：

| 参数 | 文献值 |
|---|---:|
| `J_w` | `1.5e-3 kg m^2` |
| deterministic `k_m` | `0.0270 Nm/A` |
| statistical `k_m` | `0.0228 Nm/A` |
| deterministic viscous `b` | `5.16e-6 Nms` |
| statistical viscous `b` | `4.83e-6 Nms` |
| Coulomb torque `c` | `0.8795e-3 Nm` |
| static torque `T_s` | `0.9055e-3 Nm` |
| Stribeck speed | `4 rpm` |

但 `2.md`/`4.md` 的 ITHACO Type-A 参数中 Coulomb friction 为 `0.002 Nm`，惯量为 `0.0077`。两个文献对应的反作用轮规格不同，因此第一版应采用：

- **系统尺度参数**：优先采用 `2.md`/`4.md` 的 ITHACO Type-A 参数；
- **摩擦形式**：采用 `1.md` 的 Stribeck/Coulomb/viscous 结构；
- **参数随机化**：以 ITHACO 参数为中心，对 `tau_c`、`viscous_coeff`、`static_torque` 进行小范围随机扰动。

推荐第一版摩擦形式：

```text
friction_torque(t) = b_v(t, T, L) * wheel_speed(t)
                   + sign(wheel_speed(t)) * [tau_c(t) + (tau_s(t) - tau_c(t)) * exp(-(wheel_speed(t) / omega_stribeck)^2)]
                   + torque_noise(t)
```

其中 `L` 表示润滑健康状态。需要注意：`2.md` 中 ITHACO 模型的短期黏性摩擦项随润滑温度升高而下降；而本数据集还需要表达长期润滑损伤导致的边界摩擦和低速摩擦恶化。因此第一版应把“温度导致黏度变化”和“润滑耗尽导致摩擦恶化”拆开建模，不能简单声称文献直接支持 `b_v` 随温度或损伤单调上升。

---

### 3.5 润滑损耗模型依据

`2.md` 给出润滑损耗率与润滑温度的指数关系：

```text
beta(t) = beta0 * exp(-b / T_lub(t)) + w_k
```

累计润滑损耗：

```text
X(t) = beta0 * integral_0^t exp(-b / T_lub(τ)) dτ + B(t)
```

离散形式：

```text
X(t_i) ≈ beta0 * sum_j exp(-b / T_j) * Δt + B(t_i)
```

该论文将 RUL 预测建立在润滑剩余量/累计损耗达到阈值的基础上，并考虑两类故障：

| 故障场景 | 文献含义 | 本数据集映射 |
|---|---|---|
| Excessive lubrication loss | 正常或过量润滑损耗，资源更快耗尽 | `lubrication_accelerated_loss` |
| Insufficient lubrication injection | 补充润滑不足，轴承干涸 | `insufficient_lubrication` |

文献故障参数示例：

| 场景 | 时间 | `b` | `beta` |
|---|---|---:|---:|
| excessive loss | `t <= 159 min` | `37` | `7.494e-6 mL/s` |
| excessive loss | `t > 159 min` | `25` | `8.054e-6 mL/s` |
| insufficient injection | `t <= 159 min` | `37` | `7.494e-6 mL/s` |
| insufficient injection | `t > 159 min` | `49` | `7.000e-6 mL/s` |

注意：文献公式中的温度单位和符号在 OCR 文本中存在一定混乱。实现时建议内部统一使用 Kelvin 温度计算指数项，并在 metadata 中明确说明转换方式；同时将 `b`、`beta0` 作为可配置参数，而非硬编码为唯一真实值。

---

### 3.6 轴承润滑阶段和异常代理依据

`3.md` 强调：航天动量轮/反作用轮中，轴承是唯一主要磨损机械部件，润滑问题是提前失效的主要原因。与地面轴承不同，航天轴承常见问题不是疲劳剥落，而是润滑不足导致的摩擦、cage temperature、cage instability 和 torque anomaly。

关键阶段现象：

| 阶段 | 文献描述 | 数据集代理 |
|---|---|---|
| EHL / 正常润滑 | 温度、力矩较稳定 | `health_stage=0` |
| mixed lubrication | cage temperature/motion 开始异常，race temperature 和 torque 可能还不明显 | `vibration_proxy` 上升，`temperature` 缓慢上升 |
| mode 1 instability | 低频 cage instability，可由 cage sensor 观察，可能不影响 torque | `vibration_proxy` 明显上升 |
| mode 2 instability | torque anomalies 出现，接近破坏性失效 | `motor_current` 波动、`friction_torque` 阶跃或脉冲 |

该文献还指出：在真空中 cage temperature 更高；cage temperature 对速度比对载荷更敏感。这支持我们在仿真中加入：

```text
temperature = ambient_orbit_temperature + speed_heating + friction_heating + noise
```

并让 `vibration_proxy` 和 `torque_noise` 在后期阶段上升。

---

### 3.7 速度、温度和润滑剂对摩擦力矩的影响

`5.md` 通过 free run-down characterization 研究航天反作用轮精密球轴承摩擦力矩。关键结论：

- 输入变量可用 `temperature` 和 `rotation speed` 来预测 `friction_torque`。
- 文献实测/训练摩擦力矩数据覆盖 `0°C`、`25°C`、`40°C`，并利用 RT/ANN/EBT 模型预测 `-5°C`、`-10°C`、`50°C`、`60°C` 等极端温度下的摩擦力矩。因此本数据集在 `0~40°C` 内的温度依赖属于实测范围支撑，超出该范围的温度扰动应标注为基于文献预测结果的工程外推。
- 速度范围覆盖 `-3500 rpm` 到 `3500 rpm`，低速区尤其关键。
- 低速 `0~500 rpm` 或 `0~1000 rpm` 区间能体现润滑状态和 Stribeck 特征。
- Nye / Nyetorr grease 在低速区表现出更接近 Stribeck 曲线的 EHD film 特征，相比 Kluber 更适合低速/过零工况。
- free run-down 是健康评估的重要方法，但在轨不能频繁执行，因为会影响姿态稳定和载荷任务。

对本数据集的设计影响：

1. 需要在 command profile 中包含低速、换向、过零工况。
2. 需要让摩擦力矩对速度和温度敏感。
3. 需要加入 `lubricant_type` 或 `lubricant_regime` 元数据，至少用于模拟不同单元差异。
4. 不应假设在轨可以频繁做 free run-down 测试；因此 `friction_torque` 默认应作为估计代理或派生遥测，而不是必然的直接传感器量。

---

## 4. 第一版数据集范围

### 4.1 不做什么

第一版不做以下事情：

- 不复刻完整 MATLAB/Simulink ITHACO Type-A 高保真模型。
- 不模拟完整整星姿态动力学和四轮冗余控制分配。
- 不声称生成真实飞行数据。
- 不频繁模拟在轨 free run-down 健康检查作为必要操作。
- 不把 XJTU-SY 原始高频振动直接作为目标域输入。

### 4.2 做什么

第一版做以下事情：

- 生成多虚拟单元的反作用轮长期退化遥测序列。
- 同时包含 `k_t` 指数退化和润滑/摩擦退化。
- 提供明确 RUL 标签，失效时间由最先达到的 EOL 条件确定。
- 提供健康阶段标签，用于 stage-aware LMMD。
- 加入高斯噪声、随机缺失、线性插值和缺失掩码。
- 输出 CSV、NPZ、metadata、README。
- 支持目标域监督训练、少标签训练、无标签目标对齐、conformal 校准。

---

## 5. 建议生成机制

### 5.1 单元级参数随机化

每个虚拟反作用轮单元 `unit_id` 采样一组固定参数：

| 参数 | 推荐采样方式 | 依据/说明 |
|---|---|---|
| `kt0` | `Normal(0.029, 0.002)` 后截断 | ITHACO nominal `0.029 Nm/A` |
| `kt_decay_rate` | log-uniform 或按寿命目标反推 | `4.md` 的 `1e-4/1e-6/1e-7` 跨尺度参考 |
| `tau_c0` | `0.002 * Uniform(0.8, 1.2)` | ITHACO Coulomb friction |
| `viscous_coeff0` | small positive random | `1.md`/`2.md` viscous friction |
| `omega_stribeck` | low-speed threshold random | `1.md` Stribeck speed与低速区机理 |
| `lubricant_type` | categorical: `nominal`, `low_temp_sensitive`, `high_loss` | `5.md` 润滑剂差异思想 |
| `temperature_bias` | unit-level random | 航天器热环境差异 |
| `explicit_fault_injection_type` | categorical | 见 5.8 |

`kt_decay_rate` 不建议固定为单一值，否则所有单元寿命过于一致。可采用两种严谨方式之一：

**方式 A：按时间尺度采样**

```text
hours-like:  a ~ LogUniform(5e-5, 2e-4)
days-like:   a ~ LogUniform(5e-7, 2e-6)
months-like: a ~ LogUniform(5e-8, 2e-7)
```

**方式 B：按目标寿命反推**

随机采样目标寿命 `T_eol`，令：

```text
a = -ln(0.3) / T_eol
```

因为 `kt(T_eol) = 0.3 * kt0`。第一版推荐方式 B，更容易控制每个单元长度和失效覆盖。

---

### 5.2 指令和工况序列

为覆盖反作用轮典型运行，应生成混合 command profile：

| 工况片段 | 目的 |
|---|---|
| small sinusoidal command | 模拟姿态微调和周期扰动 |
| step command | 模拟姿态机动和电流响应 |
| low-speed dwell | 覆盖低速摩擦区 |
| zero-speed crossover | 覆盖过零和 stiction/Stribeck 敏感区 |
| occasional high-speed segment | 覆盖速度升高带来的温升和摩擦变化 |

建议字段：

```text
command_voltage ∈ [0, 5] V
wheel_speed in rad/s and optionally rpm
orbit_phase or thermal_phase optional
```

其中 `command_voltage` 不应完全随机白噪声，而应有时间相关结构；否则不符合真实遥测序列特征。

---

### 5.3 遥测动力学简化

推荐第一版简化动力学：

```text
motor_current[t+1] = motor_current[t]
                   + dt * (G_d * command_voltage[t] - motor_current[t]) / tau_driver
                   + current_noise
```

```text
motor_torque[t] = kt[t] * motor_current[t]
```

```text
wheel_speed[t+1] = wheel_speed[t]
                 + dt / J * (motor_torque[t] - friction_torque[t] - external_disturbance[t])
```

其中：

```text
J = 0.0077
G_d = 0.19
omega_d = 9
```

注：`4.md` 参数表中同时出现 `tau_d = 0.245 s` 和 `omega_d = 9 rad/s`，且 `1 / omega_d ≈ 0.111 s` 不等于 `tau_d`。第一版为避免电气环节解释冲突，仅把 `tau_driver` 作为可配置的一阶电流响应时间常数；默认可参考 `1 / omega_d`，但 metadata 必须记录该简化不等同于完整 Bialke 驱动器模型。

---

### 5.4 摩擦退化

推荐摩擦力矩：

```text
friction_torque = viscous_term + stribeck_term + torque_noise
```

其中：

```text
viscous_term = b_v_eff(t) * wheel_speed
```

```text
stribeck_term = sign(wheel_speed) * [tau_c(t) + (tau_s(t) - tau_c(t)) * exp(-(wheel_speed / omega_stribeck)^2)]
```

黏性项拆分为温度效应与长期损伤效应：

```text
b_v_temp(T_lub) = max(b_min, 0.0049 - 0.0002 * (T_lub_C + 30))
b_v_eff(t) = b_v_temp(T_lub) * [1 + alpha_v * lubricant_damage(t)]
```

这样处理的原因是：`2.md` 的短期黏性项表达了温度升高导致润滑剂黏度下降；而长期 `lubricant_damage` 表达润滑耗尽、边界润滑增强和低速摩擦恶化。二者方向不同，因此必须分离，而不能把 `alpha_v` 解释为文献原公式中的温度系数。

Coulomb 与 Stribeck 静摩擦项随长期润滑损伤增强：

```text
tau_c(t) = tau_c0 * [1 + alpha_c * lubricant_damage(t)]
tau_s(t) = tau_s0 * [1 + alpha_s * lubricant_damage(t)]
```

`lubricant_damage(t)` 可取：

```text
lubricant_damage = clip(lubricant_loss / lubricant_capacity, 0, 1)
```

`alpha_c`、`alpha_s`、`alpha_v` 均为工程化简化参数，必须写入配置和 metadata，并在报告中说明其代表长期润滑损伤对摩擦项的放大作用。

---

### 5.5 `k_t` 退化

```text
kt[t] = kt0 * exp(-a_kt * t) + small_process_noise
kt_ratio[t] = kt[t] / kt0
```

失效条件：

```text
kt_ratio <= 0.3
```

`kt` 应整体单调下降或近似单调下降。若加入过程噪声，应对 `kt` 做物理约束，例如：

```text
kt[t] = min(kt[t-1], kt_raw[t])
```

或者只对观测代理加噪声，不对真实 `kt` 状态加破坏单调性的噪声。

---

### 5.6 温度和润滑损耗

推荐温度模型：

```text
temperature = base_temperature
            + orbital_temperature_cycle
            + speed_heating
            + friction_heating
            + fault_heating
            + measurement_noise
```

其中：

```text
speed_heating    ∝ abs(wheel_speed)
friction_heating ∝ abs(friction_torque * wheel_speed)
fault_heating    increases in late lubrication stages
```

润滑损耗率：

```text
loss_rate = beta0 * exp(-b_lub / T_lub_K) + process_noise
```

累计损耗：

```text
lubricant_loss[t+1] = lubricant_loss[t] + loss_rate[t] * dt
lubricant_hi[t] = 1 - lubricant_loss[t] / lubricant_capacity
```

失效条件：

```text
lubricant_hi <= lubricant_threshold
```

第一版可设：

```text
lubricant_threshold = 0.05 或 0.10
```

该阈值是工程假设，必须在 metadata 中标记为 configurable engineering assumption，而非文献直接给定值。`lubricant_capacity` 第一版建议内部使用归一化容量 `1.0`；如果输出物理单位 `mL`，则应根据目标寿命和 `loss_rate` 积分反推，而不要把某个润滑总量硬编码成文献事实。

---

### 5.7 低频不稳定代理 `vibration_proxy`

`vibration_proxy` 用于表达 `3.md` 中 cage instability、torque anomaly 和低速不稳定等现象在低频遥测健康指标层面的代理。它不是直接宣称存在真实星载振动传感器，而是 synthetic surrogate observable，必须在 README 和 metadata 中说明。

推荐生成形式：

```text
vibration_proxy(t) = sigma_base
                   + sigma_lub_gain * lubricant_damage(t) * lognormal_noise(t)
                   + low_speed_burst(t)
                   + measurement_noise(t)
```

其中低速/过零突发项为：

```text
low_speed_burst(t) = A_burst
                   * exp(-(abs(wheel_speed(t)) / omega_low)^2)
                   * lubricant_damage(t)
                   * Bernoulli(p_burst)
```

解释：

- `sigma_base` 表示健康阶段的低频背景波动。
- `sigma_lub_gain * lubricant_damage` 表示润滑损伤越重，cage instability 代理越强。
- `low_speed_burst` 表示低速/过零区 Stribeck、stiction 和不稳定更明显。
- `measurement_noise` 表示普通遥测噪声。

---

### 5.8 故障类型

为与顶层文档“第一版每个场景只保留 2 类关键故障”的口径一致，`k_t` 指数下降不再作为显式故障注入类型，而是所有单元共有的基础退化机制：

```text
base_degradation = kt_exponential_decay
```

第一版显式故障注入只保留 2 类：

| `explicit_fault_injection_type` | 含义 | 主要依据 | 是否第一版启用 |
|---|---|---|---|
| `lubrication_accelerated_loss` | 润滑加速损耗/温升导致资源更快耗尽 | `2.md`, `3.md` | 是 |
| `friction_step` | 某时刻 Coulomb/viscous friction 阶跃上升 | `1.md`, `3.md`, `5.md` | 是 |
| `insufficient_lubrication_injection` | 补充润滑不足导致轴承干涸 | `2.md` | 可选或第二版 |
| `low_speed_instability` | 低速/过零时振动和电流波动增强 | `3.md`, `5.md` | 可作为派生现象，不一定作为独立显式故障类型 |

建议第一版每个单元都有 `kt_exponential_decay` 基础趋势；约 `20%~30%` 单元叠加一种显式故障注入，避免所有单元过于理想。

---

### 5.9 工程假设集中清单

下表中的工程假设必须写入 `configs/sim_reaction_wheel.yaml` 和 `metadata.json`，不能隐含在代码中。

| 参数/假设 | 推荐默认 | 推荐范围 | 是否敏感性分析 | 来源类型 | 说明 |
|---|---:|---:|---|---|---|
| `lubricant_capacity_norm` | `1.0` | `1.0` | 否 | 工程归一化 | 第一版使用归一化容量；物理 mL 仅作为可选换算 |
| `lubricant_threshold` | `0.05` | `0.05~0.10` | 是 | 工程假设 | 润滑 HI 失效阈值，文献未给通用固定值 |
| `friction_threshold` | 按训练分位数设定 | `P95~P99` | 是 | 工程假设 | 摩擦代理异常阈值，避免硬编码无依据绝对值 |
| `alpha_c` | `1.0` | `0.5~3.0` | 是 | 工程假设 | 润滑损伤对 Coulomb 摩擦的放大 |
| `alpha_s` | `1.5` | `0.5~4.0` | 是 | 工程假设 | 润滑损伤对静摩擦/Stribeck 项的放大 |
| `alpha_v` | `0.5` | `0.0~2.0` | 是 | 工程假设 | 长期损伤对黏性项的放大，不等同于温度黏度系数 |
| `fault_injection_ratio` | `0.25` | `0.20~0.30` | 是 | 工程假设 | 保证显式故障有足够样本但不主导全数据 |
| `missing_rate` | `0.05` | `0.05~0.20` | 是 | 文献启发 | `4.md` 讨论缺失和插值鲁棒性 |
| `noise_std_ratio` | `0.01` | `0.005~0.02` | 是 | 文献启发 | 高斯白噪声模拟遥测不确定性 |
| `temperature_bias` | `0°C` | `-5~5°C` | 可选 | 工程假设 | 单元级热环境差异 |
| `omega_low` | 低速阈值 | `0~500 rpm` 量级 | 是 | `5.md` 启发 | 低速/过零摩擦和不稳定更明显 |

---

## 6. 标签定义

### 6.1 EOL 时间

每个单元的失效时间定义为最早满足任一失效条件的时间：

```text
t_eol = min(
  first_time(kt_ratio <= 0.3),
  first_time(lubricant_hi <= lubricant_threshold),
  first_time(friction_torque >= friction_threshold),
  sequence_end_if_no_threshold_reached
)
```

如果某单元在截断长度内没有达到阈值，则：

- 要么将最后一个时间步作为 censoring/end-of-record；
- 要么通过采样参数保证所有单元在观测窗口内接近或达到 EOL。

第一版建议保证多数单元有清晰 EOL，否则 RUL 监督标签会不稳定。可保留少量 censored 单元作为鲁棒性测试，但必须在 `is_censored` 字段中标记。

### 6.2 RUL

```text
rul[t] = max(t_eol - t, 0)
normalized_rul[t] = rul[t] / max(rul[0], eps)
```

### 6.3 退化进度和阶段标签

```text
degradation_progress[t] = 1 - normalized_rul[t]
```

阶段标签采用确定性 hard bins：

| stage | progress 区间 | 含义 |
|---:|---|---|
| 0 | `[0.0, 0.3)` | healthy / early |
| 1 | `[0.3, 0.7)` | degradation |
| 2 | `[0.7, 1.0]` | severe / near failure |

这与我们现有 stage-aware LMMD v1 设计一致，避免 GMM 伪标签引入不稳定。

---

## 7. 输出字段设计

### 7.1 CSV 主表字段

建议输出：

| 字段 | 类型 | 说明 | 默认是否模型输入 |
|---|---|---|---|
| `unit_id` | str/int | 虚拟反作用轮编号 | 分组字段 |
| `time_step` | int | 低频采样步 | 可选 |
| `mission_time_s` | float | 物理时间，可选 | 否/可选 |
| `command_voltage` | float | 指令电压 | 是 |
| `wheel_speed` | float | 轮速 rad/s | 是 |
| `wheel_speed_rpm` | float | 轮速 rpm | 可选，不与 rad/s 同时输入 |
| `motor_current` | float | 电机电流 | 是 |
| `temperature` | float | 可观测壳体/轴承温度代理 | 是 |
| `friction_torque_proxy` | float | 由电流/速度估计的摩擦代理 | 默认否；仅 `Telemetry + estimated HI` 协议使用 |
| `vibration_proxy` | float | 低频不稳定代理 | 是 |
| `torque_noise_proxy` | float | 低频力矩扰动代理 | 可选 |
| `kt` | float | 真实/仿真健康状态 | 默认不作为主输入 |
| `kt_ratio` | float | `kt / kt0` | 默认不作为主输入 |
| `lubricant_loss` | float | 累计润滑损耗 | 默认不作为主输入 |
| `lubricant_hi` | float | 润滑健康指标 | 默认不作为主输入 |
| `degradation_progress` | float | 退化进度 | 对齐/分析，不作普通输入 |
| `health_stage` | int | 阶段标签 | LMMD 对齐可用 |
| `rul` | float | 主标签 | 标签 |
| `normalized_rul` | float | 归一化 RUL | 标签 |
| `base_degradation` | str | 基础退化机制，默认 `kt_exponential_decay` | 元数据 |
| `explicit_fault_injection_type` | str | 显式故障注入类型 | 元数据 |
| `is_censored` | bool | 是否未观测到明确 EOL | 元数据 |
| `split` | str | train/val/calib/test | 元数据 |
| `missing_mask_*` | bool | 缺失值位置 | 可选 |

### 7.2 metadata.json

必须记录：

- 生成时间、代码版本、随机种子。
- 文献参数来源表。
- 所有工程假设和可配置阈值。
- 单元数量、长度分布、显式故障注入类型分布。
- train/val/calib/test 的 unit_id 划分。
- 噪声水平、缺失率、插值方式。
- EOL 条件和 RUL 计算公式。
- 是否存在 censored units。

### 7.3 README.md

必须说明：

- 数据集不是飞行实测数据，而是机理约束仿真遥测数据。
- 如何一键复现生成。
- 每个字段含义。
- 默认模型输入列和标签列。
- `friction_torque_proxy`、`kt_ratio`、`lubricant_hi` 等代理/健康状态字段的使用边界。
- 与 XJTU-SY 源域迁移的关系。
- 大赛合规口径。

---

## 8. 噪声、缺失与鲁棒性设置

`4.md` 明确在合成数据中加入 Gaussian white noise，并随机删除数据以模拟传感器缺失，再用线性插值恢复。其缺失比例敏感性测试覆盖 `5%` 到 `30%`。

第一版建议：

| 项 | 默认设置 |
|---|---|
| 高斯噪声 | 每个输入字段按字段标准差的 `0.5%~2%` 加噪 |
| 缺失点比例 | `5%` 默认，敏感性可测 `10%`, `20%` |
| 缺失片段 | 少量连续短片段缺失，模拟遥测丢包 |
| 插值方式 | 线性插值 |
| 缺失掩码 | 保留 `missing_mask_*` 字段或 metadata 统计 |

重要：噪声应加在可观测遥测字段上，不应破坏真实标签 `rul` 的确定性。

---

## 9. 数据规模和划分

建议第一版：

| 项 | 设置 |
|---|---|
| 虚拟单元数 | `100` |
| 每单元低频长度 | `600~1600` 时间步 |
| 训练/验证/校准/测试 | `60/15/10/15` 或 `70/15/15` |
| 划分方式 | 严格按 `unit_id` 划分 |
| 随机种子 | 固定，例如 `42`、`2026` |
| 故障注入比例 | `20%~30%` 单元显式故障注入 |

`100` 个虚拟单元的选择依据：

- train/val/calib/test 四划分需要每个 split 都有足够 unit，尤其 conformal calibration 至少需要约 10 个独立单元。
- 显式故障注入比例为 `20%~30%` 时，100 单元能保证两类主故障各有若干样本，避免某类故障仅出现 1~2 个单元。
- C-MAPSS FD001 train/test 合计约 200 台发动机，100 单元属于较轻量但统计上可用的合成目标域规模。
- 可在后续敏感性中设置 `30 / 50 / 100` 三档，验证结论是否依赖数据规模。

如果启用 conformal prediction，建议必须独立划分 calibration units：

```text
train units != validation units != calibration units != test units
```

不能从同一 unit 的不同窗口同时进入 train 和 test。

---

## 10. 与 XJTU-SY 源域的映射关系

XJTU-SY 源域已处理为低频 HI：

```text
2 channels x 7 statistics = 14-dimensional HI
```

`reaction_wheel_sim` 的默认输入也应保持低频遥测/健康指标，而非高频原始信号。推荐迁移输入映射：

| XJTU-SY HI 概念 | reaction_wheel_sim 对应概念 |
|---|---|
| RMS / STD | 电流、速度、摩擦代理的局部波动 |
| skewness / kurtosis | 异常脉冲、低速不稳定代理 |
| peak-to-peak | 电流/振动代理波动幅度 |
| crest factor | torque/vibration transient |
| trend slope | 温度、摩擦、`kt_ratio` 或润滑 HI 的趋势 |

因此生成器可以额外输出窗口级 HI 文件：

```text
data/processed/reaction_wheel_sim_hi.npz
```

或者在 loader 中对 CSV 动态提取窗口统计特征。第一版优先输出 CSV，随后由 loader/windowing 生成模型输入。

### 10.1 迁移合理性与局限性

必须正面承认：XJTU-SY Bearing 与反作用轮轴承不是物理失效机理同构。

| 维度 | XJTU-SY Bearing | reaction_wheel_sim |
|---|---|---|
| 场景 | 地面轴承加速寿命试验 | 航天反作用轮低频遥测仿真 |
| 主导失效 | 疲劳剥落、振动突变 | 润滑损耗、摩擦增强、cage instability |
| 原始信号 | 高频径向振动 | 转速、电流、温度、代理 HI |
| 迁移层面 | 低频统计 HI | 低频遥测/健康代理 |

因此本项目不假设两域机理完全相同，而是假设两域低频 HI 的退化轨迹形态存在可迁移结构，例如早期平稳、中期缓慢退化、后期加速恶化、接近 EOL 时波动和非高斯统计特征增强。该假设必须通过实验验证，而不是只靠文字论证。

迁移价值主要应在以下条件下体现：

- 目标域标签较少或仅有少量 target units。
- source-only 明显差于 target-supervised。
- stage-aware LMMD 或 HI-weighted LMMD 相比 naive transfer 更稳定。
- 若 target-only 在充足目标标签下已经很强，则报告中应明确说明迁移收益主要面向小样本航天目标域适配。

---

## 11. 评估指标与实验用途

### 11.1 目标域监督实验

用于验证 `reaction_wheel_sim` 本身是否可学习：

- P-TCN target-only
- P-SA-MCD-TCN target-only
- GRU/Transformer baseline 可选

默认应设置两套输入协议，防止 `friction_torque_proxy` 被误认为 oracle 信息：

| 协议 | 输入列 | 用途 |
|---|---|---|
| Telemetry-only | `wheel_speed`, `motor_current`, `command_voltage`, `temperature`, `vibration_proxy` | 主基线，最接近普通遥测预测 |
| Telemetry + estimated HI | Telemetry-only + `friction_torque_proxy` | 增强实验，需声明摩擦代理由可观测电流/速度估计，不是真实隐藏摩擦标签 |

默认主报告以 Telemetry-only 为公平基线；with-proxy 结果单独成表，不能混作同一设置。

指标：

- RMSE
- MAE
- NASA Score 或类 NASA Score
- RA
- PH
- alpha-lambda accuracy
- conformal coverage
- average interval width
- 参数量和推理延迟

评估口径与 C-MAPSS 阶段保持一致：unit-level last-window 或 last-k-window RMSE/Score 作为主报告口径，full-sequence/dense metrics 作为诊断指标同步报告。这样可避免长序列早期健康样本过多而稀释临近失效阶段误差。若使用 conformal prediction，calibration units 必须与 train/test units 独立。

### 11.2 跨域迁移实验

主线实验：

```text
source: XJTU-SY low-frequency HI
target: reaction_wheel_sim
```

对比：

| 方法 | 作用 |
|---|---|
| target-only P-TCN | 目标域监督基线 |
| source-only | 检验不适配时的域差距 |
| P-TCN + LMMD | 简单迁移基线 |
| P-SA-MCD-TCN + stage-aware LMMD | 主方法 |
| P-SA-MCD-TCN + HI-weighted stage-aware LMMD | 吸收 HIWSAN 思想的增强版 |

---

## 12. 从既有实验吸取的经验

### 12.1 不盲目增加训练轮数

XJTU-SY B1 跨工况实验显示，150 epochs 变体全部比 50 epochs 更差，说明小样本跨域迁移中长训可能导致过拟合或迁移退化。`4.md` 也指出其 LSTM 状态预测/参数预测损失在约 50~60 epochs 后平台化。

因此反作用轮实验建议：

```text
epochs <= 50~80
启用 validation 最优 checkpoint
优先调数据质量、阶段标签和 LMMD 权重，而不是单纯拉长训练
```

### 12.2 小模型可能更稳

XJTU-SY B1 中，小模型 `h32_w32_lmmd005_50e` 的 MAE/Score 最好，P-TCN 的 RMSE 最好。这说明反作用轮目标域第一版也应保留 P-TCN 和小容量 P-SA-MCD-TCN 作为稳健 baseline。

### 12.3 HI/阶段对齐比原始信号迁移更重要

论文9 HIWSAN 的核心思想是 HI 加权和子域对齐。我们的反作用轮目标域应明确输出：

```text
degradation_progress
health_stage
lubricant_hi / kt_ratio
```

用于后续 stage-aware LMMD 或 HI-weighted LMMD，而不是只输出普通遥测。

---

## 13. 必须执行的数据质量检查

生成脚本完成后，必须自动输出以下检查结果：

### 13.1 结构检查

- `unit_id` 数量是否等于配置。
- 每个 unit 是否按时间升序。
- train/val/calib/test 是否按 unit 严格隔离。
- 缺失值插值后是否无 NaN/inf。
- 标签列是否存在并范围合理。

### 13.2 物理合理性检查

- `kt_ratio` 是否总体下降。
- `lubricant_loss` 是否总体上升。
- `normalized_rul` 是否从接近 1 下降到 0。
- 后期 `temperature`、`friction_torque_proxy`、`vibration_proxy` 是否整体更高。
- 低速/过零附近是否更容易出现 friction/vibration 异常。

### 13.3 统计检查

- 每个 `explicit_fault_injection_type` 的数量。
- 每个 split 的 unit 数和样本数。
- RUL 分布是否覆盖 early/middle/late。
- stage 分布是否严重不平衡。
- 输入特征均值/方差是否存在异常尺度。

### 13.4 泄漏检查

默认训练输入不能包含：

```text
rul
normalized_rul
degradation_progress
is_censored
split
future-derived EOL fields
```

如果使用 `kt_ratio` 或 `lubricant_hi` 作为输入，必须在实验中明确标注为“可观测/估计 HI 输入”或“oracle auxiliary input”，不能和普通遥测基线混淆。

如果使用 `friction_torque_proxy` 作为输入，必须单独报告为 `Telemetry + estimated HI` 协议；默认 `Telemetry-only` 协议不得输入该字段。

---

## 14. 交叉审阅重点问题

请其他模型/审阅者重点检查以下问题：

### 14.1 文献依据是否充分

- `k_t0 = 0.029 Nm/A` 和 `kt_eol = 0.3 * kt0` 是否引用充分？
- 摩擦模型采用 ITHACO 参数 + Carrara Stribeck 结构是否合理？
- 润滑损耗公式中温度单位是否需要统一为 Kelvin？
- `lubricant_threshold = 0.05/0.10` 是否应作为工程假设标明？
- `omega_s = 680 rad/s` 和 `K_s = 95 V/(rad/s)` 的采用口径是否足够清楚？
- `B = 0.22 Nm` 作为周期扰动代理是否比区分 cogging/ripple 更适合第一版？

### 14.2 机理是否过度简化

- 是否需要完整 Bialke 模型，还是当前低复杂度机理约束模型足够？
- `friction_torque_proxy` 是否应作为输入，还是只作为隐藏状态/辅助标签？
- `vibration_proxy` 的生成是否足以代表 `3.md` 中 cage instability？
- 温度导致黏性下降与润滑损伤导致摩擦恶化的拆分是否合理？

### 14.3 大赛合规性

- Python 自研仿真引擎是否符合“卫星仿真或模拟软件”口径？
- 是否需要在报告中更明确说明“非飞行实测数据”？
- 是否覆盖了“长期运行、故障注入、退化演化场景”？

### 14.4 迁移学习适配

- 输出字段是否足以支撑 `XJTU-SY -> reaction_wheel_sim`？
- stage 标签是否应由 RUL progress 生成，还是由 `kt/lubricant_hi` 融合生成？
- 是否需要额外输出窗口级 HI 以对齐 XJTU-SY 的 14 维 HI？
- 对 XJTU 疲劳剥落与反作用轮润滑退化之间机理差异的解释是否充分？

### 14.5 可复现性

- 是否所有随机项都有 seed？
- 是否 metadata 足以重建数据？
- 是否需要同时输出 `config.yaml` 快照？
- 是否需要提供 `check_reaction_wheel_sim.py` 自动检查脚本？

---

## 15. 建议的第一版实现文件

交叉审阅通过后，建议实现：

```text
configs/sim_reaction_wheel.yaml
scripts/generate_reaction_wheel_sim.py
scripts/check_reaction_wheel_sim.py
src/xa202608/data/reaction_wheel.py
data/simulated/reaction_wheel/reaction_wheel_sim.csv
data/simulated/reaction_wheel/metadata.json
data/simulated/reaction_wheel/README.md
```

生成命令建议：

```text
D:\anaconda\envs\jiebang\python.exe scripts/generate_reaction_wheel_sim.py --config configs/sim_reaction_wheel.yaml
D:\anaconda\envs\jiebang\python.exe scripts/check_reaction_wheel_sim.py --data data/simulated/reaction_wheel/reaction_wheel_sim.csv
```

---

## 16. 与顶层执行文档的承接

本文档与 `XA-202608模型与数据集最终选择.md` 的关系如下：

| 顶层文档口径 | 本文档落实方式 |
|---|---|
| 自建目标域必须包含 `reaction_wheel_sim` 和 `satellite_battery_sim` | 本文档只细化第一个目标域 `reaction_wheel_sim` |
| Python 自研机理约束仿真引擎 | 本文采用可配置公式、固定 seed、CSV/NPZ、metadata 和 README 的实现口径 |
| 反作用轮第一版保留 2 类关键故障 | 本文将 `kt_exponential_decay` 定义为基础退化，只保留 `lubrication_accelerated_loss` 与 `friction_step` 两类显式故障注入 |
| stage-aware LMMD 使用 hard bins `[0.0, 0.3, 0.7, 1.0]` | 本文 `health_stage` 由 `degradation_progress = 1 - normalized_rul` 确定性生成 |
| XJTU-SY Bearing → reaction_wheel_sim 不迁移原始高频振动 | 本文只使用低频 HI / 遥测统计层映射，并显式说明两域机理差异 |
| 指标口径延续 C-MAPSS 阶段经验 | 本文将 last-window / last-k-window 作为主报告，full-sequence 作为诊断 |

---

## 17. 本设计的最终立场

第一版 `reaction_wheel_sim` 应采用如下定位：

```text
基于公开反作用轮/航天轴承润滑文献的机理约束低频遥测退化仿真数据集。
```

其核心机理为：

```text
1. motor torque constant kt 指数退化，30% nominal 为 EOL 阈值；
2. 润滑温度驱动的累计润滑损耗；
3. Coulomb + Stribeck 摩擦随长期润滑损伤增强，viscous 项拆分为温度黏度效应与长期损伤放大效应；
4. 低速/过零区引入力矩和振动代理异常；
5. 噪声、缺失和插值模拟真实遥测不确定性。
```

其核心用途为：

```text
验证 XJTU-SY Bearing 公开源域到航天反作用轮仿真目标域的跨领域 RUL 迁移能力。
```

如果交叉审阅未发现根本问题，下一步即可进入代码实现阶段。
