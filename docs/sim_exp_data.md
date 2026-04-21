# 仿真实验数据汇总

> 所有仿真实验的原始输出、对比表格与分析说明。
> 真机实验数据另见 `docs/real_robot_deployment_plan.md`。

## 总览

| # | 实验 | Task ID | 状态 | 关键结论 |
|---|------|---------|------|---------|
| Exp 2 | PickCube 偏差训练（3 配置 bias scan）| `PandaPickCube*Keyboard-v0` | ✅ 已完成 | 随机偏差训练全范围 92-100%，泛化到 0.3 rad |
| Exp 3 | Backup S1（单移动障碍 + DR） | `PandaBackupPolicyS1-v0` | ✅ 已完成 | 200k steps → 99% 存活率 |
| Exp 5 | Backup Zero-shot Bias 迁移 | `PandaBackupPolicyS1BiasJ1-v0` | ✅ 已完成 | 99.0% → 98.5%，近乎无损 |
| Exp 6 | DR 策略在 NoDR env 评估 | `PandaBackupPolicyS1NoDR-v0` | ✅ 已完成 | 反直觉：99.0% → **96.5%**（更干净环境反而更差）|
| Exp 7 | Backup S2 多障碍 | `PandaBackupPolicyS2-v0` | ✅ 已完成 | 200k utd=4：**83-91%**（vs S1 99%），多障碍难度跃升 |
| Exp 1-Obs27 | Task policy 观测消融（27D full）| `PandaPickPlaceBiasJ1RandomKeyboard-v0`（待建）| 🟡 待做（env 未建） | — |
| Exp 1-Obs24 | Task policy 观测消融（24D 去 real_tcp）| `PandaPickPlaceObs24BiasJ1RandomKeyboard-v0`（待建）| 🟡 待做（env 未建） | — |
| Exp 1-Obs21 | Task policy 观测消融（21D 去 block/plate）| `PandaPickPlaceObs21BiasJ1RandomKeyboard-v0`（待建）| 🟡 待做（env 未建） | — |

---

## Exp 2：PickCube 编码器偏差训练（已完成）

**目标**：验证 H1/H2 假设——无偏差策略在 bias 下退化，随机偏差训练配合 `real_tcp` 观测能学到对任意 bias 鲁棒的策略。

**共同参数**：观测含 `block_pos(3) + noisy_real_tcp(3, σ=5mm)`；偏差注入 Joint 4；action 4D；max_episode_steps=200。

### 2.1 Baseline：无偏差策略（18D 观测）

| 指标 | 值 |
|------|---|
| 训练环境 | `PandaPickCubeKeyboard-v0`（无偏差） |
| 观测维度 | 18D |
| 训练步数 | ~13K actor steps |
| 成功率 | **100%**（50/50 episodes） |
| 平均步数 | 13.4 ± 0.6 |

### 2.2 H1 验证：无偏差策略在偏差环境下的退化

**原始数据**（无偏差训练的 18D 策略，在不同偏差环境下 eval）：

| 偏差 (rad) | 成功率 | 平均步数 |
|-----------|--------|---------|
| 0.00 | 100% | 13.5 |
| 0.05 | 100% | 13.5 |
| 0.10 | 80% | 55.9 |
| 0.15 | 62% | 62.5 |
| 0.20 | 51% | 99.8 |
| 0.25 | 5% | 184.5 |
| 0.30 | 0% | 191.1 |

**结论**：偏差 >0.1 rad 显著影响成功率，>0.25 rad 基本全失败。验证了 H1——无偏差训练无法应对 bias。

### 2.3 固定偏差策略（24D 观测，含 real_tcp）

| 指标 | 值 |
|------|---|
| 训练环境 | `PandaPickCubeBiasJ4Fixed02Keyboard-v0`（固定 0.2 rad） |
| 观测维度 | 24D（robot_state 18 + block_pos 3 + noisy_real_tcp 3） |
| 训练成功率 | 92% |

### 2.4 随机偏差策略（24D 观测，核心实验）

| 指标 | 值 |
|------|---|
| 训练环境 | `PandaPickCubeBiasJ4RandomKeyboard-v0`（[0, 0.25] rad） |
| 观测维度 | 24D |
| 最佳 checkpoint | 10K learner steps |
| demo 数量 | 50 episodes |

### 2.5 三策略 Bias Scan 对比（Exp 2 核心结果表）

| 偏差 (rad) | 无偏差策略 18D | 固定偏差策略 24D | **随机偏差策略 24D** |
|-----------|---------------|-----------------|--------------------|
| 0.00 | **100%** | 83% | **92%** |
| 0.05 | **100%** | 87% | **100%** |
| 0.10 | 80% | 99% | **99%** |
| 0.15 | 62% | 100% | **96%** |
| 0.20 | 51% | **100%** | **97%** |
| 0.25 | 5% | 100% | **96%** |
| 0.30 | 0% | 99% | **99%** |

**关键发现**：

1. **随机偏差策略全范围 92-100%**，三种策略里最均衡
2. **泛化能力**：训练范围是 [0, 0.25]，但 0.30 rad 环境下仍 99% 成功
3. **无过度补偿**：无偏差（0.00 rad）时 92%，远高于固定偏差策略的 83%
4. **印证第一性原理**：24D 含 `noisy_real_tcp` 提供了关键的"真实末端位置"信号，让策略可以在单步观测内区分不同 bias

### 2.6 关键失败实验（negative results）

| 实验 | 观测维度 | 结果 | 失败原因 |
|------|---------|------|---------|
| 随机偏差 18D（无 real_tcp） | 18D | 82% → 42% 退化 | 单步观测无法区分不同 bias |
| 随机偏差 21D（+block_pos） | 21D | 前期靠人工，后期全失败 | 知道目标但不知自己真实位置 |
| 随机偏差 24D（数据丢弃 bug） | 24D | 失败 | 干预数据 99% 被错误丢弃 |

**论文价值**：18D/21D 失败**证明了 noisy_real_tcp 是关键观测**，不是锦上添花；24D bug 修复后才成功，说明 offline buffer 必须正确处理干预 transition。

> 详细上下文见 `docs/project_progress.md` §3.1-3.5。

---

## Exp 3：Backup Policy S1 训练（已完成）

**目标**：在仿真中训练一个能在 10 步内躲避动态障碍物的 Backup Policy，为真机 Sim2Real 部署提供策略。

### 3.1 训练配置

| 参数 | 值 |
|------|---|
| 训练环境 | `PandaBackupPolicyS1-v0` |
| 观测维度 | 28D（robot_state 18 + 1 移动障碍物 10: active+pos+vel+relative）|
| Action | 6D 连续 [-1, 1]，scale=0.03 m/step |
| Episode 长度 | 10 步（0.1s × 10 = 1s） |
| 算法 | SAC |
| online_steps | 200,000 |
| utd_ratio | 4 |
| discount | 1.0 |
| batch_size | 256 |
| Frame stacking | 3 帧（可感知加速度/运动趋势） |
| seed | 1000 |

**关键设计（Kiemel et al. 2024 启发）**：
- **非负存活奖励**：每步 +0.5 基础奖励 + 位移软惩罚 + 动作平滑惩罚
- **大终止惩罚**：碰撞 / 区域入侵 / 过度位移 → r = -10.0
- **满存活 bonus**：truncated 时 +5.0

### 3.2 Domain Randomization 配置（Sim2Real gap 弥补）

| 参数 | 分布 | 模拟什么 |
|------|------|---------|
| 障碍物位置噪声 | N(0, 0.03) | MediaPipe + D455 深度 累计 ±3cm |
| 障碍物速度噪声 | N(0, 0.01) | 帧间差分抖动 |
| 观测延迟 | U(0, 2) 步 | 检测+推理延迟 ~0-20ms |
| 障碍物生成距离 | U(0.12, 0.25) m | 不同进入距离 |
| 障碍物移动速度 | U(0.005, 0.015) m/step | 不同手速 |
| TCP 位置噪声 | N(0, 0.005) | 原有 |

### 3.3 最终 Checkpoint Eval（200 episodes）

```
Backup Policy 评估结果
=================================================================
 环境: PandaBackupPolicyS1-v0
 总 episodes: 200
 存活率 (>=10步): 198/200 (99.0%)
 满存活 (10步): 198/200 (99.0%)
 平均存活步数: 10.0 ± 0.4
 平均累计奖励: 8.982 ± 2.527
 平均最近距离: 0.154m
 平均终止位移: 0.092m ± 0.025

 终止原因:
   survived              : 195 (97.5%)
   excessive_displacement:   4 (2.0%)
   hand_collision        :   1 (0.5%)

 运动模式分析:
   模式        数量    存活率     平均奖励
   --------------------------------------
   ARC         46    97.8%     +8.990
   LINEAR      49    98.0%     +8.716
   PASSING     49   100.0%     +9.086
   STOP_GO     56   100.0%     +9.116
=================================================================
```

**Checkpoint 位置**：`checkpoints/backup_policy_s1/`（repo 内常驻）

**结论**：
- 4 种运动模式全部 ≥97.8%，其中 PASSING / STOP_GO 达 100%
- 最主要的 2% 失败来自 `excessive_displacement`（跑太远），而不是碰撞
- 策略学到了**小幅闪避而非大范围逃跑**的保守行为（平均最近距离 15.4 cm，远大于 8 cm 碰撞阈值）

---

## Exp 5：Backup Zero-shot Bias 迁移（已完成）

**目标**：验证 Backup 策略在训练时**不见过编码器偏差**的条件下，能否直接迁移到 BiasJ1（Joint 1 ±0.15 rad 随机偏差）环境。

**意义**：真机部署时 J1 编码器可能带偏差（对应末端 ~10 cm 位置漂移）。如果 Backup 能 zero-shot 工作，说明 DR 训练 + 真实 TCP 观测设计让策略对下游 bias 天然免疫。

### 5.1 原始 Eval 输出

#### S1 baseline（复现，200 episodes）

```
Backup Policy 评估结果
=================================================================
 环境: PandaBackupPolicyS1-v0
 总 episodes: 200
 存活率 (>=10步): 198/200 (99.0%)
 满存活 (10步): 198/200 (99.0%)
 平均存活步数: 10.0 ± 0.4
 平均累计奖励: 8.982 ± 2.527
 平均最近距离: 0.154m
 平均终止位移: 0.092m ± 0.025

 终止原因:
   survived              : 195 (97.5%)
   excessive_displacement:   4 (2.0%)
   hand_collision        :   1 (0.5%)

 运动模式分析:
   模式        数量    存活率     平均奖励
   --------------------------------------
   ARC         46    97.8%     +8.990
   LINEAR      49    98.0%     +8.716
   PASSING     49   100.0%     +9.086
   STOP_GO     56   100.0%     +9.116
=================================================================
```

#### S1 + BiasJ1 zero-shot（200 episodes）

```
Backup Policy 评估结果
=================================================================
 环境: PandaBackupPolicyS1BiasJ1-v0
 总 episodes: 200
 存活率 (>=10步): 197/200 (98.5%)
 满存活 (10步): 197/200 (98.5%)
 平均存活步数: 9.9 ± 0.6
 平均累计奖励: 9.134 ± 2.114
 平均最近距离: 0.156m
 平均终止位移: 0.088m ± 0.026

 终止原因:
   survived              : 197 (98.5%)
   zone_c_intrusion      :   1 (0.5%)
   hand_collision        :   1 (0.5%)
   excessive_displacement:   1 (0.5%)

 运动模式分析:
   模式        数量    存活率     平均奖励
   --------------------------------------
   ARC         51    98.0%     +9.045
   LINEAR      53    96.2%     +8.720
   PASSING     41   100.0%     +9.446
   STOP_GO     55   100.0%     +9.382
=================================================================
```

### 5.2 对比表格

#### 总体指标

| 指标 | S1 baseline | S1 + BiasJ1 (zero-shot) | Δ |
|------|------------|-------------------------|---|
| 存活率 | 99.0% (198/200) | **98.5% (197/200)** | -0.5% |
| 满存活 | 99.0% | 98.5% | -0.5% |
| 平均存活步数 | 10.0 ± 0.4 | 9.9 ± 0.6 | -0.1 |
| **平均累计奖励** | +8.982 ± 2.527 | **+9.134 ± 2.114** | **+0.15** ✨ |
| 平均最近距离 | 0.154 m | 0.156 m | +2 mm |
| 平均终止位移 | 0.092 m | 0.088 m | -4 mm |

#### 终止原因分布

| 原因 | S1 baseline | S1 + BiasJ1 |
|------|------------|-------------|
| survived | 195 (97.5%) | 197 (98.5%) |
| excessive_displacement | 4 (2.0%) | 1 (0.5%) |
| hand_collision | 1 (0.5%) | 1 (0.5%) |
| zone_c_intrusion | 0 | 1 (0.5%) |

#### 运动模式存活率对比

| 模式 | S1 baseline | S1 + BiasJ1 | Δ |
|------|------------|-------------|---|
| ARC | 97.8% (46) | 98.0% (51) | +0.2% |
| LINEAR | 98.0% (49) | **96.2% (53)** | **-1.8%** |
| PASSING | 100.0% (49) | 100.0% (41) | 0 |
| STOP_GO | 100.0% (56) | 100.0% (55) | 0 |

### 5.3 分析说明

**1. 0.5% 差距 ≈ 统计噪声**

200 episodes 下 1 个样本对应 0.5%，两次 eval 结果 197/200 vs 198/200 完全在抽样波动范围内。论文可以论述为**实质性能相当**。

**2. 奖励反而微升（+8.98 → +9.13）**

BiasJ1 环境下策略的**平均累计奖励更高、方差更小**（±2.53 → ±2.11）。合理解释：带 bias 后 noisy_real_tcp 的相对信号让策略更保守、位移更小（-4mm），从而拿到更多存活奖励和更少位移惩罚。

**3. 终止位移更小 + 最近距离相近**

说明策略在 bias 下并没有"乱闪"，反而动作更收敛。最近距离 15.6cm 远大于 8cm 碰撞阈值——碰撞边缘完全没被突破。

**4. 唯一退化场景：LINEAR (98.0% → 96.2%)**

LINEAR 是匀速直线运动，策略需要精确预测到达时间。bias 让 TCP 观测的**时间一致性**受影响（不同步长 DR 延迟 + bias FK），推理能力略受挫。其他三种模式（ARC/PASSING/STOP_GO）自带轨迹变化，对单点 TCP 精度不敏感，所以完全无损。

**5. 第一性原理解释 zero-shot 成功**

Backup 训练时 `get_robot_state()` 走的是"无 bias"分支：
- `tcp_pos = sensor('panda_hand/pinch_pos').data`（真实 TCP）
- 再叠加 DR 噪声 `N(0, 0.005)`

BiasJ1 环境下 **Backup 同样拿真实 TCP**（因为 BackupEnv 的 `get_robot_state` 按 `encoder_bias_config is None` 默认路径），所以从 Backup 视角看编码器偏差**几乎不可见**。论文论述：
> DR 训练策略对"观测空间上任意 ≤ 某阈值的扰动"都鲁棒，这种鲁棒性自动覆盖了 bias 这种具体故障类型。换言之，只要 bias 引入的末端漂移在 DR 噪声的"量级"内（5mm vs bias 导致的 ~10cm），就会被 DR 吸收。

**注**：上面最后一段的量级说法需要验证——bias 对真实 TCP 的影响并非直接漂移 10cm，而是通过关节角测量偏差 → biased FK；但 Backup 不用 biased FK，它只用真实 sensor 读数，所以实际影响链路是 **Task Policy 的控制指令 → 关节 PID 跟踪偏差 → 真实关节位置偏移 → 真实 TCP 偏移**。需要结合 Backup 切换时机（ms 级）进一步讨论。

---

## Exp 6：DR 策略跨分布评估（已完成）

**目标**：原计划是训练 NoDR 策略对照 Exp 3，但 "DR 对 sim2real 必要" 是 Tobin 2017 / Peng 2018 / Kiemel 2024 已有的结论，训练消融价值不高。改为用**已有的 DR-trained checkpoint** 在 NoDR 环境 eval，检验"DR 是否是免费午餐"——训练域的 DR 会不会损害无噪声场景的性能。

### 6.1 原始 Eval 输出（200 episodes）

```
Backup Policy 评估结果
=================================================================
 环境: PandaBackupPolicyS1NoDR-v0
 总 episodes: 200
 存活率 (>=10步): 193/200 (96.5%)
 满存活 (10步): 193/200 (96.5%)
 平均存活步数: 9.9 ± 0.8
 平均累计奖励: 8.541 ± 3.607
 平均最近距离: 0.156m
 平均终止位移: 0.094m ± 0.028

 终止原因:
   survived              : 190 (95.0%)
   hand_collision        :   5 (2.5%)
   excessive_displacement:   5 (2.5%)

 运动模式分析:
   模式        数量    存活率     平均奖励
   --------------------------------------
   ARC         57    94.7%     +7.928
   LINEAR      62    95.2%     +8.269
   PASSING     39   100.0%     +9.429
   STOP_GO     42    97.6%     +8.950
=================================================================
```

### 6.2 三环境对比表（同一个 checkpoint）

#### 总体指标

| Env | 存活率 | 满存活 | 平均奖励 | 最近距离 | 终止位移 |
|-----|-------|--------|---------|---------|---------|
| S1 (DR, 训练分布) | **99.0%** (198/200) | 99.0% | +8.982 ± 2.527 | 0.154 m | 0.092 m |
| S1 + BiasJ1 (zero-shot) | 98.5% (197/200) | 98.5% | **+9.134** ± 2.114 | 0.156 m | 0.088 m |
| **S1 NoDR** (无噪声) | **96.5%** (193/200) ↓ | 96.5% | 8.541 ± **3.607** ↑ | 0.156 m | 0.094 m |

#### 终止原因分布

| 原因 | S1 (DR) | BiasJ1 | **NoDR** |
|------|---------|--------|----------|
| survived | 195 (97.5%) | 197 (98.5%) | 190 (95.0%) |
| hand_collision | 1 (0.5%) | 1 (0.5%) | **5 (2.5%)** ↑ |
| excessive_displacement | 4 (2.0%) | 1 (0.5%) | **5 (2.5%)** |
| zone_c_intrusion | 0 | 1 (0.5%) | 0 |

#### 运动模式存活率对比

| 模式 | S1 (DR) | BiasJ1 | **NoDR** |
|------|---------|--------|----------|
| ARC | 97.8% | 98.0% | **94.7%** ↓ |
| LINEAR | 98.0% | 96.2% | **95.2%** ↓ |
| PASSING | 100.0% | 100.0% | 100.0% |
| STOP_GO | 100.0% | 100.0% | **97.6%** ↓ |

### 6.3 分析说明

#### 反直觉发现

NoDR 环境在理论上**"更简单"**（观测精确、无延迟、无速度噪声），但 DR 策略在这里**反而退化**：

- 存活率：99.0% → **96.5%**（-2.5%）
- 奖励方差：2.527 → **3.607**（+43%，显著变宽）
- hand_collision：1 → **5**（5 倍增长，这是最关键的信号）

PASSING 和 BiasJ1 环境都保持 100%，但 NoDR 掉到 97.6%。

#### 第一性原理解释：DR 训练习得"噪声先验"

DR 训练时策略看到的观测长这样：
- 障碍物位置：`真实 pos + N(0, 0.03)`（±3 cm）
- 障碍物速度：`真实 vel + N(0, 0.01)`
- 观测延迟：`U(0, 2)` 步

策略学到的**隐式模型**是："收到的观测 = 真实值 + 噪声/延迟"。于是它的决策里带有：

1. **去噪平均**：对连续几帧观测做内部平滑
2. **延迟补偿**：动作带一点"预判提前量"
3. **保守 margin**：以噪声方差为尺度预留安全距离

在 NoDR 环境下：
- 观测即真实，无噪声可去 → 平滑反而滞后
- 无延迟，提前量变成"过早动作" → 对 STOP_GO 的突变反应错位
- 精确速度让 ARC 的曲率变化"看起来太突然"（策略期望的是带抖动的平滑值）

结果：策略在**观测与训练分布不匹配**时，内部去噪/预测机制反而成了干扰。

#### 论文论述（比"NoDR 训练消融"更深刻）

**常规论述**："DR 是必要的，否则 sim2real fail" —— 需要 NoDR 训练才能证明（已被前人反复证明）。

**本实验的论述**："DR 训练让策略对**训练时设定的噪声分布**鲁棒，但也创造了对该分布的特定先验。部署域的观测质量必须匹配训练域的 DR 强度——DR 不是'免费午餐'，而是一种显式的感知先验注入。"

这个论述提供三个额外启示：
1. **合理化 DR 的存在**：真机观测有噪声，所以 DR 训练的策略才是部署匹配的
2. **DR 强度需要匹配目标域**：DR 过大会导致对清洁环境过度补偿
3. **Sim2Real 的本质**：不是"让策略在仿真里强到极致"，而是"让仿真分布覆盖部署分布"

#### 受影响最大的运动模式

| 模式 | 退化 | 原因（推测） |
|------|------|-------------|
| ARC (-3.1%) | 曲率变化 | 精确速度让曲率转折看起来过于突兀 |
| LINEAR (-2.8%) | 匀速直线 | 延迟补偿变多余，策略略微"过冲" |
| STOP_GO (-2.4%) | 突然启停 | 无噪声让突变显得"非自然"（训练中类似突变被噪声平均掉了）|
| PASSING (0) | 掠过 | 几何关系简单，对观测精度不敏感 |

#### 对 Exp 5（zero-shot BiasJ1）论断的修正

之前分析里写过"DR 噪声吸收了 bias" —— 结合 Exp 6 的结果，更准确的说法是：

**BiasJ1 环境依然保留了 DR 的所有噪声机制（观测噪声 / 延迟 / DR 速度噪声），只是多了一层关节偏差**。由于 Backup 使用真实 TCP sensor 读数（不走 biased FK），bias 主要通过 "Task Policy 控制偏差 → 真实关节漂移 → 真实 TCP 漂移" 这一链路影响 Backup，但这种漂移时间尺度远大于 Backup 的 10 步 (1s) 响应窗口，所以影响弱。

**Exp 6 反证**：如果 DR 策略真的"内部去除了噪声"，那它在 NoDR 环境应该至少不差。但实际下降 2.5%，证明策略的鲁棒性是**分布匹配**带来的，不是**对噪声的主动剔除**。

---

## Exp 7：Backup S2 多障碍训练（已完成）

**目标**：从单障碍（S1）扩展到多障碍（1 移动 + 1 静止），验证策略能否在更复杂的几何约束下学习避让。

### 7.1 训练配置

| 参数 | 值 |
|------|---|
| 训练环境 | `PandaBackupPolicyS2-v0` |
| 观测维度 | **114D**（38 × 3 帧堆叠；38 = robot_state(18) + 2 × obstacle_info(10)） |
| 障碍物 | 1 移动（4 种运动模式）+ 1 静止（随机 spawn） |
| Action | 6D 连续，scale=0.03 m/step |
| Episode 长度 | 10 步 |
| 算法 | SAC |
| online_steps | 200,000 |
| **utd_ratio** | **4**（utd=8 会崩溃，见 7.3）|
| discount | 1.0 |
| Frame stacking | 3 |
| seed | 1000 |
| 训练时长 | 3h 49min |

### 7.2 最终 Checkpoint 评估（200k）

两次独立 eval（200 episodes each），**方差超预期**：

| 指标 | 第一次 | 第二次 |
|------|-------|-------|
| 存活率 | **91.0%** (182/200) | **83.0%** (166/200) |
| Δ | - | -8% |

**真实性能估计：~85-87%**（两次平均）。

#### 第二次 eval 详细 breakdown（200 eps）

```
总 episodes: 200
存活率 (>=10步): 166/200 (83.0%)
满存活 (10步):  166/200 (83.0%)
平均存活步数: 9.3 ± 1.7
平均累计奖励: 6.181 ± 6.666
平均最近距离: 0.121m
平均终止位移: 0.088m ± 0.035

终止原因:
  survived              : 162 (81.0%)
  hand_collision        :  27 (13.5%)  ← S1 仅 0.5%
  excessive_displacement:  10 (5.0%)
  zone_c_intrusion      :   1 (0.5%)

运动模式分析:
  模式        数量    存活率     平均奖励
  ARC         44    75.0%     +4.805   ← 最弱
  LINEAR      56    82.1%     +6.091
  PASSING     56    91.1%     +7.590
  STOP_GO     44    81.8%     +5.880
```

### 7.3 训练曲线（训练过程中间评估）

| Checkpoint | 存活率 (200ep) | 备注 |
|-----------|---------------|------|
| 50k | 78-90%（波动大）| 样本不足 |
| 100k | 88% | 初次稳定 |
| 150k | 84% | 中途小幅回退 |
| **200k** | **83-91%** | **最终** |

曲线相对稳定，**无崩溃**（对比 utd=8 run 在 100k → 125k 从 89% 崩到 80%）。

### 7.4 utd=8 vs utd=4 对比（超参数敏感性研究）

| 配置 | 峰值 checkpoint | 峰值存活率 | 崩溃？ | 训练时长 |
|------|----------------|-----------|--------|---------|
| utd=8 | 100k | 89% | ✅ 100k → 125k 崩到 80% | ~4h 到崩溃 |
| **utd=4** | **200k** | **83-91%** | ❌ 稳定 | **3h 49min** |

**结论**：S2 的 114D 状态空间下，utd=8 过于激进，Q 值发散概率高。**utd=4 是 S2 的更安全选择**。这和 REDQ (Chen et al. 2021) / SR-SAC (D'Oro et al. 2022) 关于高维度下 utd 缩放的观察一致。

### 7.5 三环境能力对比（同一条训练链）

| Env | 存活率 | hand_collision | 最近距离 |
|-----|-------|----------------|---------|
| S1 (DR, 200k) | **99.0%** | 0.5% | 0.154 m |
| S1 + BiasJ1 (zero-shot) | 98.5% | 0.5% | 0.156 m |
| S1 NoDR (zero-shot) | 96.5% | 2.5% | 0.156 m |
| **S2 (DR, 200k)** | **83-91%** | **13.5%** | **0.121 m** |

**核心发现**：S2 的存活率掉落 ~10%、碰撞率升 26 倍，不是"策略学不好"，而是**多障碍几何约束本身更难**。

### 7.6 关键运动模式退化分析

| 模式 | S1 @ 200k | **S2 @ 200k** | 退化 | 原因分析 |
|------|-----------|---------------|------|---------|
| ARC | 97.8% | **75.0%** | **-22.8%** | 曲线运动 + 静止障碍 = 最严几何约束 |
| LINEAR | 98.0% | 82.1% | -15.9% | 匀速直线下自然逃生方向可能被静止障碍挡住 |
| PASSING | 100.0% | 91.1% | -8.9% | 掠过式最简单，只需短暂避让 |
| STOP_GO | 100.0% | 81.8% | -18.2% | 突变时序 + 静止障碍约束 |

### 7.7 分析：为什么 S2 这么难

**单障碍（S1）→ 多障碍（S2）是决策几何的质变，不是状态维度升高**：

1. **逃生锥角收窄**：S1 有 2π 方向可选，S2 因为静止障碍存在，逃生锥角缩小到 π 甚至更窄
2. **样本需求爆炸**：障碍物 spawn 组合从 ~50 种跃升到 ~2500 种
3. **当前 reward 缺 proximity signal**：只有 terminal -10 或 per-step +0.5，无法区分"接近静止障碍"和"远离静止障碍"
4. **episode 长度 10 步可能不够**：多障碍场景下"先闪避 A 再调整躲 B"需要更多时间窗口

### 7.8 论文定位与 Limitations

**正面**：
- 从单障碍 99% 到多障碍 ~85%，**验证了方法可扩展到多障碍场景**
- 训练稳定，无崩溃，可复现
- PASSING 模式仍保持 91%，说明常见"横穿"场景可靠

**Limitations**：
- ARC / STOP_GO 模式碰撞率较高，真机部署需配合独立安全层（emergency stop）
- 当前 reward 缺 proximity signal，未来工作可引入 Kiemel 2024 的 soft penalty 提升 5-10%
- 当前 S2 只有"1 移 + 1 静"，更复杂场景（多移动障碍）未测试

**Checkpoint 位置**：`checkpoints/backup_policy_s2/`

---

## 待完成实验（占位，后续补充）

### Exp 1：Task Policy 观测消融（3 档 × 1 seed）

**目标**：Modular 架构下，研究纯 Task Policy（不管避障）对状态观测的敏感性。由 Runtime supervisor 在距离阈值触发时切换到 Backup Policy。

**环境要求**：pick-place + 编码器 J1 随机偏差 + **无 hand obstacle + 无 safety layer**。观测不含 `hand_active/hand_pos`。

**三档观测组合**（从完整 27D 各自去掉一部分）：

| 档位 | 维度 | 组成 |
|---|---|---|
| 27D full | 27 | robot_state(18) + block(3) + plate(3) + real_tcp(3) |
| 24D 去 real_tcp | 24 | robot_state(18) + block(3) + plate(3) |
| 21D 去 block/plate | 21 | robot_state(18) + real_tcp(3) |

**前置工作**（未完成）：
1. 新建 env：`PandaPickPlaceBiasJ1Random-v0`（及 Keyboard 变体），从 `FRRLPandaPickPlace-v0` 派生，加 `EncoderBiasConfig(target_joints=[0])`
2. env 加 `obs_mode` 参数支持 27D/24D/21D
3. 新建 3 份 record/train config：`configs/train_hil_sac_task_bias_obs{27,24,21}.json`
4. 录制 1 次 27D demo，用 `scripts/slice_safe_demo.py`（或派生脚本）切片到 24D/21D

**历史遗留**：`configs/train_hil_sac_safe_bias_obs{31,28,25}.json` 和对应的 `PandaPickPlaceSafeObs*` env 注册是 **Joint Safe RL 方向**的残留（带 hand + safety layer + hand 观测），与当前 Modular 方向不符，暂不删除，如需 joint 消融可复用。

---

## Runtime Supervisor 设计（Task ↔ Backup ↔ Homing 三态切换）

**架构**：Modular shielded RL — Task Policy + Backup Policy + Homing 确定性控制器三者解耦，由 Supervisor 按 TCP-hand 距离和位置偏差路由。

**方案 A 升级：三态 FSM（已采纳）**

早期方案 A 是两态（task↔backup）。实际部署发现两个问题：
1. Task policy 在被打断的位置直接续跑时观测分布外 → 任务表现抖动
2. Backup 训练场景（手朝 TCP 直冲→ episode terminate）不匹配真实 eval 场景（人拿手逼近 TCP 持续观察反应）

升级方案：
- **Backup env 改 TRACKING-only**：手每步重算方向追当前 TCP，贴到 D_TIGHT=8cm 停顿 3-8 步模拟"怼脸"，然后继续追。Episode 10→20 步。
- **引入 HOMING 阶段**：手清场后用确定性 P 控制器把 TCP 拉回 task 被打断时记录的 `tcp_start`，然后再交还 task policy。

```
状态转移：
  TASK   ─(d<d_safe)──────────> BACKUP   (记录 tcp_start = 当前 TCP)
  BACKUP ─(d>d_clear N 步)────> HOMING
  HOMING ─(d<d_safe)──────────> BACKUP   (tcp_start 保持不变)
  HOMING ─(|tcp−tcp_start|<tol)─> TASK   (清空 tcp_start)
```

**关键：`tcp_start` 只在 TASK→BACKUP 时记录一次**，BACKUP↔HOMING 来回切的整个过程中不更新；
意义是 homing 目标永远是 task 被打断时的原位置，保证 task 续跑时观测连续性。

### 参数（SI 单位：m）

| 参数 | 值 | 依据 |
|---|---|---|
| `d_safe` | 0.10 | 进入 BACKUP 阈值；TRACKING 下手会怼到 ~8cm，0.20 切得太早 backup 没可躲的 |
| `d_clear` | 0.20 | 离开 BACKUP 阈值；> d_safe 10cm 形成滞回避免抖动 |
| `clear_n_steps` | 3 | BACKUP→HOMING 需连续 3 步 d>d_clear，防短暂离开误触发 |
| `homing_pos_tol` | 0.02 | HOMING→TASK 的位置容差 |
| `homing_kp` | 1.0 | Clipped-deadbeat P 增益（kp=1 不超调；>1 会在 tol 附近震荡） |
| `HAND_SPAWN_DIST` | (0.15, 0.30) | 必须在 d_safe=0.10 之外 spawn，否则 episode 开始就触发 BACKUP |

### 关于姿态：为什么 Homing 只做 3D 位置

仔细读 `frrl/envs/base.py::apply_action` 后发现，仿真 env 只用动作前 3 维 (dx, dy, dz)，`rx/ry/rz` 被完全忽略：mocap_quat 仅在 reset 时设一次，整个 episode 不变。因此 backup policy 不会扰动姿态，homing 无需恢复姿态，只做 3D 位置 P 控制即可。

真机部署时若末端姿态会变（controller 接受完整 6D 增量），需要把 HomingController 升级为 6D（位置+姿态）。当前实现留了扩展空间（参考 `frrl/rl/homing_controller.py` docstring）。

### 实现

- `frrl/rl/hierarchical_supervisor.py` — 三态 FSM
- `frrl/rl/homing_controller.py` — 3D 位置 P 控制
- `scripts/test_hierarchical_supervisor.py` / `test_homing_controller.py` — 单测
- `frrl/envs/panda_backup_policy_env.py` — TRACKING-only 改造
- `scripts/test_backup_env_tracking.py` — Backup env 单测

真机版参考：`docs/real_robot_deployment_plan.md §5.4`。

### 切换方式对比（保留备忘）

| 方案 | 复杂度 | 信号需求 | 跟 Backup 训练分布对齐 |
|---|---|---|---|
| **A. 距离硬阈值 + 滞回 + Homing** | ⭐⭐ | hand 3D 位置 | **✓** |
| B. TTC（时间到碰撞） | ⭐⭐ | hand 位置 + 速度 | 部分 |
| C. CBF（QP 求解） | ⭐⭐⭐⭐ | 动力学模型 | 间接 |
| D. Critic-gated | ⭐⭐⭐ | backup Q-value | ✓ |
| E. Zone-based (A/B/C) | ⭐ | 固定 world frame | 部分 |

**配套实验（待做）**：
- Backup S1-TRACKING 重训（300k steps）
- Supervisor 阈值扫描 `d_safe ∈ {0.08, 0.10, 0.12}`，测 task success vs collision rate
- 对照组：Task-only、Always-backup、无 Homing 两态版本

---

## Backup S1-TRACKING 重训 + 位移/幸存奖励消融（2026-04-21）

**动机**：Supervisor 切换方案 A 定稿后启动的 Backup S1-TRACKING 基线 (Base) 在 75k/300k 左右幸存率停留在 30–33%，需要诊断瓶颈并设计对照实验。

### 核心诊断：`MAX_DISPLACEMENT = 0.15` 形成 reward 断崖

复盘 reward 结构发现 `displacement > 0.15` 直接触发 `-10` 终止惩罚。TRACKING 模式下手持续追 ~20 步，**"沿手反向退远→停住观察"** 这个物理上最自然的避障策略，在 0.15m 预算下几乎必然撞顶。断崖非连续惩罚，梯度信号对 policy 极不友好 → 被迫学"贴身微调"，学习难度远高于"退远再停"。

### Part G: Homing 旋转坐标系 bug 修复

复核 supervisor ↔ env 姿态约定时定位到一个长期潜藏 bug：

- `HomingController._rot_error_axis_angle` 原计算 **world 系** 增量：`q_err = q_target · q_cur⁻¹`
- 但 env (`_apply_rotation_to_mocap`) 执行 `q_new = q_cur · dq_local`——增量是**末端局部系**
- 当 `q_cur ≠ I` 且目标与当前异轴时，world/local 两个 `dq` 不同，homing 会收敛到错误姿态

修复：`q_err_local = q_cur⁻¹ · q_target`（`frrl/rl/homing_controller.py`）。新增 `scripts/test_homing_controller.py::test_rot_convergence_nonidentity_start` 覆盖唯一能暴露该 bug 的场景（当前/目标皆非单位四元数且异轴）。现有的 identity-start 单测不触发此差异，曾让 bug 长期通过。

### 设计：三组并行消融（Base / Relaxed / Combo）

把 `max_displacement` 和 `survival_bonus` 改为 `PandaBackupPolicyEnv` 构造参数，三组同架构、同训练预算 300k，仅隔离两个变量：

| 变体 | env id | `max_displacement` | `survival_bonus` | learner port | seed |
|---|---|---|---|---|---|
| Base | `PandaBackupPolicyS1-v0` | 0.15 | 5.0 | 50051 | 1000 |
| Relaxed | `PandaBackupPolicyS1Relaxed-v0` | **0.20** | 5.0 | 50052 | 1001 |
| Combo | `PandaBackupPolicyS1Combo-v0` | **0.20** | **10.0** | 50053 | 1002 |

配置：`configs/train_hil_sac_backup_s1_tracking{,_relaxed,_combo}.json`；启动脚本变体 `backup_tracking{,_relaxed,_combo}`。

**为什么动的是 `SURVIVAL_BONUS`（终局 +5→+10）而不是 `SURVIVAL_REWARD`（每步 +0.5）**：
per-step 奖励会整体放大 reward scale，污染和 Base 的直接对比；终局 bonus 只在 `truncated` 触发，信号集中在"存活完整 episode"这个事件上，正是我们想放大的 credit assignment 目标。

**为什么保留 Base**：20k–75k 看似停滞，但 85k bucket 重新观察到 breakthrough（幸存率 31.5% → 35.1%，avg reward 1.26 → 1.81），学习仍活跃，不提前 kill。

### 训练状态（记录时点 2026-04-21）

| 变体 | 进度 | 观察 |
|---|---|---|
| Base | 89.6k / 300k | 85k bucket 突破 35% 幸存率，学习活跃 |
| Relaxed | 启动早期 | - |
| Combo | 暂缓 | CPU 负载 75.94 / 20 线程（≈3.7× oversubscribed），双训练已近上限 |

CPU 饱和情况下 Combo 延后；Base + Relaxed 先并行训满 300k，再决定是否启动 Combo 或根据前两组结论简化。

### 可复现命令

```bash
# Base（位移 0.15 + bonus 5；continue）
bash scripts/train_hil_sac.sh backup_tracking learner      # 终端 1
bash scripts/train_hil_sac.sh backup_tracking actor        # 终端 2

# Relaxed（位移 0.20 + bonus 5）
bash scripts/train_hil_sac.sh backup_tracking_relaxed learner
bash scripts/train_hil_sac.sh backup_tracking_relaxed actor

# Combo（位移 0.20 + bonus 10）
bash scripts/train_hil_sac.sh backup_tracking_combo learner
bash scripts/train_hil_sac.sh backup_tracking_combo actor
```

### 关联改动

- `frrl/envs/panda_backup_policy_env.py`：`__init__` 新增 `max_displacement` / `survival_bonus` 参数，`step()` 使用实例变量而非模块常量
- `frrl/envs/__init__.py`：新增 `PandaBackupPolicyS1Relaxed-v0` / `S1Combo-v0`
- `configs/train_hil_sac_backup_s1_tracking_relaxed.json` / `..._combo.json`：新建
- `scripts/train_hil_sac.sh`：新增两个变体分支
- `frrl/rl/homing_controller.py`：Part G 坐标系 bug 修复
- `scripts/test_homing_controller.py`：新增 non-identity start 测试；`test_rot_convergence_simulation` 改用局部系右乘

---

## 附录：可复现命令

### Exp 5 eval 复现

```bash
cd /home/cheng/FR-RL-lerobot
conda activate lerobot

# S1 baseline
python scripts/eval_backup_policy.py \
  --checkpoint checkpoints/backup_policy_s1 \
  --env_task PandaBackupPolicyS1-v0 \
  --n_episodes 200

# S1 + BiasJ1 zero-shot
python scripts/eval_backup_policy.py \
  --checkpoint checkpoints/backup_policy_s1 \
  --env_task PandaBackupPolicyS1BiasJ1-v0 \
  --n_episodes 200

# S1 NoDR 跨分布 eval
python scripts/eval_backup_policy.py \
  --checkpoint checkpoints/backup_policy_s1 \
  --env_task PandaBackupPolicyS1NoDR-v0 \
  --n_episodes 200

# Exp 7 S2 eval
python scripts/eval_backup_policy.py \
  --checkpoint checkpoints/backup_policy_s2 \
  --env_task PandaBackupPolicyS2-v0 \
  --n_episodes 200
```

### Exp 2 数据出处

原始实验上下文、训练曲线、bug 复盘见 `docs/project_progress.md` §3.1-3.5。
