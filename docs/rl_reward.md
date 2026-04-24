# Backup Policy 奖励函数设计演进

记录 backup policy 奖励函数从初版到当前版本的变化过程和修改原因。

---

## V1: 初始设计（碰撞惩罚 + 可选 proximity reward）

**时间**：初始版本

```python
# 碰撞终止
if collision: reward = -1.0

# 存活每步
reward = -0.01 * ||action||   # 动作惩罚

# 可选 proximity reward
if use_proximity_reward:
    r_moving = min(1.0, (d_moving / DANGER_ZONE)²) 
    r_static = mean(min(1.0, (d_static / DANGER_ZONE)²))
    reward = 0.6 * r_moving + 0.4 * r_static - 0.01 * ||action||
```

**参数**：
- 碰撞惩罚: -1.0
- 观测: 48D（3 个障碍物槽位）
- discount: 0.99

**问题**：
1. Proximity reward 鼓励"尽量远离障碍物"，与部署目标（原地闪避后交还 task policy）矛盾
2. 3 个障碍物槽位冗余，真机只有 1 只手
3. 无 Domain Randomization，sim2real gap 大

---

## V2: 重构（位移 + 平滑惩罚，去掉 proximity reward）

**时间**：2026-04-08

**修改原因**：
- 去掉 proximity reward，改为位移惩罚（鼓励原地闪避而非远离）
- 添加动作平滑惩罚（真机力矩/加速度有限）
- 观测精简：48D → 28D (S1) / 38D (S2)
- 添加 Domain Randomization

```python
# 碰撞终止
if collision: reward = -1.0

# 存活每步（全负）
reward = - 0.5  * ||tcp - tcp_start||      # 位移惩罚
         - 0.01 * ||action||               # 动作幅度
         - 0.01 * ||action - action_prev||  # 动作平滑
```

**参数**：
- 碰撞惩罚: -1.0
- 存活每步: 全负（≈ -0.03 ~ -0.08）
- discount: 0.99

**问题**：
训练 16k steps 后 reward 从 -0.88 恶化到 -1.00（接近每 episode 都碰撞）。

**根因分析**：

| 场景 | 总回报 |
|------|--------|
| 存活 10 步 | ≈ -0.5 ~ -0.8（位移积累） |
| 第 1 步碰撞 | -1.0 |

存活 10 步的总回报与碰撞差距太小。更糟的是，如果位移积累较大，存活可能比碰撞更差。
策略学到了"**快死比慢死好**"——这是完全错误的方向。

**训练日志证据**：
```
step  0- 2k: avg_reward = -0.88  (241 eps, ~8步/ep)
step  4- 6k: avg_reward = -0.77  (略有改善)
step  8-10k: avg_reward = -0.90  (开始恶化)
step 14-16k: avg_reward = -0.97  (持续恶化)
step 16-18k: avg_reward = -1.00  (每 ep 都碰撞)
```

Episode 数从 241→335/2k步，说明 episode 越来越短——策略越训越差。

---

## V3: 当前版本（每步正奖励 + 大惩罚 + 位移硬上限）

**时间**：2026-04-09

**修改原因**：
参考 Kiemel et al. 2024 "Safe RL of Robot Trajectories in the Presence of Moving Obstacles" (IEEE RAL) 的奖励设计原则。

论文的核心洞察：**"the immediate reward is never negative, discouraging an early termination"**（每步奖励非负，使提前终止永远不如存活）。

论文参数：
- 存活终止 bonus: +15
- 碰撞终止惩罚: -15
- 移动障碍物每步最大奖励: +3.0
- discount: 1.0

**V3 设计**（结合论文原则 + 我们的"原地闪避"目标）：

```python
# 终止条件（全部 -10.0）
if collision:                reward = -10.0
if block_dropped:            reward = -10.0
if zone_c_intrusion:         reward = -10.0
if displacement > 0.15m:     reward = -10.0   # 新增：位移硬上限

# 存活每步（非负）
reward = 0.5                                   # 存活基础正奖励
       - 0.5  * ||tcp - tcp_start||            # 位移软惩罚
       - 0.01 * ||action||                     # 动作幅度
       - 0.01 * ||action - action_prev||        # 动作平滑

# 存活完整 episode
if truncated: reward += 5.0                    # 存活 bonus

# discount = 1.0
```

**参数**：
| 参数 | 值 | 理由 |
|------|-----|------|
| SURVIVAL_REWARD | +0.5 | 每步存活基础正奖励 |
| TERMINATION_PENALTY | -10.0 | 碰撞/失败大惩罚 |
| SURVIVAL_BONUS | +5.0 | 存活完整 episode 额外奖励 |
| MAX_DISPLACEMENT | 0.15m | 位移硬上限（超过 = 终止） |
| DISPLACEMENT_COEFF | 0.5 | 位移软惩罚系数 |
| ACTION_NORM_COEFF | 0.01 | 动作幅度惩罚 |
| ACTION_SMOOTH_COEFF | 0.01 | 动作平滑惩罚 |
| discount | 1.0 | 短 episode（10步），每步等价 |

**各场景总回报**：

| 场景 | 行为 | 总回报 |
|------|------|--------|
| 原地不动，手路过 | 完美 | 10×0.5 + 5.0 = **+10.0** |
| 闪避 5cm 后存活 | 正确闪避 | ≈ **+9.7** |
| 闪避 12cm 后存活 | 可接受 | ≈ **+8.5** |
| 跑远 16cm | 过度反应 → 终止 | **-10.0** |
| 第 1 步碰撞 | 该闪没闪 | **-10.0** |

**排序**：不动 > 小闪避 > 大闪避 >> 碰撞 = 跑远

### V2 → V3 核心变化总结

| 改动 | V2 | V3 | 原因 |
|------|-----|-----|------|
| 每步奖励符号 | ≤0（全负） | ≥0（基础正+小惩罚） | V2 导致"快死比慢死好" |
| 碰撞惩罚 | -1.0 | -10.0 | 拉大存活与碰撞的差距 |
| 存活 bonus | 无 | +5.0 | 奖励完整存活 |
| 位移硬上限 | 无 | 0.15m → 终止 -10 | 软惩罚不够，跑远=任务失败 |
| discount | 0.99 | 1.0 | 短 episode 每步等价 |

---

## V4: 旋转预算软惩罚（防"扭腕作弊"）

**时间**：2026-04-21

**修改原因**：V2 新几何（`ARM_SPHERE_RADIUS=0.10`，wrist+hand 单球）下，策略学会用**剧烈转腕**让末端避开移动障碍物、同时腕部仍与手接触但因为 Minkowski 球半径不够大而没被判定碰撞。原本的旋转**硬预算终止**（`max_rotation=0.5 rad`）过于激进，导致 eval 方差大，训练不稳定。

**V4 改法**：把旋转硬上限改软惩罚，和 `DISPLACEMENT_COEFF` 对称。

```python
# V3 继承（不变）
if collision:                reward = -10.0
if displacement > MAX_DISPLACEMENT:  reward = -10.0

reward = 0.5                                      # 存活基础
       - DISPLACEMENT_COEFF * ||tcp - tcp_start|| # 0.5，位移软惩罚
       - ROTATION_COEFF     * ||axis_angle(quat_rel)|| # V4 新增
       - ACTION_NORM_COEFF  * ||action||          # 0.01
       - ACTION_SMOOTH_COEFF * ||action - action_prev|| # 0.01

# V4 旋转预算关键参数
ROTATION_COEFF = 0.2       # 从早期的 0.5 降下来，保留"能不转就不转"偏置
MAX_ROTATION   = π         # 约 180°，等效关闭硬终止
```

**为什么降到 0.2**：与 `DISPLACEMENT_COEFF=0.5` 作用半径不同——位移量级 0.1m、旋转量级 0.5-1 rad。系数 0.2 让两项惩罚在典型轨迹上的贡献大致相当（≈ 0.05/step）。

**为什么 `MAX_ROTATION=π`**：硬终止导致 eval 不稳定（策略在硬上限边缘摆动），等效关闭后让软惩罚的梯度平滑引导策略"能不转就不转"。

**实验结果（Backup S1 V2 newgeom）**：
- 35k ckpt（真机首选）：训练早期就稳定，旋转幅度明显下降
- 145k ckpt（训练峰值）：真机备用
- 训练曲线对比：硬终止版本 50k 左右会出现 eval 崩溃，软惩罚版本无崩溃

完整实验数据见 `docs/sim_exp_data.md` 的 Backup S1 V2 段落。

---

## 设计原则总结

1. **每步奖励必须非负**：保证存活永远优于碰撞终止
2. **碰撞惩罚远大于存活总回报**：`|终止惩罚|` >> `存活总正回报`
3. **位移需要硬约束**：软惩罚无法阻止极端行为，硬上限 + 终止才能真正约束（但**旋转**这种方向模糊的量改用软惩罚，见 V4）
4. **discount = 1.0**：短 episode 中每步安全等价重要
5. **不使用 proximity reward**：我们的目标是"原地微调"不是"远离障碍物"
6. **硬 vs 软惩罚选择原则**：约束量越"方向明确、越界 = 任务失败"越适合硬终止（位移）；越"连续、量级不固定"越适合软惩罚（旋转）
