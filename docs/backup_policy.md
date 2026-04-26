# Backup Policy 设计方案

## 概述

Backup Policy（安全策略）在 HIL 训练中保护人类安全：当人手进入机械臂工作空间时，自动切换到 Backup Policy 进行主动闪避，人手离开后交还 Task Policy 继续任务。

策略在仿真中训练，通过 Domain Randomization 弥补 Sim2Real gap，部署到真机时使用 MediaPipe Hands + D455 深度相机提供手部位置。

参考：Kiemel et al. "Safe RL of Robot Trajectories in the Presence of Moving Obstacles", IEEE RAL 2024

---

## 1. 真机部署数据流

```
┌─ GPU Workstation ──────────────────────────────────────────┐
│                                                            │
│  D455 RGB ──→ MediaPipe Hands ──→ 关键点(u,v)             │
│  D455 Depth ─→ 查询(u,v)深度 ──→ 反投影3D(相机系)         │
│                    │                                       │
│                    ▼                                       │
│            T_cam_to_robot (标定矩阵)                       │
│                    │                                       │
│                    ▼                                       │
│          hand_pos(机器人坐标系 3D)                          │
│                    │                                       │
│                    ▼                                       │
│  ┌─────────────────────────────┐                           │
│  │ Task Policy 正常运行         │                           │
│  │   hand_dist < 阈值？         │──→ 否 → 继续 Task Policy │
│  │                             │                           │
│  │   是 ↓                      │                           │
│  │ 切换 → Backup Policy (≤10步) │                           │
│  │   hand 离开 / 10步到期 ↓     │                           │
│  │ 切换 → Task Policy 恢复      │                           │
│  └──────────────┬──────────────┘                           │
│                 │ 6D 动作                                  │
│                 ▼                                          │
│          HTTP POST /pose                                   │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌─ RT PC ────────────────────────┐
│  Flask Server → ROS → libfranka │
│  → Franka Panda 执行            │
└─────────────────────────────────┘
```

### 精度链分析

| 环节 | 误差 | 说明 |
|------|------|------|
| MediaPipe 关键点检测 | ±5-10 px | 640×480 下约 ±1-2cm |
| D455 深度测量 | ±1-2cm @1m | Intel 官方规格 |
| 相机标定 (T_cam_to_robot) | ±0.5-1cm | AprilTag 标定 |
| **累计** | **±2-4cm** | 各环节误差叠加 |

碰撞检测距离 `HAND_COLLISION_DIST = 8cm`，精度 margin = 4-6cm，可接受。

---

## 2. 场景设计

### S1: 单障碍物

- 1 个移动障碍物，4 种运动模式随机选 1 种
- 观测 28D
- 基础场景，先训练验证

### S2: 移动 + 静止

- 1 个移动障碍物（同 S1 运动模式）+ 1 个静止障碍物（随机放置）
- 观测 38D
- 进阶场景，模拟工作空间内有固定障碍

### 运动模式

每个 episode 随机选 1 种：

| 模式 | 速度 | 轨迹 | 参数 | 模拟场景 |
|------|------|------|------|---------|
| 直线冲刺 | 恒定 | 朝 TCP 直线 | dir = toward_tcp + N(0, 0.2) | 人直接伸手拿东西 |
| 弧线接近 | 恒定 | 带曲率绕向 TCP | 每步 dir 旋转 ω∈U(-0.1, 0.1) rad | 人从侧面伸手 |
| 停走式 | 0 或正常 | 朝 TCP 直线 | 每步 30% 概率停顿 | 人犹豫后再伸手 |
| 路过 | 恒定（较快） | 横穿工作空间 | dir ⊥ toward_tcp + N(0, 0.2) | 人手经过但不针对 TCP |

---

## 3. 观测空间

### S1 场景（28D）

```
robot_state (18D):
  q          (7D)  关节位置 — 关节极限/奇异性感知
  dq         (7D)  关节速度 — 惯性/动量感知
  gripper    (1D)  夹爪开合 [0,1] — 闪避时锁定，但提供状态
  tcp        (3D)  末端位置 + N(0, 0.005) — TCP 噪声（已有）

obstacle_1 (10D):
  active     (1D)  是否存在
  pos        (3D)  绝对位置（+ DR 噪声）
  vel        (3D)  帧间差分速度（+ DR 噪声）
  rel_pos    (3D)  pos - tcp 相对位置
```

### S2 场景（38D）

```
robot_state (18D):  同 S1
obstacle_1  (10D):  移动障碍物
obstacle_2  (10D):  静止障碍物
```

### 关节状态保留理由

q(7) + dq(7) 对闪避策略必要：
- **关节极限**：闪避方向可能导致某关节到极限
- **冗余构型**：7-DOF 同一 TCP 位置有多种构型，闪避空间不同
- **奇异性**：靠近奇异点时小笛卡尔位移需要大关节运动
- **Sim2Real**：q/dq 来自关节编码器，零延迟、高精度，无 gap

---

## 4. 动作空间

- **6D**: [dx, dy, dz, rx, ry, rz]
- 缩放: × ACTION_SCALE (0.03m/step)
- 夹爪锁定，不参与动作
- 控制频率: 10Hz

---

## 5. 奖励函数

参考 Kiemel et al. 2024 的核心原则：**每步奖励非负，碰撞大惩罚**。

```python
# 终止条件（r = -10.0）
if collision:                terminated, r = -10.0  # 碰到人手
if block_dropped:            terminated, r = -10.0  # 方块掉落
if zone_c:                   terminated, r = -10.0  # 进入人员禁区
if displacement > 0.15m:     terminated, r = -10.0  # 跑太远（任务无法恢复）

# 存活每步（非负）
r = 0.5                                    # 存活基础正奖励
  - 0.5  * ||tcp_current - tcp_start||     # 位移软惩罚
  - 0.01 * ||action||                      # 动作幅度惩罚
  - 0.01 * ||action - action_prev||         # 动作平滑惩罚

# 存活完整 episode bonus
if truncated: r += 5.0

# discount = 1.0（10 步短 episode，每步等价）
```

### 典型场景回报

| 场景 | 行为 | 总回报 |
|------|------|--------|
| 原地不动，手路过 | 完美 | 10×0.5 + 5.0 = **+10.0** |
| 闪避 5cm 后存活 | 正确 | ≈ **+9.7** |
| 闪避 12cm 后存活 | 可接受 | ≈ **+8.5** |
| 跑远 16cm | 过度 → 终止 | **-10.0** |
| 第 1 步碰撞 | 该闪没闪 | **-10.0** |

### 设计理由

| 优先级 | 目标 | 机制 | 说明 |
|--------|------|------|------|
| P0 | 不碰人手 / 不越界 | 终止 -10 | 安全底线 |
| P1 | 不掉方块 | 终止 -10 | 闪避不能牺牲任务状态 |
| P2 | 不跑远 | 位移 > 15cm 终止 -10 + 软惩罚 | 跑太远 task policy 无法恢复 |
| P3 | 原地微调 | 每步 +0.5 - 位移 | 不动 = 最大奖励 |
| P4 | 动作平滑 | 幅度 + 变化量惩罚 | 真机力矩/加速度有限 |

> 详细的奖励设计演进历史见 `docs/rl_reward.md`
>
> **参数化（2026-04-21 起）**：`MAX_DISPLACEMENT` 与 `SURVIVAL_BONUS` 已改为 `PandaBackupPolicyEnv` 构造参数（默认 0.15 / 5.0）。消融变体 `PandaBackupPolicyS1Relaxed-v0`（0.20 / 5.0）与 `PandaBackupPolicyS1Combo-v0`（0.20 / 10.0）用于探测"退远再停"策略的激励边界，详见 `docs/sim_exp_data.md`。
>
> **V3 proximity reward（2026-04-26 起）**：当 `use_full_arm_collision=True` 时，每步 reward 追加 saturating proximity bonus，诱导 policy 维持 ~10cm 表面间隙：
>
> ```python
> proximity = PROXIMITY_REWARD_MAX × clip(surface_clearance / PROXIMITY_SAFE_DIST, 0, 1)
> # 默认 0.20 / 0.10：clear=0 时 0, clear≥0.10 时饱和 0.20
> ```
>
> 与 V2 相比每步 max reward 从 0.5 → 0.7。设计意图："手追近时退一点（梯度 2.0/m > disp 梯度 0.5/m），退到 10cm 间隙就停（饱和后 disp penalty 接管）"。参考 `docs/rl_reward.md` V5 段落。
>
> **V2 防作弊（2026-04-21 起）**：`PandaBackupPolicyS1V2-v0` 用 wrist+hand 单球（半径 10cm，球心 = mocap weld 点，对旋转 rigid）替代 TCP+指尖三检测点，配合旋转预算（≤0.5rad）和旋转惩罚（对称 DISPLACEMENT_COEFF），堵住 V1 下"剧烈转腕让末端躲开但腕部仍碰撞"的作弊路径。开关：`use_arm_sphere_collision`/`max_rotation`/`rotation_coeff`，默认关闭向后兼容。
>
> **V2 直线退让几何（2026-04-21 晚间定稿）**：早先 V2 用 0.20m 预算 + 收紧 TCP_INIT + 工作空间 clamp，但 30 ep eval 观察策略出现"贴边绕角""捉迷藏"的歪行为——因为工作空间把最短退让方向夹弯了。定稿方案：
> - `max_displacement=0.30`：匹配 `ARM_SPAWN_DIST` 上限 30cm（手追距上限），沿 -hand_dir 直线退让足够
> - `enforce_cartesian_bounds=False`：sim 训练关闭工作空间硬 clamp，让位移 penalty 的梯度天然偏好最短路径 = 沿 -hand_dir 直线退让（真机部署层另行提供 workspace clamp）
> - `TCP_INIT` 放宽到自然操作区：X=(0.30,0.55) / Y=(-0.30,0.10) / Z=(0.15,0.40)，不再为预算让边距
>
> 原则：**工作空间是真机硬约束**（训练关、部署开），**位移预算是软惩罚设计参数**（匹配手追距）；两者正交。

---

## 6. Domain Randomization

弥补 Sim2Real gap，所有噪声仅加在**观测**上，不影响碰撞检测（碰撞用真实位置判断）。

| 参数 | 分布 | 模拟什么 |
|------|------|---------|
| 障碍物位置噪声 | N(0, 0.03) | MediaPipe + 深度 累计误差 ±3cm |
| 障碍物速度噪声 | N(0, 0.01) | 帧间差分抖动 |
| 观测延迟 | U(0, 2) 步 | 检测+推理延迟 ~0-20ms |
| 障碍物生成距离（V1 `HAND_SPAWN_DIST`）| U(0.15, 0.30) m | 不同进入距离 |
| 障碍物生成距离（V2 `ARM_SPAWN_DIST`） | U(0.21, 0.30) m | 下限 >1.5×ARM_COLLISION_DIST(0.135m) 防 1-step death |
| 障碍物生成距离（V3 `ARM_SPAWN_DIST_V3`） | U(0.30, 0.40) m | 下限 = 1.5×ARM_COLLISION_DIST_V3(0.20m)；多球安全检查 |
| 障碍物移动速度 V1/V2（`HAND_SPEED_RANGE`） | U(0.005, 0.015) m/step | 不同手速 |
| 障碍物移动速度 V3（`HAND_SPEED_RANGE_V3`） | U(0.015, 0.030) m/step | 对齐真机 15-30cm/s 区间 |
| TCP 位置噪声 | N(0, 0.005) | 已有，保留 |

### 观测延迟实现

```python
# 维护位置历史缓冲区
pos_history = deque(maxlen=5)
pos_history.append(current_pos)

# 随机延迟 0-2 步
delay = random.randint(0, min(2, len(pos_history) - 1))
obs_pos = pos_history[-(1 + delay)]
```

---

## 7. 训练配置

- 算法: SAC (Soft Actor-Critic)
- 先训 S1，确认收敛后训 S2；V2 几何 + rotation budget 软惩罚继承 S1 超参
- Episode 长度: **20 步**（2026-04-21 V2 从 10 步改 20，给 HOMING 阶段足够窗口）
- 评估: 200 episodes 存活率统计

| 参数 | S1 v0 | S1 V2（新几何 + rotation budget 软惩罚） | **S1 V3（全臂避障 + proximity reward）** | S2 |
|------|------|------------------------------------|----------------------------------------|-----|
| online_steps | 200,000 | 145,000（训练峰值 ckpt） | 300,000（默认 budget） | 200,000 |
| max_episode_steps | 10 → 20（V2 起） | 20 | 20 | 10 |
| online_buffer_capacity | 200,000 | 200,000 | 300,000 | 200,000 |
| online_step_before_learning | 1,000 | 1,000 | 100 | 1,000 |
| discount | 1.0 | 1.0 | 1.0 | 1.0 |
| temperature_init | 0.1 | 0.1 | 0.1 | 0.1 |
| critic_lr / actor_lr | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| utd_ratio | 4 | 4 | 4（V3 84D 高维 + multi-sphere 信号更复杂，utd=8 风险高，参考 S2 教训） | **4**（utd=8 会在 100k 崩，见 sim_exp_data Exp 7）|
| batch_size | 256 | 256 | 256 | 256 |
| `ROTATION_COEFF` | — | 0.2（旋转软惩罚） | 0.2 | — |
| `MAX_ROTATION` | — | π（等效关闭硬终止） | π | — |
| `ARM_SPHERE_RADIUS` | N/A（用三点 TCP+指尖） | 0.10 m（wrist+hand 单球） | 5 球 (0.07/0.07/0.065/0.065/0.10) | N/A |
| obstacle radius | 0.035 | 0.035 | **0.10**（对齐真机 hand bbox） | 0.035 |
| `ARM_SPAWN_DIST` | (0.15, 0.30) TCP | (0.21, 0.30) panda_hand | **(0.30, 0.40) panda_hand** | (0.15, 0.30) TCP |
| `HAND_SPEED_RANGE` | (0.005, 0.015) | (0.005, 0.015) | **(0.015, 0.030)**（对齐真机 15-30cm/s） | (0.005, 0.015) |
| `max_displacement` | 0.15 | 0.30 | **0.50** (V3 首训 71.5% 后 0.40→0.50, 救 excessive_displacement) | 0.15 |
| `PROXIMITY_REWARD_MAX` | — | — | **0.20**（饱和上限） | — |
| `PROXIMITY_SAFE_DIST` | — | — | **0.10**（饱和点：表面间隙 10cm） | — |

**预期 eval 成功率**：

| 版本 | sim eval 满存活率 | 真机已知问题 |
|---|---|---|
| S1 V2 (145k ckpt) | 100% | **肘/前臂可能撞**（V2 collision 只查 panda_hand 单球）|
| S1 V3（待训完）| 92-95%（worst-case 5-10% 不可解，fast hand + 近 spawn 同时发生时位移预算不够） | 全臂避障，无肘撞问题 |

V2 → V3 sim 数字略降但**真机肘撞失效模式被根除**——是质的改进。worst-case 推导见 [`sim_exp_data.md`](sim_exp_data.md) Backup S1 V3 段落 "Worst-case 可解性"。

---

## 8. Sim2Real 对齐检查表

| 环节 | 仿真 | 真机 | Gap 弥补方式 |
|------|------|------|-------------|
| 手部位置来源 | MuJoCo 精确坐标 | MediaPipe + D455 深度 | DR: 位置噪声 N(0, 0.03) |
| 手部速度来源 | 精确差分 | 帧间差分 | DR: 速度噪声 N(0, 0.01) |
| 观测延迟 | 0 | ~10ms | DR: U(0,2) 步延迟 |
| 坐标变换 | 无 | T_cam_to_robot | 标定误差含在位置噪声里 |
| 机器人状态 (q, dq) | MuJoCo | 关节编码器 | 无 gap（编码器零延迟高精度） |
| 控制器 | OSC (仿真) | 阻抗控制 (真机) | 动作缩放可能需微调 |
| 障碍物运动 | 4 种模式随机 | 真实人手 | 模式覆盖主要运动模式 |

---

## 9. 文件清单

### [sim] 仿真侧

```
frrl/envs/sim/panda_backup_policy_env.py          ✓
  - S1/S2 场景，28D/38D 观测
  - V2 几何（wrist+hand 单球 R=0.10m）+ rotation budget 软惩罚
  - V3 几何（5 球 link3/4/5/6/hand）+ obstacle r=0.10 + proximity reward
    （开关：use_arm_sphere_collision / use_full_arm_collision）
  - 4 种运动模式（LINEAR/ARC/STOP_GO/PASSING）
  - 位移 + 动作幅度 + 平滑惩罚
  - DR：位置噪声、速度噪声、观测延迟
  - 编码器偏差路径（使用 get_robot_state()）

frrl/envs/__init__.py                          ✓ 已注册
  - S1 / S2 / NoDR / BiasJ1 / V2 / V3 / Relaxed / Combo 等变体

scripts/configs/train_hil_sac_backup_s1.json      ✓（S1 v0）
scripts/configs/train_hil_sac_backup_s1_v2.json   ✓（V2 新几何 + 软惩罚）
scripts/configs/train_hil_sac_backup_s1_v3.json   ✓（V3 全臂 5 球 + obstacle r=0.10 + proximity reward）
scripts/configs/train_hil_sac_backup_s1_tracking*.json  ✓（tracking/tracking_combo/tracking_relaxed）
scripts/configs/train_hil_sac_backup_s2.json      ✓（38D S2）
scripts/sim/eval_backup_policy.py                 ✓（运动模式+奖励统计）
scripts/sim/check_backup_env.py                   ✓（S1/S2 选择+DR 开关）
scripts/real/train_hil_sac.sh                     ✓（backup/backup_s2/backup_s1_v2 分支）
```

### [real] 真机侧（已实装，2026-04）

```
frrl/robots/franka_real/vision/__init__.py        ✓

frrl/robots/franka_real/vision/hand_detector.py   ✓
  - HandDetector: WiLoR 模型（精度/鲁棒性优于原计划 MediaPipe）
  - 输入: RGB + Depth (D455) → 输出 hand_active / hand_pos_robot / confidence
  - 依赖 T_cam_to_robot (.npy)

frrl/robots/franka_real/vision/aruco_tcp.py       ✓
  - ArUco 4 角 TCP 追踪（privileged 作弊通道 / workspace 校准）

scripts/tools/calibrate_cam_to_robot.py           ✓
  - 默认 SVD 点对齐（AX=XB hand-eye 作为诊断）
  - 输出 T_cam_to_robot.npy

scripts/real/deploy_backup_policy.py              ✓
  - HierarchicalSupervisor (TASK/BACKUP/HOMING 三态) + HomingController + BiasMonitor
  - 详见 backup_policy_deployment.md
```

**待办（非实现工作，详见 `docs/project_progress.md` §7）**：
- `abs_pose_limit` 工作空间边界手动标定
- ArUco 4 角 workspace 校准脚本（`scripts/define_workspace_crop.py`）

---

## 10. 验证计划

### 仿真验证

```bash
# 1. S1 训练（两个终端）
# 终端 1:
bash scripts/real/train_hil_sac.sh backup learner
# 终端 2:
bash scripts/real/train_hil_sac.sh backup actor

# 2. S1 评估（200 步存活率 > 90%）
python scripts/sim/eval_backup_policy.py --checkpoint outputs/.../pretrained_model \
    --n_episodes 100 --env_task PandaBackupPolicyS1-v0

# 3. 无 DR 对比（测试 DR 带来的鲁棒性提升）
python scripts/sim/eval_backup_policy.py --checkpoint outputs/.../pretrained_model \
    --env_task PandaBackupPolicyS1NoDR-v0

# 4. S2 训练
# 终端 1:
bash scripts/real/train_hil_sac.sh backup_s2 learner
# 终端 2:
bash scripts/real/train_hil_sac.sh backup_s2 actor

# 5. S2 评估
python scripts/sim/eval_backup_policy.py --checkpoint outputs/.../pretrained_model \
    --n_episodes 100 --env_task PandaBackupPolicyS2-v0

# 6. 可视化环境（调试用）
python scripts/sim/check_backup_env.py --num_obstacles 1
python scripts/sim/check_backup_env.py --num_obstacles 2 --no_dr
```

### 真机验证（RT PC 就绪后）

1. 标定相机: `python scripts/tools/calibrate_cam_to_robot.py`（SVD 点对齐）
2. 验证 HandDetector 精度: 手部位置误差 < 4cm
3. 部署 checkpoint（**ckpt 版本必须与 D_SAFE/D_CLEAR 阈值匹配**）：
   - V2 ckpt: `python scripts/real/deploy_backup_policy.py --ckpt-version v2`（自动配 D_SAFE=0.30 / D_CLEAR=0.35）
   - V3 ckpt（训完后）: `python scripts/real/deploy_backup_policy.py --ckpt-version v3`（D_SAFE=0.40 / D_CLEAR=0.45）
   - 阈值版本错配会让 BACKUP 在训练分布外激活 → policy OOD
4. 验证策略切换: Task → Backup → Homing → Task 流畅过渡（详见 `docs/backup_policy_deployment.md`）
