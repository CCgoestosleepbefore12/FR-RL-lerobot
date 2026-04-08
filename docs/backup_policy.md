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

---

## 6. Domain Randomization

弥补 Sim2Real gap，所有噪声仅加在**观测**上，不影响碰撞检测（碰撞用真实位置判断）。

| 参数 | 分布 | 模拟什么 |
|------|------|---------|
| 障碍物位置噪声 | N(0, 0.03) | MediaPipe + 深度 累计误差 ±3cm |
| 障碍物速度噪声 | N(0, 0.01) | 帧间差分抖动 |
| 观测延迟 | U(0, 2) 步 | 检测+推理延迟 ~0-20ms |
| 障碍物生成距离 | U(0.12, 0.25)m | 不同进入距离 |
| 障碍物移动速度 | U(0.008, 0.025)m/step | 不同手速 |
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
- 先训 S1，确认收敛后训 S2
- Episode 长度: 10 步
- 评估: 200 步连续生存率

| 参数 | 值 |
|------|-----|
| online_steps | 50,000 |
| online_buffer_capacity | 200,000 |
| online_step_before_learning | 1,000 |
| discount | 1.0 |
| temperature_init | 0.1 |
| critic_lr | 3e-4 |
| actor_lr | 3e-4 |
| utd_ratio | 4 |
| batch_size | 256 |

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

### 已完成（仿真侧）

```
frrl/envs/panda_backup_policy_env.py          ✓ 已重构
  - S1/S2 场景，28D/38D 观测
  - 4 种运动模式（LINEAR/ARC/STOP_GO/PASSING）
  - 位移 + 动作幅度 + 平滑惩罚
  - DR：位置噪声、速度噪声、观测延迟
  - 编码器偏差路径修复（使用 get_robot_state()）

frrl/envs/__init__.py                          ✓ 已更新
  - S1/S2/NoDR/BiasJ1 共 6 个 gym 注册

configs/train_hil_sac_backup_s1.json           ✓ 已创建（28D 输入）
configs/train_hil_sac_backup_s2.json           ✓ 已创建（38D 输入）
scripts/eval_backup_policy.py                  ✓ 已更新（运动模式+奖励统计）
scripts/check_backup_env.py                    ✓ 已更新（S1/S2 选择+DR 开关）
scripts/train_hil_sac.sh                       ✓ 已更新（backup/backup_s2 分支）
```

### 待实现（真机侧，需要硬件）

```
frrl/vision/__init__.py
  - 视觉模块入口

frrl/vision/hand_detector.py
  - HandDetector 类
  - 输入: RGB + Depth (D455)
  - 内部: MediaPipe Hands (CPU 推理, ~5-10ms)
  - 输出: hand_active(bool), hand_pos_robot(3D), hand_confidence(float)
  - 依赖: T_cam_to_robot (标定矩阵, .npy 文件)

frrl/vision/camera_calibration.py
  - CameraCalibrator 类
  - AprilTag 标定: 多组 (末端位姿, 像素检测) → T_cam_to_robot (4×4)
  - 保存/加载 .npy

scripts/calibrate_camera.py
  - 标定流程脚本
  - 步骤: 采集多位姿 → 求解 AX=XB → 保存 T_cam_to_robot
```

---

## 10. 验证计划

### 仿真验证

```bash
# 1. S1 训练（两个终端）
# 终端 1:
bash scripts/train_hil_sac.sh backup learner
# 终端 2:
bash scripts/train_hil_sac.sh backup actor

# 2. S1 评估（200 步存活率 > 90%）
python scripts/eval_backup_policy.py --checkpoint outputs/.../pretrained_model \
    --n_episodes 100 --env_task PandaBackupPolicyS1-v0

# 3. 无 DR 对比（测试 DR 带来的鲁棒性提升）
python scripts/eval_backup_policy.py --checkpoint outputs/.../pretrained_model \
    --env_task PandaBackupPolicyS1NoDR-v0

# 4. S2 训练
# 终端 1:
bash scripts/train_hil_sac.sh backup_s2 learner
# 终端 2:
bash scripts/train_hil_sac.sh backup_s2 actor

# 5. S2 评估
python scripts/eval_backup_policy.py --checkpoint outputs/.../pretrained_model \
    --n_episodes 100 --env_task PandaBackupPolicyS2-v0

# 6. 可视化环境（调试用）
python scripts/check_backup_env.py --num_obstacles 1
python scripts/check_backup_env.py --num_obstacles 2 --no_dr
```

### 真机验证（RT PC 就绪后）

1. 标定相机: `python scripts/calibrate_camera.py`
2. 验证 HandDetector 精度: 手部位置误差 < 4cm
3. 部署 S1 checkpoint，人手进入 → 验证主动闪避
4. 验证策略切换: Task → Backup → Task 流畅过渡
