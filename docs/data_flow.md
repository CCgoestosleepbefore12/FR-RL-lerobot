# FR-RL 完整数据流

## 概览

```
Actor端（环境交互）                          Learner端（策略更新）
┌─────────────────┐                      ┌─────────────────┐
│ 环境 → 观测处理  │  ── transition ──→   │ Buffer → 采样    │
│ → 策略推理      │  ←── 权重推送 ───    │ → SAC更新        │
│ → 动作处理      │                      │ → 推送权重       │
│ → OSC控制       │                      │                  │
└─────────────────┘                      └─────────────────┘
```

---

## Actor 端

### Step 1: 环境输出原始观测

`env.step(action)` 返回：

```python
obs = {
    "agent_pos": np.array(24D, float32),
    "pixels": {
        "front": np.array(128, 128, 3, uint8),
        "wrist": np.array(128, 128, 3, uint8),
    }
}
```

agent_pos 24D 组成：

| 维度 | 内容 | 来源 | 准确性 |
|------|------|------|--------|
| 0-6 | 关节角度 qpos | 编码器读数 (q_true + bias) | ❌ biased |
| 7-13 | 关节速度 qvel | 编码器导数 | ✓ 正确（固定bias导数=0）|
| 14 | 夹爪位置 gripper | 控制指令值 | ✓ 正确 |
| 15-17 | 末端位置 tcp_pos | biased FK(q_true + bias) | ❌ biased |
| 18-20 | 方块位置 block_pos | 外部传感器 | ✓ 正确 |
| 21-23 | 真实末端 noisy_real_tcp | 外部定位 + 5mm高斯噪声 | ✓ 大致正确 |

### Step 2: Env Processor（4步pipeline）

```
原始obs
  │
  ▼ Numpy2TorchActionProcessorStep
  │   numpy array → torch tensor
  │
  ▼ VanillaObservationProcessorStep
  │   重命名key：
  │     "agent_pos"    → "observation.state"       (24D)
  │     "pixels.front" → "observation.images.front" (128×128×3)
  │     "pixels.wrist" → "observation.images.wrist" (128×128×3)
  │
  ▼ AddBatchDimensionProcessorStep
  │   添加batch维度：
  │     (24,) → (1, 24)
  │     (128,128,3) → (1, 3, 128, 128)  # 同时HWC→CHW
  │
  ▼ DeviceProcessorStep
      cpu → cuda
```

### Step 3: 策略推理（SACPolicy.select_action）

```
┌──────────────────────────────────────────────────────────────┐
│                    SACPolicy 内部                             │
│                                                               │
│  ┌─────────────── State Encoder ──────────────┐              │
│  │                                             │              │
│  │  "observation.state" (1, 24)                │              │
│  │      │                                      │              │
│  │      ▼ MIN_MAX归一化                        │              │
│  │      │  公式: (x - min)/(max - min)*2 - 1   │              │
│  │      │  → 每个维度映射到 [-1, 1]            │              │
│  │      │                                      │              │
│  │      ▼ Linear(24, 256)                      │              │
│  │      │  全连接层：24维 → 256维               │              │
│  │      │                                      │              │
│  │      ▼ LayerNorm(256)                       │              │
│  │      │  层归一化：稳定训练                   │              │
│  │      │                                      │              │
│  │      ▼ Tanh()                               │              │
│  │      │  激活函数：输出压缩到 [-1, 1]         │              │
│  │      │                                      │              │
│  │      → 256D 特征                            │              │
│  └─────────────────────────────────────────────┘              │
│                                                    ╲          │
│  ┌─────────── Image Encoder (front) ──────────┐    ╲         │
│  │                                             │     ╲        │
│  │  "observation.images.front" (1, 3, 128, 128)│      ╲       │
│  │      │                                      │       ╲      │
│  │      ▼ MEAN_STD归一化                       │        ╲     │
│  │      │  ImageNet标准化：                     │         ╲    │
│  │      │  (x - [0.485,0.456,0.406])           │          ╲   │
│  │      │    / [0.229,0.224,0.225]             │           ╲  │
│  │      │                                      │            ╲ │
│  │      ▼ ResNet10 (冻结，5M参数)              │      concat  │
│  │      │  卷积神经网络，提取图像特征           │      → 768D │
│  │      │  输出: 特征图 (1, C, H, W)           │            ╱ │
│  │      │                                      │           ╱  │
│  │      ▼ SpatialLearnedEmbeddings             │          ╱   │
│  │      │  学习空间位置编码                     │         ╱    │
│  │      │                                      │        ╱     │
│  │      ▼ Flatten + Linear(C*8, 256)           │       ╱      │
│  │      │  Dropout(0.1) → 全连接 → LayerNorm   │      ╱       │
│  │      │  → Tanh                              │     ╱        │
│  │      │                                      │    ╱         │
│  │      → 256D 特征                            │   ╱          │
│  └─────────────────────────────────────────────┘  ╱           │
│                                                  ╱            │
│  ┌─────────── Image Encoder (wrist) ──────────┐ ╱             │
│  │  同上处理流程                               │╱              │
│  │  → 256D 特征                               │               │
│  └────────────────────────────────────────────┘               │
│                                                               │
│  768D                                                         │
│    │                                                          │
│    ├──→ Actor MLP                                             │
│    │      Linear(768, 256) → ReLU                             │
│    │      Linear(256, 256) → ReLU                             │
│    │      → mean_layer: Linear(256, 3) → 3D均值               │
│    │      → std_layer:  Linear(256, 3) → 3D标准差             │
│    │      → TanhSquash采样 → 3D连续动作 [dx, dy, dz]          │
│    │                                                          │
│    └──→ Discrete Critic MLP                                   │
│           Linear(768, 256) → ReLU                             │
│           Linear(256, 256) → ReLU                             │
│           Linear(256, 3) → 3个Q值 [Q_关, Q_保持, Q_开]        │
│           → argmax → 夹爪动作 (0=关/1=保持/2=开)              │
│                                                               │
│  最终输出: action = [dx, dy, dz, gripper] = 4D tensor         │
└──────────────────────────────────────────────────────────────┘
```

### Step 4: Action Processor

```
action (1, 4) torch tensor
  │
  ▼ Torch2NumpyActionProcessorStep
      → numpy array (4,)
```

### Step 5: EEActionWrapper 扩展动作

```
输入: [dx, dy, dz, gripper]  范围: [-1,-1,-1,0] ~ [1,1,1,2]

处理:
  xyz = [dx, dy, dz] * ee_step_size(0.025)    # 缩放到米
  rotation = [0, 0, 0]                         # 不控制旋转
  gripper_cmd = gripper - 1.0                  # [0,2]→[-1,1]

输出: [dx*0.025, dy*0.025, dz*0.025, 0, 0, 0, gripper-1] = 7D
      → 送入 FrankaGymEnv.apply_action()
```

### Step 6: OSC 控制（FrankaGymEnv.apply_action 内部）

```
for 每个 substep (共50个):

  ┌─ 如有编码器偏差 ─────────────────────────────────┐
  │  保存真实 qpos                                    │
  │  qpos ← q_true + bias    # 临时替换为编码器读数   │
  │  mj_forward()             # 更新Jacobian和FK       │
  │  → Jacobian 方向偏了                               │
  │  → FK 位置偏了                                     │
  └──────────────────────────────────────────────────┘

  tau = opspace(                # OSC计算力矩
    target_pos = mocap_pos,     # 目标位置
    target_ori = mocap_quat,    # 目标姿态
    current_state = data,       # 当前（biased）状态
    Kp = 200, damping = 1.0,    # PD增益
    nullspace = home_position,  # 零空间目标
    gravity_comp = True,        # 重力补偿
  )

  ┌─ 如有编码器偏差 ─────────────────────────────────┐
  │  恢复真实 qpos                                    │
  │  mj_forward()                                     │
  └──────────────────────────────────────────────────┘

  data.ctrl = tau               # 施加力矩
  mj_step()                     # 物理仿真（用真实qpos）
```

### Step 7: 收集 Transition 发给 Learner

```python
transition = {
    "state":      当前observation (处理后的tensor),
    "action":     实际执行的action (4D),
    "reward":     float (0.0 或 1.0),
    "next_state": 下一步observation,
    "done":       bool,
    "info": {
        "is_intervention": bool,    # 人类是否干预
        "teleop_action":   array,   # 实际执行的动作
    }
}
→ gRPC 序列化 → 发送给 Learner
```

---

## Learner 端

### Step 8: 接收 Transition → 存入 Buffer

```
收到 transition
  │
  ├─ 如果 is_intervention=True 且 reward=0:
  │     丢弃（失败的干预数据质量差）
  │
  ├─ 否则: → 放入 Online Buffer
  │
  └─ 如果 is_intervention=True 且 reward>0:
        → 额外放入 Offline Buffer（高质量干预数据）
```

Buffer 状态：

```
Offline Buffer (容量100K):
  初始: 50个demo的~3000条transition
  训练中: + 人类成功干预的transition
  采样: 每次128条

Online Buffer (容量100K):
  初始: 空
  训练中: 所有非丢弃的transition
  采样: 每次128条
```

### Step 9: 混合采样

```
batch = concat(
    128条 from Online Buffer,    # 策略自己探索的
    128条 from Offline Buffer,   # demo + 人类干预
) = 256条

每条包含: (state, action, reward, next_state, done)
```

### Step 10: SAC 更新（每步4个loss）

```
batch中的数据:
  observations      (256, 24D+图像)   当前状态
  actions           (256, 4D)          执行的动作
  rewards           (256, 1)           奖励
  next_observations (256, 24D+图像)   下一个状态
  done              (256, 1)           是否结束

特征提取（共享encoder，只算一次）:
  observation_features      = encoder(observations)       → 768D
  next_observation_features = encoder(next_observations)  → 768D

┌─────────────────────────────────────────────────────────┐
│ a. Critic Loss (更新 Q-function)                         │
│                                                          │
│    目标: Q(s,a) ≈ r + γ * (1-done) * Q_target(s',a')   │
│                                                          │
│    Q_target(s',a'):                                      │
│      a' ~ Actor(s')           # 用actor采样下一步动作    │
│      Q1, Q2 = Critic(s',a')  # 两个Q网络                │
│      Q_target = min(Q1, Q2) - α*log_prob(a')            │
│                  ↑ 取小值防过估计   ↑ 熵正则化            │
│                                                          │
│    loss = MSE(Q(s,a), r + γ*(1-done)*Q_target)          │
│    → 反向传播 → clip_grad_norm(40) → 更新critic参数      │
├─────────────────────────────────────────────────────────┤
│ b. Discrete Critic Loss (更新夹爪Q-function)             │
│                                                          │
│    Q_disc(s) → [Q_关, Q_保持, Q_开]  3个Q值              │
│    用TD-target更新，逻辑同上                              │
│    → 更新 discrete_critic 参数                           │
├─────────────────────────────────────────────────────────┤
│ c. Actor Loss (更新策略网络)                              │
│                                                          │
│    a_new ~ Actor(s)                                      │
│    loss = α*log_prob(a_new) - Q(s, a_new)               │
│            ↑ 鼓励探索          ↑ 最大化Q值               │
│    → 更新 actor 参数                                     │
├─────────────────────────────────────────────────────────┤
│ d. Temperature Loss (自动调节探索程度)                    │
│                                                          │
│    α (temperature) 控制探索vs利用的平衡                   │
│    loss = -α * (log_prob(a) + target_entropy)            │
│    → 更新 log_alpha 参数                                 │
├─────────────────────────────────────────────────────────┤
│ e. Target Network 软更新                                  │
│                                                          │
│    Q_target ← 0.995 * Q_target + 0.005 * Q              │
│    （缓慢跟踪主网络，稳定训练）                           │
└─────────────────────────────────────────────────────────┘
```

### Step 11: 推送权重给 Actor

```
每4个优化步:
  state_dicts = {
    "policy": actor.state_dict(),                # Actor网络权重
    "discrete_critic": discrete_critic.state_dict(),  # 夹爪Critic权重
  }
  → 序列化 → gRPC → Actor

Actor 收到后:
  policy.actor.load_state_dict(...)
  policy.discrete_critic.load_state_dict(...)
  → 用新权重继续推理
```

---

## Warmup 阶段（在主循环之前）

```
Learner启动 → 加载offline demo到buffer

Warmup (500步):
  for i in range(500):
    batch = 从 offline buffer 采样 512条
    → Critic更新
    → Discrete Critic更新
    → Actor更新（每policy_update_freq步）
    → Temperature更新
    → Target网络软更新

推送预训练权重给Actor → Actor从有能力的策略开始交互
```

---

## 编码器偏差的影响路径

```
编码器偏差 bias (每episode随机 [0, 0.25] rad)
  │
  ├─→ Reset时: q_true = home - bias
  │     → 真实末端位置偏移（MuJoCo窗口可见）
  │
  ├─→ 观测: qpos_measured = q_true + bias
  │         tcp_pos = FK(q_true + bias)
  │     → 策略看到的状态是biased的
  │     → 但 noisy_real_tcp 是真实的 → 策略可以对比发现偏差
  │
  └─→ 控制: OSC用biased Jacobian计算力矩
        → 运动方向有偏差
        → 但物理仿真用真实qpos，末端实际运动正确（误差小）
```
