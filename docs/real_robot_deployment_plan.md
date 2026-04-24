# FR-RL-lerobot Franka Panda 真机部署方案

> **⚠️ 本文档是 2026-03 编写的实施规划**。大部分阶段（RT PC 基础栈、FrankaRealEnv、
> 真机编码器偏差 B+D 注入、相机/teleop 基础）已经完成并落地。
> 当前实施状态的权威来源是 [`project_progress.md §2.5`](project_progress.md)。
>
> **Task policy 训练 pipeline（pretrain + online HIL）见独立文档
> [`task_policy_training.md`](task_policy_training.md)**（2026-04-23 阶段 1–5 完结）。
>
> 本文保留作为**设计决策与原始规划**的历史记录，里面的"从 ~/hil-serl/ 移植 ..."
> 等描述反映的是**计划**，对应代码现在都已进入 `frrl/robot_servers/` /
> `frrl/envs/franka_real_*.py` / `frrl/cameras/` / `frrl/teleoperators/` 等正式位置。
> 真机启动/验证流程请看 [`rt_pc_runbook.md`](rt_pc_runbook.md) 和
> [`fault_injection_realhw.md`](fault_injection_realhw.md)。

## Context

FR-RL-lerobot 是基于 lerobot hil-serl 的 PyTorch 版 RL 框架，两个核心创新：
1. **编码器偏差注入** — 故障容错学习
2. **Backup Policy + 安全层** — 人手避让（仿真中用状态机模拟人手，真机用 MediaPipe Hands + D455 深度检测）

目前仅有 MuJoCo 仿真，需部署到真实 Franka Panda（固件4.2.1, libfranka 0.9.1）。参考项目 `~/hil-serl` 有完整的 ROS1+Flask HTTP 真机控制栈。

**硬件拓扑（两台机器）：**
```
[笔记本 RT PC]                          [GPU Workstation]
  Intel x86_64 CPU                        Ubuntu 20.04
  Ubuntu 20.04 + PREEMPT_RT              RTX 3090 Ti (24GB)
  libfranka 0.9.1                         Actor + Learner (localhost gRPC)
  ROS Noetic                              FrankaRealEnv (HTTP → 笔记本:5000)
  serl_franka_controllers                 策略推理 + SAC训练 (GPU)
  franka_server.py (:5000)                MediaPipe 手部检测 (CPU)
  franka_gripper_server.py                RealSense相机 (USB)
  以太网直连 Franka                        SpaceMouse (USB)
  Franka原装夹爪(Franka Hand)
  无GPU，无NVIDIA驱动冲突
```

### 双机分离的原因

PREEMPT_RT 内核与 NVIDIA 闭源驱动有兼容性问题（编译失败、驱动不稳定）。
分离后各司其职：
- **笔记本 RT PC**：纯 CPU，装 RT 内核无顾虑，专注 1kHz 实时控制
- **GPU 工作站**：标准内核，NVIDIA 驱动正常工作，专注训练和推理

### 机器职责分工

| 机器 | 职责 | 关键要求 |
|------|------|----------|
| **笔记本 (RT PC)** | Franka 1kHz 实时控制 + Flask Server | x86_64, Ubuntu 20.04, PREEMPT_RT, 无需GPU |
| **GPU Workstation** | Actor + Learner + 相机 + 遥操作 + 手部检测 | Ubuntu 20.04, RTX 3090 Ti, 标准内核 |

### 通信链路

```
GPU Workstation (3090 Ti)                      笔记本 RT PC
┌─────────────────────────┐    HTTP :5000    ┌─────────────────┐
│ Actor (FrankaRealEnv)   │ ──────────────→ │ Flask Server     │
│   ↕ localhost gRPC      │ ← JSON state ── │ (franka_server)  │
│ Learner (SAC训练)       │                 │                  │
│                         │                 │   以太网 1kHz     │
│ RealSense相机 (USB)     │                 │   ↕              │
│ SpaceMouse (USB)        │                 │ Franka Panda     │
│ MediaPipe手部检测 (CPU)  │                 │                  │
│ 策略推理 (GPU)          │                 └─────────────────┘
└─────────────────────────┘
```

Actor 和 Learner 同机运行，使用线程模式（`concurrency.actor = "threads"`），共享内存，gRPC 走 localhost。

### GPU 资源分配（RTX 3090 Ti, 24GB）

| 占用项 | 估算显存 | 说明 |
|--------|---------|------|
| SAC训练 (Learner) | ~4-6 GB | ResNet视觉编码器 + Critic反向传播 |
| 策略推理 (Actor) | ~1-2 GB | 与Learner共享编码器权重 |
| MediaPipe 手部检测 | ~0 GB (CPU) | 仅阶段五启用 |
| 相机图像/Replay Buffer | <1 GB | CPU内存为主 |
| **合计** | **~8-10 GB** | 24GB 充足 |

### 网络配置

| 连接 | 协议 | 说明 |
|------|------|------|
| 笔记本 ↔ Franka | 以太网直连, 静态IP | `172.16.0.1` (笔记本) ↔ `172.16.0.2` (Franka), 1kHz实时 |
| GPU机器 → 笔记本 | HTTP :5000, 局域网 | Actor 发送动作/读取状态 (~10Hz, JSON) |
| GPU机器内部 | gRPC localhost | Actor ↔ Learner 通信 |

笔记本需要**两个网口**：一个以太网直连 Franka，一个连局域网（可用 USB 转以太网适配器）。

### 笔记本 RT PC 配置要求

**必须做的系统优化（确保1kHz控制环稳定）：**

1. **安装 Ubuntu 20.04**（双系统或替换现有 22.04）

2. **安装 PREEMPT_RT 内核**
   ```bash
   # 下载对应版本的RT补丁，编译安装
   # Ubuntu 20.04 推荐内核 5.15-rt
   ```

3. **BIOS设置**
   - 关闭超线程 (Hyper-Threading)
   - 关闭CPU C-States节能
   - 关闭Intel SpeedStep / Turbo Boost

4. **系统优化**
   ```bash
   # 锁定CPU频率到最高
   sudo cpufreq-set -g performance
   # 关闭Wi-Fi（中断干扰）
   sudo nmcli radio wifi off
   # 最小化运行（关闭桌面环境）
   sudo systemctl set-default multi-user.target
   # 禁用不必要的服务
   sudo systemctl disable bluetooth cups avahi-daemon
   ```

5. **网络配置**
   ```bash
   # 以太网口配置静态IP（连Franka）
   # IP: 172.16.0.1, 子网: 255.255.255.0
   # Franka控制柜默认IP: 172.16.0.2
   ```

### GPU Workstation（RTX 3090 Ti）

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 3090 Ti (24GB VRAM) |
| 系统 | Ubuntu 20.04（标准内核，无需 PREEMPT_RT） |
| 软件 | PyTorch, CUDA, FR-RL-lerobot 代码库 |
| USB | USB 3.0 ×2 (RealSense相机) + USB 2.0 (SpaceMouse) |
| 网口 | 千兆以太网 (局域网连笔记本 RT PC) |
| 位置 | **物理上放在机器人旁边**（USB线需够到相机和SpaceMouse） |

### 设备清单

| 设备 | 连接到 | 接口 |
|------|--------|------|
| Franka Panda (控制柜) | 笔记本 RT PC | 以太网直连 |
| Franka Hand (原装夹爪) | Franka Panda | 内置 |
| RealSense 前置相机 | GPU Workstation | USB 3.0 |
| RealSense 腕部相机 | GPU Workstation | USB 3.0 |
| SpaceMouse | GPU Workstation | USB HID |
| 显示器 (调试用) | GPU Workstation | HDMI/DP |

### 备用设备

| 设备 | 说明 |
|------|------|
| Jetson AGX Orin 32GB | 暂不使用，可作为备用视觉推理节点或独立实验平台 |

---

## 阶段零：RT PC 基础设施

在 RT PC (Ubuntu 20.04) 上搭建：
1. ROS Noetic + libfranka 0.9.1 (源码编译，匹配固件4.2.1) + franka_ros
2. serl_franka_controllers（提供 cartesian_impedance_controller）
3. `franka_server.py` + Franka Gripper server (从 `~/hil-serl/serl_robot_infra/robot_servers/` 复制，使用 `franka_gripper_server.py` 而非 Robotiq 版本)

**验证**：`curl -X POST http://127.0.0.1:5000/getstate` 返回完整状态 JSON

---

## 阶段一：FrankaRealEnv 核心环境

**目标**：创建真机 Gym 环境，观测和动作格式与仿真 FrankaGymEnv 对齐。

### 1.1 动作空间统一为 7D

仿真和真机统一使用 **7D 动作空间** `[dx, dy, dz, rx, ry, rz, gripper]`：
- 仿真端：去掉/绕过 `EEActionWrapper`（原来把 4D 扩展为 7D），policy 直接输出 7D
- 真机端：7D action → `action_scale` 缩放 → 笛卡尔位姿增量 → HTTP POST /pose
- SpaceMouse 天然输出 6D + gripper，与 7D 动作空间对齐

**需要同步修改仿真端**：
- `frrl/envs/wrappers/factory.py` 中的 wrapper 栈调整
- `frrl/policies/sac/configuration_sac.py` 中 `output_features.action.shape` 改为 `(7,)`
- 训练配置中 action shape 统一为 7

### 1.2 观测空间对齐

**agent_pos (18D)**：

| 字段 | 仿真来源 | 真机来源 (HTTP getstate) |
|------|----------|------------------------|
| qpos(7) | `data.qpos[panda_dofs]` | `ps["q"]` |
| qvel(7) | `data.qvel[panda_dofs]` | `ps["dq"]` |
| gripper(1) | `data.ctrl[gripper_id] / 255` | `ps["gripper_pos"]` |
| tcp_pos(3) | `site_xpos[pinch_site]` | `ps["pose"][:3]` |

**gripper 统一为 [0, 1]**：
- 仿真端改一行：`gripper = ctrl[gripper_id] / 255.0`
- 真机端已经是 [0, 1]

**environment_state (6D)**：

真机也需要提供 `environment_state = [block_pos(3), plate_pos(3)]`，因为有 encoder bias 时纯图像可能不够。

| 数据 | 真机获取方式 |
|------|------------|
| plate_pos(3) | 固定位置，写死在配置里（板不动） |
| block_pos(3) | AprilTag + 相机检测，或 YOLO + 深度相机 |

**pixels**：`{"front": (128,128,3), "wrist": (128,128,3)}` — RealSense 采集

**完整观测格式**（仿真和真机一致）：
```python
{
    "agent_pos": np.ndarray(18),                   # robot state
    "environment_state": np.ndarray(6),            # block_pos + plate_pos
    "pixels": {"front": (128,128,3), "wrist": (128,128,3)},  # 图像
}
```

### 1.3 新增文件

**`frrl/envs/franka_real_env.py`** (核心，~350行)
- 移植 hil-serl `FrankaEnv` 的 HTTP 通信逻辑
- `_update_currpos()` → `requests.post(url+"getstate")`
- `get_robot_state()` → 构建 18D agent_pos（gripper 归一化到 [0,1]）
- `get_environment_state()` → 6D [block_pos, plate_pos]（plate 写死，block 视觉检测）
- `step()` → 7D 动作缩放 + `clip_safety_box()` + HTTP POST /pose
- `reset()` → precision模式 → interpolate_move → compliance模式
- `get_im()` → RealSense 图像采集
- 关键参考：`~/hil-serl/serl_robot_infra/franka_env/envs/franka_env.py`

**`frrl/envs/franka_real_config.py`** (~100行)
- `FrankaRealConfig` dataclass：server_url, cameras, action_scale, rotation_scale, reset_pose, compliance_param, precision_param, safety bounds (xyz + rpy), max_episode_length, hz, plate_position, encoder_bias_config
- 参考：hil-serl 的 `DefaultEnvConfig` + 各实验的 `EnvConfig`

**`frrl/cameras/realsense.py`** (~100行)
- 移植 `~/hil-serl/serl_robot_infra/franka_env/camera/rs_capture.py` 和 `video_capture.py`

**`frrl/cameras/block_detector.py`** (~80行)
- 物块检测：YOLO 或颜色检测 → 像素坐标 + D455 深度 → 相机坐标 → 机器人坐标
- 依赖 `T_cam_to_robot` 变换矩阵（从配置文件加载）

**`scripts/calibrate_camera.py`** (~100行)
- 前置相机标定脚本：AprilTag + Franka TCP 位置 → 计算 `T_cam_to_robot`

### 1.4 修改文件

**`frrl/rl/env_factory.py`** — `make_robot_env()` 添加 Franka 分支：
```python
if cfg.robot is None:
    # 仿真 (现有)
elif hasattr(cfg, 'franka_config') and cfg.franka_config is not None:
    # Franka 真机 (新增)
    env = FrankaRealEnv(config=cfg.franka_config)
else:
    # SO100 真机 (现有)
```

**`frrl/rl/env_factory.py`** — `make_processors()` 添加 Franka processor pipeline：
- env_processor: 与仿真相同（VanillaObs → AddBatch → Device）
- action_processor: 简化版（Teleop → Intervention → Torch2Numpy），无需IK

**`frrl/envs/configs.py`** — `HILSerlRobotEnvConfig` 添加 `franka_config` 字段

**`frrl/envs/base.py`** — gripper 观测归一化：
```python
def get_gripper_pose(self):
    return np.array([self._data.ctrl[self._gripper_ctrl_id] / 255.0], dtype=np.float32)
```

### 1.5 观测方案与消融实验设计

**核心实验问题**：编码器偏差下，policy 是否需要显式的物体位置信息（environment_state）才能保持鲁棒性？还是纯图像观测就足够？

**实验矩阵（4组对比，论文结论之一）**：

| 实验组 | 观测 | Bias | 预期 |
|--------|------|------|------|
| A1 | images + state（无 env_state） | 无 | 基线，应该能学会 |
| A2 | images + state（无 env_state） | 有 | 纯图像在 bias 下的鲁棒性？ |
| B1 | images + state + env_state | 无 | 基线，有额外信息应该更好 |
| B2 | images + state + env_state | 有 | 显式位置信息能否帮助 bias 下的恢复？ |

**论文价值**：
- A2 vs B2：有 bias 时，显式物体位置信息是否显著提升任务成功率？
- A2 vs A1：bias 对纯图像策略的影响有多大？
- 如果 A2 ≈ B2：说明 bias 的影响主要在执行层，感知层的额外信息帮助有限
- 如果 A2 << B2：说明 bias 干扰了图像与状态的融合，显式位置信息提供了必要的锚点

**相机标定是必须的**（阶段五的 MediaPipe 手部检测强制需要 `T_cam_to_robot`），所以 `environment_state` 直接加入——同一个标定、同一个 D455，多检测一个物块即可。

**消融实验设计（2组对比，论文结论之一）**：

| 实验组 | 观测 | Bias | 目的 |
|--------|------|------|------|
| A | images + state + env_state | 无 bias | 基线 |
| B | images + state + env_state | 有 bias | bias 下的鲁棒性 |

A vs B：量化 encoder bias 对真机 pick-and-place 任务成功率的影响。

**观测格式（统一）**：
```python
{
    "agent_pos": np.ndarray(18),                    # robot state
    "environment_state": np.ndarray(6),             # block_pos + plate_pos
    "pixels": {"front": (128,128,3), "wrist": (128,128,3)},
}
```
- `observation.state` shape = 24（18 + 6 拼接）
- 需要前置相机标定（一次性 AprilTag 法，阶段五手部检测复用）
- 需要物块检测（YOLO/颜色 + D455 深度）

### 1.5.1 相机硬件与标定方案

| 相机 | 安装方式 | 用途 | 标定需求 |
|------|---------|------|---------|
| 前置 D455 | 固定在桌面/支架上 | 物块检测 (block_pos) + policy 图像输入 (front) | 一次标定 `T_cam_to_robot`，永久有效 |
| 手腕 D455 | 安装在 Franka 手腕 | policy 图像输入 (wrist) | hand-eye 标定 `T_cam_to_ee` |

**environment_state (6D) 获取方式**：

| 数据 | 来源 | 方法 |
|------|------|------|
| plate_pos(3) | 固定位置 | 写死在 `FrankaRealConfig.plate_position` 中 |
| block_pos(3) | 前置 D455 | CV检测 (YOLO/颜色) → 像素(u,v) + 深度d → 相机坐标 → `T_cam_to_robot` 变换 → 机器人坐标 |

**前置相机标定流程**（一次性，AprilTag 法）：

```bash
# 1. 打印 AprilTag 标定板，放在工作台面上
# 2. 控制 Franka TCP 移动到标定板上 4+ 个已知点
#    记录每个点的 TCP 位置（从 getstate 获取）和对应的像素坐标+深度
# 3. 用 OpenCV solvePnP 计算 T_cam_to_robot
# 4. 将变换矩阵写入配置文件

# 标定脚本输出：
T_cam_to_robot = np.array([
    [r11, r12, r13, tx],
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [0,   0,   0,   1 ],
])
```

**物块位置检测流程**（每步运行）：

```python
def get_block_pos(self) -> np.ndarray:
    rgb, depth = self.front_camera.read()  # D455 RGB + 深度
    # 1. CV 检测物块像素坐标
    u, v = detect_block(rgb)  # YOLO 或颜色检测
    # 2. 像素 + 深度 → 相机坐标
    z = depth[v, u]  # 深度值 (米)
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    p_cam = np.array([x_cam, y_cam, z, 1.0])
    # 3. 相机坐标 → 机器人坐标
    p_robot = self.T_cam_to_robot @ p_cam
    return p_robot[:3]
```

D455 内参 (fx, fy, cx, cy) 出厂自带，通过 `pyrealsense2` 直接读取，不需要额外标定。

**精度评估**：
- D455 深度精度在 0.5-1m 工作距离下约 ±1cm
- AprilTag 标定后的坐标变换精度约 ±5mm
- 物块检测综合精度约 ±1-2cm，对 RL policy 足够

### 1.6 注意事项

1. **图像和状态的时间同步**：机器人状态通过 HTTP 获取（笔记本），图像通过 USB 获取（本机 HPC）。两者有微小时间差（~几ms），在 10Hz 控制下可忽略。
2. **阶段一联调时**：可以先硬编码 block 固定位置跳过检测，先验证环境基础功能，后续再接入 CV 检测。
3. **手腕相机标定**：hand-eye 标定可以后续再做，阶段一只用手腕图像作为 policy 输入（不用于坐标计算）。

### 验证

```
□ env.reset() → 机器人移动到 reset 位姿
□ env.step(np.zeros(7)) → 返回合法 18D obs + 6D env_state + 图像
□ env.step([0.5,0,0,0,0,0,0]) → 机器人沿 x 方向移动
□ gripper 观测值在 [0, 1] 范围内
□ 对比仿真和真机的 agent_pos 值范围和量级一致
□ 图像正常显示（OpenCV imshow 检查）
```

---

## 阶段二：遥操作集成

遥操作设备确定使用 **SpaceMouse**。

### SpaceMouse 特性

- 输出 6D 笛卡尔速度 `[dx, dy, dz, rx, ry, rz]` + 2 个按钮（夹爪开/关）
- 与 7D 动作空间天然对齐：6D + gripper = 7D
- hil-serl 有完整代码可直接移植
- **论文语义**：SpaceMouse 干预经过 biased 阻抗控制器，模拟"操作员在有故障的机器人上工作"

### 新增文件

```
frrl/teleoperators/spacemouse/
  configuration_spacemouse.py       # SpaceMouseConfig(TeleoperatorConfig)
  teleop_spacemouse.py              # SpaceMouseTeleoperator(Teleoperator)
  pyspacemouse.py                   # 从 ~/hil-serl/.../spacemouse/ 移植
```

### 修改文件

**`frrl/teleoperators/__init__.py`** — 注册 SpaceMouse teleoperator

### 验证

```
□ SpaceMouse USB 连接成功，get_action() 返回合法 7D 数据
□ Actor + SpaceMouse + FrankaRealEnv，人工操作机器人运动正常
□ 干预标志 is_intervention 正确传播到 transition
□ 遥操作完成一次完整的 pick-and-place
```

---

## 阶段三：编码器偏差注入（真机）

**论文核心**：bias 必须同时影响控制器（Jacobian）和观测，与仿真行为一致。

> 详细的架构分析和数据流对比见 `docs/fault_injection_architecture.md`

### 仿真中的完整因果链（目标）

```
bias → q_measured = q_true + bias
  ├→ 控制器用 biased q 算 Jacobian → 力矩不准（执行偏差）
  ├→ FK(biased q) → 错误的 TCP 位置（感知偏差）
  └→ Reset: q_true = home - bias（初始位姿偏移）
```

### 注入方案：双注入点（B+D）

hil-serl 架构中，控制器和观测走两条独立的数据路径，需要在两个点分别注入：

```
libfranka 硬件
    ↓ q_true
franka_hw
    │
    ├──→ FrankaStateHandle (C++ 内存)
    │        ↓
    │    C++ 阻抗控制器 update()
    │        q_biased = q_true + bias  ←← 【注入点B: C++控制器】
    │        J = Jacobian(q_biased)     ← 错误的 Jacobian
    │        tau = J^T * F              ← 偏差力矩 → 执行偏差 ✓
    │
    ├──→ franka_state_controller
    │        ↓
    │    /franka_state_controller/franka_states (q_true)
    │        ↓
    │    franka_server.py
    │        self.q = q_true + bias  ←← 【注入点D: Flask server】
    │        ↓
    │    HTTP getstate → 返回 biased q → 感知偏差 ✓
    │
    └──→ /joint_states → 不管（控制器不读）
```

**两个注入点读同一个 ROS param `/encoder_bias`**，保证 bias 值一致。

### 修改文件

**RT PC 侧**：

**`serl_franka_controllers/.../cartesian_impedance_controller.cpp`** — 注入点B：
```cpp
void CartesianImpedanceController::update(...) {
    franka::RobotState robot_state = state_handle_->getRobotState();

    // 读取 bias（从 ROS param，每次 update 读取）
    std::vector<double> bias(7, 0.0);
    if (nh_.getParam("/encoder_bias", bias)) {
        for (int i = 0; i < 7; i++) {
            robot_state.q[i] += bias[i];
        }
    }

    // 后续所有计算（Jacobian、阻抗控制）基于 biased q
    // ...
}
```

**`robot_servers/franka_server.py`** — 注入点D + HTTP API：
```python
# 初始化
rospy.set_param("/encoder_bias", [0.0] * 7)

# 新增路由
@webapp.route("/set_encoder_bias", methods=["POST"])
def set_encoder_bias():
    bias = request.json["bias"]  # 7D list
    rospy.set_param("/encoder_bias", bias)
    return "Bias set"

# 修改 _set_currpos：应用 bias 到观测
def _set_currpos(self, msg):
    bias = rospy.get_param("/encoder_bias", [0.0] * 7)
    self.q = np.array(list(msg.q)) + np.array(bias)
    # pose (TCP) 也应该用 biased FK 重算（可选）
    # ... 其余不变 ...
```

**HPC 侧（FR-RL-lerobot）**：

**`frrl/envs/franka_real_env.py`** — `reset()` 中设置 bias：
```python
def reset(self, **kwargs):
    # ... 物理重置 ...
    if self.bias_injector:
        self.bias_injector.on_episode_start(num_joints=7)
        bias = self.bias_injector.current_bias
        if bias is not None:
            requests.post(self.url + "set_encoder_bias", json={"bias": bias.tolist()})
            logging.info(f"真机 Episode: bias = {np.round(bias, 4)}")
        else:
            requests.post(self.url + "set_encoder_bias", json={"bias": [0.0]*7})
    obs = self._get_obs()
    return obs, {}
```

**`frrl/envs/franka_real_env.py`** — `get_robot_state()` 不需要额外加 bias（getstate 返回的已是 biased 值）

### Bias 安全范围

| 关节 | 范围 (rad) | 约 (度) |
|------|-----------|---------|
| J1-J4 | [-0.15, 0.15] | ±8.6° |
| J5-J7 | [-0.10, 0.10] | ±5.7° |

**渐进式验证**：无 bias → 0.02 rad 固定 → 0.05 → 0.10 → 目标范围 → 随机 bias

### 验证

```
□ set_encoder_bias API 正常工作（curl 测试）
□ C++ 控制器能读到 /encoder_bias param
□ 设置固定 bias 后，getstate 返回的 q 包含偏移
□ 设置 bias 后机器人行为可观察到变化（执行偏差的物理效果）
□ 对比同一位姿命令在有/无 bias 时的到达位姿差异
□ 逐步增大 bias，监控力矩和稳定性
□ 随机 bias 在不同 episode 正确切换
□ bias=0 时行为与未注入时完全一致
```

---

## 阶段四：端到端训练

### 4.1 配置文件变更（相对仿真）

基于 `configs/train_hil_sac_base.json`，真机配置需修改以下字段：

| 字段 | 仿真值 | 真机值 | 原因 |
|------|--------|--------|------|
| `env.robot` | `null` | FrankaRealConfig | 切换到真机环境 |
| `env.teleop` | `null` | SpaceMouse/GELLO配置 | 遥操作设备 |
| `output_features.action.shape` | `[3]` | `[7]` | 7D统一动作空间 |
| `dataset_stats.action.min/max` | 3D | 7D `[-1,1]×7` | 动作维度对齐 |
| `dataset_stats.observation.state` | 仿真值域 | 真机值域（需重新采集） | 关节角/速度范围不同 |
| `num_discrete_actions` | `3` | 待定（连续or离散夹爪） | 7D动作空间下夹爪处理 |
| `policy.actor_learner_config.learner_host` | `"127.0.0.1"` | `"127.0.0.1"` | 同机，不变 |

### 4.2 新增文件

```
configs/real_robot/
  train_franka_no_bias.json           # 无 bias 基线
  train_franka_with_bias.json         # 有 bias
scripts/
  train_franka_real.sh                # 真机训练启动脚本
```

两组配置仅 `encoder_bias_config` 不同，其余完全一致（observation.state shape=24）。

### 4.3 训练前数据采集

正式 RL 训练前，用遥操作收集演示数据：

```bash
# 步骤1：录制 demo（30个episode）
python -m frrl.rl.core.env_factory \
    --config_path configs/real_robot/record_demo.json \
    --mode record

# 步骤2：从 demo 数据计算 dataset_stats
python scripts/tools/compute_dataset_stats.py \
    --repo_id frrl/franka_pick_place_demo

# 步骤3：用 dataset_stats 更新训练配置
```

数据用于：
- 计算 `observation.state` 的 min/max（归一化参数）
- 训练 reward classifier（成功/失败图像分类）
- 填充 offline buffer（SAC 可用 offline 数据启动）

### 4.4 Reward 机制

**方案一：位姿阈值（初期，需要 environment_state 可靠）**
```python
def compute_reward(self):
    block_pos = self.environment_state[:3]
    plate_pos = self.environment_state[3:]
    dist = np.linalg.norm(block_pos[:2] - plate_pos[:2])
    return float(dist < 0.08 and block_pos[2] < 0.05)
```
依赖阶段一的物块位置检测。板位置固定写配置。

**方案二：RewardClassifier（更鲁棒）**
- 用 demo 中的成功/失败帧训练图像分类器
- 已有基础设施：`frrl/policies/sac/reward_model/`
- 在 processor pipeline 中用 `RewardClassifierProcessorStep` 在线推理
- 不依赖物块位置检测精度

**建议**：先用方案一快速验证，再切换到方案二提高鲁棒性。

### 4.5 训练流程

```bash
# HPC 4090 上，两个终端

# 终端1：Learner
python -m frrl.rl.core.learner \
    --config_path configs/real_robot/train_franka_pick_place.json

# 终端2：Actor（确保 RT PC 的 franka_server.py 已启动）
python -m frrl.rl.core.actor \
    --config_path configs/real_robot/train_franka_pick_place.json
```

Actor 和 Learner 在同一台 HPC 上，gRPC 走 localhost。
Actor 通过 HTTP 访问笔记本 RT PC 的 Flask Server。

### 验证

```
□ Learner 启动，等待 Actor 连接
□ Actor 启动，gRPC 连接 Learner 成功
□ Actor 能控制真机执行动作（HTTP → Flask → Franka）
□ 遥操作干预正常（SpaceMouse/GELLO 信号到 Actor）
□ Transition 正确发送到 Learner（观测、动作、奖励）
□ Learner SAC loss 下降
□ WandB 日志：episodic reward、intervention rate、policy frequency
□ 检查点正常保存和加载
□ 无 bias 基线能训练出可用策略（验证整个 pipeline）
```

---

## 阶段五：Backup Policy Sim-to-Real + 手部检测

**目标**：将仿真训练的 backup policy（主动避让策略）部署到真机，用 MediaPipe Hands + D455 深度相机替代状态机获取人手位置。

> 完整的 backup policy 设计方案见 `docs/backup_policy.md`

### 5.1 Backup Policy 回顾

Backup policy 不是冻结/停止，是一个**主动避让策略**：
- 仿真中在 `PandaBackupPolicyEnv` 训练（10步短 episode，SAC 算法）
- 检测到人手接近 → 切换到 backup policy → 小幅闪避 → 人手离开 → 切回 task policy
- 6D 动作空间（xyz + rpy，夹爪锁定），action_scale = 0.03m/step

**场景变体**：
- S1: 1 个移动障碍物，28D 观测 = robot_state(18) + obstacle(10)
- S2: 1 个移动 + 1 个静止障碍物，38D 观测 = robot_state(18) + 2×obstacle(10)

**运动模式**（每 episode 随机选 1 种）：
- LINEAR — 匀速直线朝 TCP
- ARC — 带曲率弧线接近 TCP
- STOP_GO — 朝 TCP 但随机停顿
- PASSING — 横穿工作空间，不朝 TCP

**奖励设计**：
- 碰撞/掉块/越界 → -1.0（终止）
- 存活每步 → -0.5×位移 - 0.01×动作幅度 - 0.01×动作变化量（鼓励原地小幅平滑闪避）

```
真机运行时的策略切换逻辑：
  Task Policy (pick-and-place) 正常运行
       ↓ HandDetector 检测到人手距 TCP < 阈值
  切换 → Backup Policy（主动避让，最多10步）
       ↓ 人手离开 / 10步到期
  切回 → Task Policy 继续任务
```

### 5.2 Sim-to-Real 部署方案

Backup policy 在仿真中训练（含 Domain Randomization），直接迁移到真机，不在真机上重新训练。

**Sim-to-Real Gap 分析**：

| 因素 | 仿真 | 真机 | Gap 及应对 |
|------|------|------|-----------|
| 障碍物位置 | 精确已知 | MediaPipe + D455（±2-4cm） | DR: 位置噪声 N(0, 0.03) |
| 障碍物速度 | 精确差分 | 帧间差分估算 | DR: 速度噪声 N(0, 0.01) |
| 障碍物形状 | 球体（8cm碰撞距离） | 真实人手（非球体） | 保守设置碰撞距离 |
| 观测延迟 | 0ms | MediaPipe ~5-10ms | DR: U(0, 2) 步延迟 |
| 动力学 | MuJoCo OSC | Franka 阻抗控制 | 动作缩放可能需微调 |
| 控制频率 | 10Hz | 10Hz | 一致 |
| 障碍物运动 | 4 种模式随机 | 真实人手 | 模式覆盖主要运动模式 |
| 机器人状态 (q, dq) | MuJoCo | 关节编码器 | 无 gap（零延迟高精度） |

**已实现的 Domain Randomization**（仅观测层，不影响碰撞检测）：
```python
# PandaBackupPolicyEnv._compute_observation() 中：
obstacle_pos += np.random.normal(0, 0.03, 3)   # 位置噪声 σ=3cm（MediaPipe+D455 精度）
obstacle_vel += np.random.normal(0, 0.01, 3)   # 速度噪声 σ=1cm/s
tcp_pos += np.random.normal(0, 0.005, 3)       # TCP 噪声 σ=5mm
# 观测延迟：使用 0-2 步前的历史位置
```

### 5.3 手部检测方案

使用 **MediaPipe Hands** + **D455 深度相机**，不使用 WiLoR（过重，backup policy 只需要手掌 3D 位置）。

**精度链**：

| 环节 | 误差 | 说明 |
|------|------|------|
| MediaPipe 关键点检测 | ±5-10 px | 640×480 下约 ±1-2cm |
| D455 深度测量 | ±1-2cm @1m | Intel 官方规格 |
| 相机标定 (T_cam_to_robot) | ±0.5-1cm | AprilTag 标定 |
| **累计** | **±2-4cm** | 各环节误差叠加 |

碰撞距离 8cm，DR 噪声 σ=3cm，策略在 4-12cm 范围内做决策，margin 可接受。

**检测数据流**：

```
D455 RGB → MediaPipe Hands → 手掌关键点 (u,v)
D455 Depth → 查询 (u,v) 深度 → 反投影 3D (相机系)
→ T_cam_to_robot → 3D 位置 (机器人系)
→ 填入 backup policy 障碍物观测槽
```

**HandDetector 输出 → Backup Policy 观测映射**：

```python
# HandDetector 检测
hand_pos, hand_active, confidence = detector.detect(rgb, depth)

# 构建 obstacle_info (10D)
hand_vel = (hand_pos - prev_hand_pos) / dt
obstacle_info = np.concatenate([
    [1.0],                          # active
    hand_pos,                       # pos (3D, 机器人坐标系)
    hand_vel,                       # vel (3D, 帧间差分)
    hand_pos - tcp_pos,             # relative_pos (3D)
])

# 构建观测（与仿真格式完全一致）
# S1: 28D = robot_state(18) + obstacle(10)
obs = np.concatenate([robot_state_18d, obstacle_info])
```

**HandDetector 特性**：
- MediaPipe Hands 在 **CPU 上运行**，无需 GPU 资源
- 推理延迟 ~5-10ms（远低于 10Hz 控制周期 100ms）
- 前置 D455 的 RGB + 深度用于检测
- **需要相机标定**（复用阶段一的 `T_cam_to_robot`）

### 5.4 真机运行架构

**切换方案**：方案 A — 距离硬阈值 + 滞回（采纳于 `docs/sim_exp_data.md` §Runtime Supervisor）。

**参数**（依据 Backup 训练分布 `HAND_SPAWN_DIST = (0.12, 0.25) m`）：

| 参数 | 值 | 依据 |
|---|---|---|
| `D_TRIGGER` | 0.20 m | Backup 训练分布中位数，切换瞬间 backup 一定 in-distribution |
| `D_RELEASE` | 0.30 m | > 训练上限 25cm，确认安全后再交回 task |
| 滞回区 | 0.10 m | 避免阈值附近 mode 抖动（每步翻转） |

```
FrankaRealEnv.step() 中的策略切换（滞回状态机）：

每步循环：
  1. HandDetector 检测人手 → hand_pos, hand_active
  2. d = min_dist(TCP, hand_pos)  (hand_active=False 时 d = +∞)
  3. 判断 mode（带滞回）：
     if mode_last == task:
         if d < D_TRIGGER (0.20m):  mode = backup
         else:                      mode = task
     else:  # mode_last == backup
         if d > D_RELEASE (0.30m):  mode = task
         else:                      mode = backup
  4. 构建观测（task: 27D/24D/21D, backup: 28D/38D）
  5. action = active_policy.select_action(obs)
  6. env.step(action)
```

```python
# FrankaRealSafeEnv 伪代码
class FrankaRealSafeEnv(FrankaRealEnv):
    D_TRIGGER = 0.20  # 进入 backup 工作域
    D_RELEASE = 0.30  # 离开 backup 工作域

    def __init__(self, ..., backup_policy_path):
        super().__init__(...)
        self.backup_policy = load_policy(backup_policy_path)
        self.hand_detector = HandDetector(T_cam_to_robot=self.T_cam_to_robot)
        self._mode = "task"   # 初始 task 模式

    def step(self, task_action):
        rgb, depth = self.front_camera.read()
        hand_pos, hand_active, _ = self.hand_detector.detect(rgb, depth)
        tcp_pos = self.currpos[:3]

        # 滞回切换：进入用 D_TRIGGER，退出用 D_RELEASE
        d = np.linalg.norm(hand_pos - tcp_pos) if hand_active else np.inf
        if self._mode == "task" and d < self.D_TRIGGER:
            self._mode = "backup"
        elif self._mode == "backup" and d > self.D_RELEASE:
            self._mode = "task"

        if self._mode == "backup":
            backup_obs = self._build_backup_obs(hand_pos, tcp_pos)
            action = self.backup_policy.select_action(backup_obs)
            full_action = np.concatenate([action, [0.0]])  # 6D + 夹爪锁定
        else:
            full_action = task_action

        return super().step(full_action)
```

**对应仿真侧模块**：`frrl/rl/supervisor/hierarchical.py`（待建），真机直接复用该 supervisor。

### 5.5 新增文件

| 文件 | 说明 |
|------|------|
| `frrl/vision/__init__.py` | 视觉模块入口 |
| `frrl/vision/hand_detector.py` | MediaPipe Hands + D455 → 3D 手部位置（CPU 推理） |
| `frrl/vision/camera_calibration.py` | AprilTag 标定工具 → T_cam_to_robot |
| `frrl/envs/franka_real_safe_env.py` | FrankaRealEnv + HandDetector + backup policy 切换逻辑 |
| `scripts/calibrate_camera.py` | 标定流程脚本 |
| `configs/real_robot/deploy_franka_safe.json` | 真机安全部署配置 |

### 5.6 已完成的仿真侧改动

| 文件 | 修改 |
|------|------|
| `frrl/envs/panda_backup_policy_env.py` | 重构：S1/S2 场景 + 4 种运动模式 + DR + 新奖励函数 |
| `frrl/envs/__init__.py` | S1/S2/NoDR/BiasJ1 共 6 个 gym 注册 |
| `configs/train_hil_sac_backup_s1.json` | S1 训练配置（28D 输入） |
| `configs/train_hil_sac_backup_s2.json` | S2 训练配置（38D 输入） |
| `scripts/sim/eval_backup_policy.py` | 评估脚本：运动模式统计 + 奖励统计 |
| `scripts/sim/check_backup_env.py` | 可视化调试脚本 |

### 5.7 部署步骤

1. **仿真训练 backup policy**：先 S1 后 S2，确认收敛（存活率 >90%）← **当前进行中**
2. **相机标定**：`python scripts/calibrate_camera.py`（RT PC 就绪后）
3. **HandDetector 独立验证**：D455 → MediaPipe → 3D 手部位置，精度 <4cm，延迟 <15ms
4. **真机部署 backup policy（sim2real）**：加载仿真 checkpoint，HandDetector 提供手部位置
5. **联合测试**：task policy + backup policy 切换，人手进出工作区

### 验证

```
□ MediaPipe 手部检测延迟 < 15ms，位置精度 < 4cm
□ 手伸入工作区 → 策略切换到 backup → 机械臂主动避让（非冻结）
□ 手离开 → 切回 task policy → 继续任务
□ 避让过程中不碰撞人手（碰撞距离 < 8cm 不发生）
□ 避让过程中不掉落已抓取的物块
□ 连续多次手部进出，策略切换稳定
```

---

## 阶段六：安全保护（贯穿所有阶段）

1. **笛卡尔空间边界** — `clip_safety_box()` (来自 hil-serl)
2. **通信超时** — HTTP 请求超时 → 停止
3. **ESC 紧急停止** — 键盘监听 (来自 hil-serl)
4. **Bias 幅度限制** — `EncoderBiasConfig.bias_range` 约束
5. **渐进部署** — 无 bias → 小固定 bias → 目标范围 → 随机 bias
6. **人手安全** — MediaPipe 手部检测 + Backup Policy 主动避让（阶段五）

---

## 文件清单

### 新增
| 文件 | 阶段 |
|------|------|
| `frrl/envs/franka_real_env.py` | 一 |
| `frrl/envs/franka_real_config.py` | 一 |
| `frrl/cameras/realsense.py` | 一 |
| `frrl/teleoperators/spacemouse/` (4个文件) | 二 |
| `configs/real_robot/` (3个JSON) | 四、五 |
| `frrl/vision/hand_detector.py` | 五 |
| `frrl/vision/camera_calibration.py` | 五 |
| `frrl/envs/franka_real_safe_env.py` | 五 |
| `scripts/calibrate_camera.py` | 五 |

### 修改
| 文件 | 阶段 |
|------|------|
| `frrl/rl/env_factory.py` | 一、五 |
| `frrl/envs/configs.py` | 一 |
| `frrl/teleoperators/__init__.py` | 二 |
| `frrl/envs/franka_real_env.py` | 三、五 |
| `robot_servers/franka_server.py` (RT PC) | 三 |

### RT PC 侧（你自行处理）
| 项目 | 阶段 |
|------|------|
| ROS Noetic + libfranka 0.9.1 + franka_ros | 零 |
| serl_franka_controllers 编译 | 零 |
| franka_server.py 部署 | 零 |
| bias 注入方案（joint_states 或 C++ 层） | 三 |

## 依赖关系

```
阶段零 (RT PC基础设施)
  └─→ 阶段一 (FrankaRealEnv) + 阶段二 (遥操作) [可并行]
        └─→ 阶段三 (编码器偏差注入，完整方案)
              └─→ 阶段四 (端到端训练 Pick-and-Place)
                    └─→ 阶段五 (手部检测 + Backup Policy)
阶段六 (安全) ← 贯穿所有阶段
```
