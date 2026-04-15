# ROS 标准架构 vs hil-serl 架构：数据流与故障注入分析

## 1. 两种架构的本质区别

### 1.1 标准 ROS 机器人控制架构

标准 ROS 架构下，控制器通过 **ROS topic** 获取关节状态反馈，形成闭环：

```
┌─────────────────────────────────────────────────────────────┐
│                      标准 ROS 控制架构                       │
│                                                             │
│  用户/MoveIt                                                │
│      ↓ 目标轨迹                                             │
│  /joint_trajectory_controller (ROS controller)              │
│      ↓                        ↑                             │
│      ↓ 关节力矩/位置命令       ↑ 读 /joint_states 反馈      │
│      ↓                        ↑                             │
│  hardware_interface ──→ 电机驱动 ──→ 编码器                   │
│                                        ↓                    │
│                              joint_state_controller         │
│                                        ↓                    │
│                                  /joint_states              │
│                                   ↓        ↓                │
│                              控制器反馈   MoveIt/rviz        │
│                              (闭环PD)    (可视化/规划)       │
└─────────────────────────────────────────────────────────────┘
```

**关键特征**：
- 控制器 **订阅 `/joint_states` topic** 做闭环反馈
- 状态通过 **ROS topic 通信**（有序列化/反序列化开销）
- 控制频率受限于 ROS 通信延迟（通常 100-500Hz）
- 所有组件通过 topic 松耦合

### 1.2 hil-serl Franka 控制架构

hil-serl 架构下，C++ 控制器通过 **franka_hw 硬件接口层的内存直读** 获取状态，绕过 ROS topic：

```
┌─────────────────────────────────────────────────────────────┐
│                    hil-serl 控制架构                         │
│                                                             │
│  RL Policy (Actor, 10Hz)                                    │
│      ↓ HTTP POST /pose                                      │
│  franka_server.py (Flask)                                   │
│      ↓ rospy.publish(PoseStamped)                           │
│  /equilibrium_pose topic                                    │
│      ↓                                                      │
│  C++ 阻抗控制器 update() (1kHz)                              │
│      │                                                      │
│      │ q = state_handle_->getRobotState().q                 │
│      │ ↑ C++ 内存直读！不经过任何 ROS topic                   │
│      │                                                      │
│      │ J = Jacobian(q)                                      │
│      │ tau = J^T * (K*(x_d - x) + D*(dx_d - dx))           │
│      │ joint_handles[i].setCommand(tau)                     │
│      ↓                                                      │
│  franka_hw (EffortJointInterface) → libfranka → Franka      │
│                                                             │
│  并行路径（仅状态报告，不参与控制闭环）：                       │
│  franka_hw → franka_state_controller                        │
│                  ↓ 发布 FrankaState msg                      │
│           /franka_state_controller/franka_states             │
│                  ↓                                           │
│           franka_server.py 订阅 → HTTP getstate → Actor      │
│                                                             │
│  franka_hw → joint_state_controller                         │
│                  ↓ 发布 JointState msg                       │
│           /joint_states → rviz（无人读，旁路）                │
└─────────────────────────────────────────────────────────────┘
```

**关键特征**：
- 控制器通过 **`FrankaStateHandle` C++ 内存直读** 获取状态（零延迟）
- 控制频率 **1kHz**（libfranka 硬实时回调）
- `/joint_states` 和 `/franka_states` 是**旁路输出**，不参与控制闭环
- 策略命令通过 ROS topic 传入（`/equilibrium_pose`），但状态反馈不走 topic

## 2. 为什么 `/joint_states` 注入在 hil-serl 下无效

### 2.1 标准 ROS：`/joint_states` 在控制闭环内

```
编码器 → joint_state_controller → /joint_states
                                      ↓
                               控制器订阅读取 ← 闭环反馈路径
                                      ↓
                               PD计算 → 力矩命令
```

在 `/joint_states` 注入 bias：
- ✅ 控制器收到 biased q → 计算错误力矩 → **执行偏差**
- ✅ 上层应用读到 biased q → **感知偏差**

### 2.2 hil-serl：`/joint_states` 在控制闭环外

```
libfranka → franka_hw ──→ FrankaStateHandle ──→ 控制器直读
                  │                               (不经过任何topic)
                  │
                  ├──→ franka_state_controller → /franka_states → Flask → Actor
                  │                               (观测路径)
                  │
                  └──→ joint_state_controller → /joint_states → rviz
                                                  (旁路，无人读)
```

在 `/joint_states` 注入 bias：
- ❌ 控制器不读 → 无执行偏差
- ❌ Flask 不读 → 无感知偏差
- ✅ rviz 显示偏了 → 但这没有任何实际意义

## 3. hil-serl 下的等效故障注入方案

### 3.1 目标：复现仿真中的 bias 因果链

```
仿真 (MuJoCo):
  bias → q_measured = q_true + bias
    ├→ OSC 用 biased q 算 Jacobian → 力矩偏差（执行偏差）
    ├→ FK(biased q) → TCP 报告错误（感知偏差）
    └→ 物理仿真用 q_true（真实物理不变）

真机目标：同样的因果链
  bias → q_measured = q_true + bias
    ├→ 阻抗控制器用 biased q 算 Jacobian → 力矩偏差（执行偏差）
    ├→ getstate 返回 biased q 和 biased TCP（感知偏差）
    └→ Franka 硬件用 q_true 执行力矩（真实物理不变）
```

### 3.2 双注入点方案（B+D）

hil-serl 的控制路径和观测路径是分开的，需要在两个点分别注入：

```
libfranka 硬件
    ↓ q_true
franka_hw
    │
    │ ┌──────────────────────────────────────────────────┐
    │ │ 路径1: 控制闭环（C++ 内存）                       │
    │ │                                                  │
    ├─┤→ FrankaStateHandle                               │
    │ │      ↓                                           │
    │ │  C++ 阻抗控制器 update() (1kHz)                   │
    │ │      q_true = state_handle_->getRobotState().q   │
    │ │      bias = nh_.getParam("/encoder_bias")        │
    │ │      q_biased = q_true + bias ←← 【注入点B】     │
    │ │      J = Jacobian(q_biased)                      │
    │ │      tau = J^T * F_impedance                     │
    │ │      → 执行偏差 ✓                                │
    │ └──────────────────────────────────────────────────┘
    │
    │ ┌──────────────────────────────────────────────────┐
    │ │ 路径2: 观测报告（ROS topic → HTTP）               │
    │ │                                                  │
    ├─┤→ franka_state_controller                         │
    │ │      ↓                                           │
    │ │  /franka_state_controller/franka_states (q_true) │
    │ │      ↓                                           │
    │ │  franka_server.py _set_currpos()                 │
    │ │      bias = rospy.get_param("/encoder_bias")     │
    │ │      self.q = q_true + bias ←← 【注入点D】       │
    │ │      ↓                                           │
    │ │  HTTP getstate → Actor                           │
    │ │      → 感知偏差 ✓                                │
    │ └──────────────────────────────────────────────────┘
    │
    │ ┌──────────────────────────────────────────────────┐
    │ │ 路径3: 旁路（无人使用）                           │
    │ │                                                  │
    └─┤→ joint_state_controller                          │
      │      ↓                                           │
      │  /joint_states → rviz（不影响控制和观测）          │
      └──────────────────────────────────────────────────┘
```

### 3.3 与标准 ROS `/joint_states` 注入的等价性

```
标准 ROS:                              hil-serl (B+D方案):

/joint_states 注入 bias                 两个注入点，同一个 bias 值
       │                                       │
       ├→ 控制器读 biased q              ├→ C++ 控制器用 biased q (注入点B)
       │   → 错误力矩                    │   → 错误 Jacobian → 错误力矩
       │   → 执行偏差 ✓                  │   → 执行偏差 ✓
       │                                 │
       └→ 上层读 biased q               └→ Flask 返回 biased q (注入点D)
           → 感知偏差 ✓                      → Actor 收到错误观测
                                             → 感知偏差 ✓

两者效果完全等价：
  bias → q_measured = q_true + bias
    ├→ 控制用错误 q → 执行偏差
    └→ 观测用错误 q → 感知偏差
```

**本质区别**：标准 ROS 只需要一个注入点（`/joint_states`）因为控制和观测都读同一个 topic；hil-serl 需要两个注入点因为控制和观测走不同的数据路径。

## 4. 各注入点总览

| 注入点 | 位置 | 影响 | 修改难度 | 说明 |
|--------|------|------|----------|------|
| **A** | franka_hw 硬件接口层 | 控制 ✓ 观测 ✓ | 高（改 franka_ros 源码） | 最底层，所有下游都受影响，但改 franka_ros 风险大 |
| **B** | C++ 阻抗控制器内部 | 控制 ✓ 观测 ✗ | 低（改几行 C++） | 只影响控制闭环，不影响观测报告 |
| **C** | FrankaState topic 拦截 | 控制 ✗ 观测 ✓ | 中（写 ROS 中间节点） | 只影响 Flask 的状态来源 |
| **D** | franka_server.py 内部 | 控制 ✗ 观测 ✓ | 低（改几行 Python） | 只影响 HTTP 返回的观测 |
| **E** | /joint_states topic | 控制 ✗ 观测 ✗ | 低 | hil-serl 下完全无效 |
| **B+D** | C++ 控制器 + Flask | 控制 ✓ 观测 ✓ | 低 | **推荐方案**，两处各改几行 |

## 5. 参考

- 仿真中的 bias 注入实现：`frrl/envs/base.py` (FrankaGymEnv.apply_action)
- EncoderBiasInjector：`frrl/fault_injection.py`
- hil-serl Flask Server：`~/hil-serl/serl_robot_infra/robot_servers/franka_server.py`
- hil-serl C++ 控制器：`~/hil-serl/serl_robot_infra/egg_flip_controller/src/cartesian_wrench_controller.cpp`
- 参考论文：*Case Study: ROS-based Fault Injection for Risk Analysis of Robotic Manipulator*

## 6. 真机实现与验证

本文档讨论的是**设计选型**。真机上的代码改动清单、编译启动流程、验证步骤、
物理行为分析和训练/部署配置指南在姊妹文档
[`fault_injection_realhw.md`](fault_injection_realhw.md) 中。

2026-04-15 已在真实 Franka Panda 上完成 B+D 端到端验证。
