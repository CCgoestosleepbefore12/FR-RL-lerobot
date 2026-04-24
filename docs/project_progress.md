# FR-RL 项目进度与实验记录

## 一、项目概述

**题目**: 基于人类介入强化学习的机器人关节故障在线自适应

**一句话定位**: 当机器人存在未知编码器偏差时，通过带有人类引导的在线强化学习（RLPD + HIL），在不停机条件下恢复操作任务性能。

**核心方法**: RLPD（SAC + offline demo buffer + online buffer）+ HIL（人类键盘干预）+ 编码器偏差注入

---

## 二、已完成的工作

### 2.1 代码架构（已完成）

统一的 `frrl` 包，合并了 lerobot HIL-SERL + MuJoCo 仿真环境：

```
frrl/
├── envs/
│   ├── base.py                    # FrankaGymEnv（统一基类 + OSC + 偏差注入）
│   ├── panda_pick_cube_env.py     # PickCube 任务（主实验环境）
│   ├── panda_pick_place_env.py    # PickPlace 任务
│   ├── panda_arrange_boxes_env.py # ArrangeBoxes 任务
│   └── wrappers/                  # EEAction + 键盘干预 + Viewer 等
├── controllers/opspace.py         # 统一 OSC 控制器
├── fault_injection.py             # 编码器偏差注入器
├── rl/                            # Actor-Learner 训练框架
├── policies/sac/                  # SAC 策略
└── ...
```

### 2.2 编码器偏差注入模型（已完成）

一个偏差源 → 三重影响：

| 影响 | 仿真实现 | 代码位置 |
|------|---------|---------|
| ① 初始位置偏移 | `q_true = home - bias` | base.py reset_robot() |
| ② 执行偏差（Jacobian偏） | 临时替换 qpos → OSC计算 → 恢复 | base.py apply_action() |
| ③ 感知偏差（FK偏） | `qpos_measured = q_true + bias`, `tcp = biased_FK()` | base.py get_robot_state() |

偏差类型：
- 固定偏差（episode内不变）
- 随机偏差（每episode从 U[0, 0.25] rad 采样）
- 目标关节：Joint 4（肘关节）

### 2.3 观测设计（已完成）

24D 观测向量：

| 维度 | 内容 | 来源 | 准确性 |
|------|------|------|--------|
| 0-6 | 关节角度 qpos | 编码器 | ❌ biased |
| 7-13 | 关节速度 qvel | 编码器导数 | ✓ 正确 |
| 14 | 夹爪位置 | 控制指令 | ✓ 正确 |
| 15-17 | 末端位置 tcp | biased FK | ❌ biased |
| 18-20 | 方块位置 block_pos | 外部传感器 | ✓ 正确 |
| 21-23 | 真实末端 noisy_real_tcp | 外部定位+5mm噪声 | ✓ 大致正确 |

+ 双目图像（front 128×128 + wrist 128×128）→ ResNet10（冻结）→ 特征

### 2.4 训练改进（已完成）

- **Warmup 预训练**: 正式训练前在 offline demo 上预训练 500 步
- **人类干预数据**: 所有干预 transition 额外加入 offline buffer
- **Reward 修正**: sparse reward 使用 `_is_success()`（距离<5cm 且 举高>20cm）
- **Buffer 保存**: 训练结束时保存 online/offline buffer（支持 resume）

### 2.5 真机部署与故障注入（2026-04 新增）

两机系统（RT PC + GPU Workstation）拓扑和分工见
[`real_robot_deployment_plan.md`](real_robot_deployment_plan.md)。
本阶段完成的关键模块：

**RT PC 侧基础栈**
- 装配 PREEMPT_RT 5.15-rt83 内核、libfranka 0.9.1、franka_ros 0.9.1（源码编译）
- serl_franka_controllers 源码编译，阻抗控制器 1 kHz 运行稳定
- 网络分离：`enp4s0 (172.16.0.1/24)` 直连 Franka，USB 网卡 `(192.168.100.1/24)`
  连 GPU 工作站
- `~/start_franka_server.sh` / `~/kill_franka_server.sh` 一键启停脚本
- 完整启动流程文档 [`rt_pc_runbook.md`](rt_pc_runbook.md)

**故障注入（B+D 双注入点，真机验证通过）**
- 架构设计：[`fault_injection_architecture.md`](fault_injection_architecture.md)
- 真机实现/使用：[`fault_injection_realhw.md`](fault_injection_realhw.md)
- C++ 阻抗控制器加入 `RealtimeBuffer<std::array<double,7>>` + biased FK/Jacobian
  计算，通过 `/encoder_bias` topic 接收 Python 侧写入的 bias，发布 `biased_state`
  topic 回传给 franka_server。上游 `serl_franka_controllers` 仓库未 fork，修改
  存成 [`patches/serl_franka_controllers_bias_injection.patch`](../patches/serl_franka_controllers_bias_injection.patch)
  入库
- `franka_server.py` + 依赖的 gripper server 从 `~/hil-serl/` 搬进
  `frrl/robots/franka_real/servers/`，彻底结束"代码分散在 hil-serl 和本仓库两边"的历史债；
  启动方式从 `python franka_server.py` 改为 `python -m frrl.robots.franka_real.servers.franka_server`
- 新增 `/set_encoder_bias` / `/clear_encoder_bias` / `/get_encoder_bias` 三个 HTTP
  路由，`/getstate` 返回的 `q`/`pose` 是 biased 值
- GPU 侧 `FrankaRealEnv._set_encoder_bias` hook 和路由名天然对齐，
  无需额外改动
- **2026-04-15 端到端验证通过**：`FrankaRealEnv.reset()` 驱动 `EncoderBiasInjector`
  采样 bias → HTTP → C++ biased torque → 真实物理扫动 ~7cm（对 J1 0.1 rad） →
  `/getstate` 返回 biased 观测

**遥操作**
- SpaceMouse 驱动与集成方案：[`spacemouse_teleop.md`](spacemouse_teleop.md)

---

## 三、实验结果

### 3.1 Baseline: 无偏差策略（18D观测）

| 指标 | 值 |
|------|---|
| 训练环境 | PandaPickCubeKeyboard-v0（无偏差） |
| 观测维度 | 18D |
| 训练步数 | ~13K actor steps |
| 成功率 | **100%**（50/50 episodes） |
| 平均步数 | 13.4 ± 0.6 |

### 3.2 H1验证: 无偏差策略在偏差环境下的退化

| 偏差(rad) | 成功率 | 平均步数 |
|-----------|--------|---------|
| 0.00 | 100% | 13.5 |
| 0.05 | 100% | 13.5 |
| 0.10 | 80% | 55.9 |
| 0.15 | 62% | 62.5 |
| 0.20 | 51% | 99.8 |
| 0.25 | 5% | 184.5 |
| 0.30 | 0% | 191.1 |

**结论**: 偏差 >0.1 rad 开始显著影响，>0.25 rad 基本完全失败。

### 3.3 固定偏差策略（24D观测，含 real_tcp）

| 指标 | 值 |
|------|---|
| 训练环境 | PandaPickCubeBiasJ4Fixed02Keyboard-v0（固定0.2rad） |
| 观测维度 | 24D（含 block_pos + noisy_real_tcp） |
| 训练成功率 | 92% |

**评估结果（偏差曲线）：**

| 偏差(rad) | 无偏差策略 | 固定偏差策略 |
|-----------|-----------|------------|
| 0.00 | 100% | 83% |
| 0.05 | 100% | 87% |
| 0.10 | 80% | 99% |
| 0.15 | 62% | 100% |
| 0.20 | 51% | **100%** |
| 0.25 | 5% | 100% |
| 0.30 | 0% | 99% |

**结论**: 策略学到了针对 0.2rad 的补偿，在训练偏差附近最优，但小偏差时过度补偿（83%）。

### 3.4 随机偏差策略（24D观测，核心实验）

| 指标 | 值 |
|------|---|
| 训练环境 | PandaPickCubeBiasJ4RandomKeyboard-v0（[0, 0.25]rad） |
| 观测维度 | 24D |
| 最佳checkpoint | 10K learner steps |
| demo数量 | 50 episodes |

**评估结果（10K checkpoint 偏差曲线）：**

| 偏差(rad) | 无偏差策略 | 固定偏差策略 | **随机偏差策略** |
|-----------|-----------|------------|----------------|
| 0.00 | 100% | 83% | **92%** |
| 0.05 | 100% | 87% | **100%** |
| 0.10 | 80% | 99% | **99%** |
| 0.15 | 62% | 100% | **96%** |
| 0.20 | 51% | 100% | **97%** |
| 0.25 | 5% | 100% | **96%** |
| 0.30 | 0% | 99% | **99%** |

**关键发现：**
1. 随机偏差策略全范围 92-100%，是最均衡的
2. 甚至泛化到训练范围外（0.3rad 99%成功）
3. 无偏差时也能100%接近（92%），不像固定偏差策略过度补偿
4. 证明 RLPD + 24D观测（含外部定位）能训练出对任意偏差鲁棒的策略

### 3.5 关键失败实验记录

| 实验 | 观测 | 结果 | 失败原因 |
|------|------|------|---------|
| 随机偏差 18D（无real_tcp） | 18D | 82%→42% 退化 | 单步观测无法区分不同bias |
| 随机偏差 21D（+block_pos） | 21D | 前期靠人工,后期全失败 | 知道目标但不知道自己真实位置 |
| 随机偏差 24D（数据丢弃bug） | 24D | 失败 | 干预数据99%被错误丢弃 |

**教训**:
- 18D/21D 失败证明了 noisy_real_tcp 是关键信息
- 数据丢弃 bug 证明了 offline buffer 的重要性
- 这些失败实验在论文中是有价值的 negative results

---

## 四、当前系统参数

### 训练配置 (train_hil_sac_base.json)

| 参数 | 值 |
|------|---|
| batch_size | 256 |
| utd_ratio | 2 |
| discount | 0.97 |
| temperature_init | 0.01 |
| critic_lr / actor_lr / temperature_lr | 3e-4 |
| critic_target_update_weight | 0.005 |
| num_critics | 2 |
| num_discrete_actions | 3（夹爪：开/关/保持）|
| online_buffer_capacity | 100,000 |
| offline_buffer_capacity | 100,000 |
| online_step_before_learning | 100 |
| warmup_steps | 500 |
| save_freq | 2,000 |
| vision_encoder | ResNet10（冻结）|
| state_encoder | Linear(24, 256) + LayerNorm + Tanh |
| actor_network | [256, 256] |
| critic_network | [256, 256] |

### 环境配置

| 参数 | 值 |
|------|---|
| 控制频率 | 10 Hz (control_dt=0.1s) |
| 物理频率 | 500 Hz (physics_dt=0.002s) |
| substeps | 50 |
| max_episode_steps | 200 |
| action空间 | 4D（3D xyz + 1D 离散夹爪）|
| EE step size | 0.025 m/step |
| 偏差范围 | [0, 0.25] rad（Joint 4）|
| noisy_real_tcp 噪声 | σ=5mm |

---

## 五、可探索的改进方向

### 5.1 观测改进

**显式偏差信号（优先级：高，改动：小）**

在 24D 观测上追加 `bias_signal = noisy_real_tcp - biased_tcp`（3D），变成 27D。
直接告诉网络偏差方向和大小，不需要网络自己学减法。

```python
bias_signal = noisy_real_tcp - biased_tcp  # 显式偏差信号
agent_pos = concat([robot_state(18), block_pos(3), noisy_real_tcp(3), bias_signal(3)])  # 27D
```

消融实验：有/无 bias_signal 的对比。

### 5.2 超参数调优

| 改动 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| utd_ratio | 2 | **4** | WSRL推荐，样本效率翻倍 |
| num_critics | 2 | **10** | REDQ风格，Q估计更稳定 |
| batch_size | 256 | **512** | 更大batch更稳定 |

### 5.3 网络结构改进

**State Encoder 加深（优先级：中）**

```
当前: Linear(24, 256) → LayerNorm → Tanh  (1层)
改进: Linear(24, 256) → ReLU → Linear(256, 256) → LayerNorm → Tanh  (2层)
```

更深的 MLP 更容易学到 biased_tcp 和 real_tcp 之间的非线性关系。

**Vision Encoder 升级（优先级：低，改动：大）**

ResNet10 → DINOv2-Small。DINOv2 的自监督特征有更好的空间理解能力。
需要修改 PretrainedImageEncoder 支持 ViT 架构（约半天工作量）。

### 5.4 训练策略改进

**课程学习（优先级：中）**

```
Phase 1 (0-5K steps):    偏差 [0, 0.10] rad  — 先学简单的
Phase 2 (5K-15K steps):  偏差 [0, 0.20] rad  — 逐步增大
Phase 3 (15K+ steps):    偏差 [0, 0.25] rad  — 全范围
```

防止一开始就被大偏差的失败数据淹没。

**Offline 采样比例调整（优先级：中）**

当前 50/50 混合。可以前期 80% offline + 20% online（防退化），后期逐步调回 50/50。

### 5.5 算法层面改进

**Bias Context Encoder / GRU（优先级：中，改动：大）**

从最近 K 步的 (action, Δstate) 推断偏差上下文。
但之前分析 relative motion 差异小，信号可能弱。
参考 RMA (Kumar et al., RSS 2021)。

需要和当前 24D（含 real_tcp）方案做对比：
- 24D + real_tcp：需要外部定位系统
- GRU context：不需要外部传感器，纯 proprioception
- 两者适用场景不同

**在线偏差估计器（优先级：中）**

用物理模型（标称 Jacobian）+ 最小二乘法从几步交互数据中估计 bias 向量。
确定性算法，可解释性强。估计的 bias 作为策略额外输入。

### 5.6 多类型故障扩展

| 故障类型 | 实现复杂度 | 描述 |
|----------|-----------|------|
| 编码器噪声 | 简单 | bias 上叠加高斯噪声 |
| 关节卡死 | 中等 | 某关节力矩限幅/锁定 |
| 执行器退化 | 简单 | 力矩输出乘衰减系数 |
| 漂移偏差 | 中等 | bias 在 episode 内缓慢变化 |

### 5.7 Sim-to-Real

- Franka Panda 实机部署
- 软件层注入编码器偏差（修改关节角度读数）
- 外部相机获取方块位置和末端真实位置
- 仿真训练策略 → 真机评估

---

## 六、论文框架

### 核心假设

```
H1: 编码器偏差显著降低操作任务成功率             ✅ 已验证
H2: 无偏差策略在随机偏差下失效                   ✅ 已验证
H3: RLPD + 外部定位观测 + 随机偏差训练           ✅ 已验证
    → 全范围鲁棒 (92-100%)
```

### 论文结构

```
第1章 Introduction
第2章 Problem Formulation（POMDP + 偏差因果链）
第3章 Method（偏差注入 + 24D观测设计 + RLPD + HIL）
第4章 Experimental Setup
第5章 Results（偏差曲线对比 + 消融实验）
第6章 Discussion（局限性 + 未来方向）
```

### 核心贡献

1. **编码器偏差的精确因果链建模**（一个故障→三重影响）
2. **外部定位辅助的观测设计**（24D: biased state + real_tcp + block_pos）
3. **RLPD + HIL 在随机偏差下的鲁棒训练框架**
4. **仿真实验验证**（三种策略的偏差曲线对比）

---

## 七、待办事项

### 真机部署（当前焦点）
- [x] RT PC 内核 + libfranka + franka_ros + serl_franka_controllers 部署
- [x] 两机网络隔离（Franka 直连 + GPU 工作站旁路）
- [x] franka_server.py 启动 + 基本 HTTP 路由可用
- [x] B+D 编码器偏差注入（C++ 控制器 + franka_server + biased_state topic）
- [x] GPU 侧 `FrankaRealEnv` 端到端触发注入链路验证
- [ ] 标定 `abs_pose_limit`（工作空间边界）—— 需要手动引导采样
- [ ] 相机标定 + T_cam_to_robot（用于 block_pos 检测）
- [ ] SpaceMouse 遥操作 + demo 采集（用于 offline buffer）
- [ ] 真机训练首跑（`random_uniform(0, 0.25)` J1 bias）
- [ ] 真机版 `eval_bias_curve_realhw.py`

### 优先级高（仿真侧）
- [ ] 加 bias_signal 显式偏差信号 + UTD=4 重新训练
- [ ] 补充消融实验：有/无 block_pos、有/无 real_tcp、不同噪声水平
- [ ] 整理所有实验数据画正式图表

### 优先级中
- [ ] 课程学习实验
- [ ] State encoder 加深到2层
- [ ] noisy_real_tcp 噪声消融（σ=1mm, 5mm, 1cm, 2cm）

### 优先级低
- [ ] GRU context encoder 实现
- [ ] DINOv2 替换 ResNet10
- [ ] 多关节偏差实验
- [ ] 多故障类型扩展

---

## 八、关键文件路径

| 文件 | 用途 |
|------|------|
| `frrl/envs/sim/base.py` | 环境基类 + 偏差注入 |
| `frrl/envs/sim/panda_pick_cube_env.py` | PickCube 环境（24D观测）|
| `frrl/rl/core/learner.py` | Learner（训练循环 + warmup）|
| `frrl/rl/core/actor.py` | Actor（环境交互）|
| `frrl/rl/core/env_factory.py` | 环境+处理器工厂 |
| `frrl/policies/sac/modeling_sac.py` | SAC 网络结构 |
| `scripts/configs/train_hil_sac_base.json` | 训练配置 |
| `scripts/real/train_hil_sac.sh` | 训练启动脚本 |
| `scripts/sim/eval_policy.py` | 单环境评估 |
| `scripts/sim/eval_bias_curve.py` | 偏差曲线评估 |
| `docs/data_flow.md` | 完整数据流文档 |
| `docs/fault_simulation_design.md` | 仿真设计文档 |

## 九、实验检查点（Checkpoints）

> ⚠️ 下表的 `outputs/` 路径是本地训练输出目录，**不入 git**，仅对 checkpoint
> 原主有效。其他人要复现需要从训练配置 + seed 重跑，或联系 checkpoint 所有者。

| 实验 | 路径（本地） | 最佳ckpt |
|------|------|---------|
| 无偏差baseline | outputs/train/2026-03-23/02-26-18_frrl_hil_sac_pick_cube/ | last |
| 固定偏差0.2rad | outputs/train/2026-03-24/21-45-16_frrl_hil_sac_pick_cube_bias/ | last |
| 随机偏差[0,0.25] | outputs/train/2026-03-25/19-26-27_frrl_hil_sac_pick_cube_bias_random/ | 010000 |

---

## 十、Task Policy 真机 HIL-SERL 训练 Pipeline（阶段 1–5 完成，2026-04-23）

完整的真机 pick-place task policy 训练框架已搭建完毕，**阶段 1–5（观测升级 → keyboard reward → actor discard hook → SpaceMouse demo 采集 → offline pretrain）**全部可工作。首次 smoke 测试 100 步 pretrain，`critic_loss=0.0135`，6M 可训参数 / 8M 总参，1s/step 真机数据吞吐验证通过。

**详细文档见 [`docs/task_policy_training.md`](task_policy_training.md)**。

### 阶段交付物

| 阶段 | 交付 | 说明 |
|------|------|------|
| **阶段 1** | `FrankaRealEnv` 29D 观测 + 双相机 | `joint_pos_biased(7) + joint_vel(7) + gripper(1) + tcp_biased(7) + tcp_true(7)`；`tcp_true` 为 privileged 作弊通道；双路相机 128×128 共享 encoder |
| **阶段 2** | Keyboard reward + `go_home` | S/Enter/Space/Backspace 四键协议；`KeyboardRewardListener` 状态机；`env.go_home()` 收尾安全复位 |
| **阶段 3** | Actor discard hook | `frrl/rl/infra/actor_utils.py::should_discard_episode` 在 episode 结束时按 `info["discard"]` 丢整条 rollout |
| **阶段 4** | SpaceMouse demo 采集 + pickle adapter | `scripts/real/collect_demo_task_policy.py` 输出 hil-serl schema pickle；`ReplayBuffer.from_pickle_transitions` 适配器（key_map + HWC→CHW + resize + /255 normalize） |
| **阶段 5** | Offline-only pretrain | `SACConfig.offline_only_mode=True` 跳过 gRPC actor，从 pickle 加载到 `offline_replay_buffer` 跑 N 步 warmup loop 保存 checkpoint；`scripts/tools/pretrain_task_policy.py` 薄 CLI |

### 4-agent 代码 Review + P0/P1/P2/P3 修复集合

session 末 4 agent（独立 review / vs 仿真 / vs hil-serl 原仓库 / meta-review）发现并修复了以下问题：

**P0 Blocker（全部已修）**：
- HIL 链路三件套（intervene_action 字段、complementary_info schema、learner 混样条件）
- Actor teleop_action 接入（RLPD intervention 路径）
- HTTP requests timeout=2.0（防止 server 卡死阻塞 loop）
- Image `/255` 归一化（对齐 online processor，避免 offline/online 分布漂移）
- Action speed cap（`max_cart_speed=0.30`，保留 action_scale 0.04）

**P1（除 GripperPenaltyWrapper 外已修）**：
- `wait_for_start` 接 `shutdown_event`
- pynput 缺失在 `required=True` 时抛错
- Pickle 路径 `optimize_memory=False`（避免最后一条 next_state bug）
- dataset_stats 合理默认值 + `scripts/tools/compute_dataset_stats.py` 工具

**P2/P3**：reward `int→float`、`close()` 幂等、去重双重 deepcopy、RealSense reader sleep、`wandb.finish()`、rewards `__init__` re-export、resize_map 长度校验。

### 测试覆盖

新增 8 个测试文件共 **115 个 pytest cases**，覆盖：config 默认字段、观测 shape layout（逐 slice 断言）、相机尺寸分发、键盘状态机（含并发压测）、pickle adapter（含端到端 round-trip + 兼容性测试）、actor discard hook、offline_only_mode config。执行时间约 2.6s。

### 关键决策记录

- **tcp_true 作弊通道**：无噪声（和 sim 的 `noisy_real_tcp` + 5mm 高斯不同），意在简化训练信号；future 消融实验可验证"无作弊通道"情况。
- **双相机 128²**：来自 SAC `modeling_sac.py:586` 硬编码同尺寸约束；future refactor 可支持异构。
- **`shared_encoder=true`**：frozen ResNet10 本无梯度，三份副本浪费（15.6M → 8.47M total params）。
- **Action scale 0.04 保留**：和 demo 采集 scale 一致避免分布漂移；用 `max_cart_speed=0.30` 做硬件 safety net。

### 下一步（阶段 6 待做）

1. **ArUco 4 角 workspace 校准**：`scripts/define_workspace_crop.py`，替换 `workspace_roi_crop_placeholder`
2. **50 条真实 demo + 真 dataset_stats**：用 `scripts/tools/compute_dataset_stats.py` 算完填回 config
3. **5000 步真 pretrain + wandb**：观察 critic_loss 收敛
4. **Online HIL 训练接通**：`InterventionWrapper` 包 FrankaRealEnv + SpaceMouse override；actor resume from pretrain checkpoint
5. **GripperPenaltyWrapper port**（P1 剩项）：对齐 sim 的 `-0.05` 夹爪频繁切换惩罚
