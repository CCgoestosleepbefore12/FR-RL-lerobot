# FR-RL 项目进度与实验记录

> 约定：本文 **[sim]** 指 MuJoCo 仿真环境、**[real]** 指 Franka Panda 真机部署，**[共通]** 指两端都适用。仿真/真机部分参数已经 diverge（观测维度、旋转 scale、obs 通道组成等），务必对照所在标签。

## 一、项目概述

**题目**: 基于人类介入强化学习的机器人关节故障在线自适应

**一句话定位**: 当机器人存在未知编码器偏差时，通过带有人类引导的在线强化学习（RLPD + HIL），在不停机条件下恢复操作任务性能。

**核心方法** [共通]: RLPD（SAC + offline demo buffer + online buffer）+ HIL（人类键盘/SpaceMouse 干预）+ 编码器偏差注入 + Task/Backup/Homing 三态监督。

---

## 二、已完成的工作

### 2.1 代码架构（2026-04 Stage 1–2c 重构后）

[共通] 统一 `frrl` 包，Stage 2 拆分后的主要子目录：

```
frrl/
├── envs/
│   ├── sim/                        # [sim] MuJoCo 仿真
│   │   ├── base.py                 #   FrankaGymEnv 基类 + OSC 调用 + 偏差注入
│   │   ├── opspace.py              #   OSC 控制器（sim 专用）
│   │   ├── panda_pick_cube_env.py  #   PickCube（Exp 2 主环境）
│   │   ├── panda_pick_place_env.py #   PickPlace（Exp 1 task policy 消融）
│   │   ├── panda_pick_place_safe_env.py  # 带 hand + safety 层（Joint Safe RL 残留）
│   │   └── panda_backup_policy_env.py    # Backup S1/S2 + BiasJ1 + NoDR 变体
│   ├── real.py                     # [real] FrankaRealEnv（HTTP → RT PC Flask server）
│   ├── real_config.py              # [real] 相机 / action_scale / clip_box
│   ├── configs.py                  # [共通] env 配置 dataclass
│   └── wrappers/                   # [共通] gym wrapper 栈（EEAction / 键盘 / viewer）
├── robots/franka_real/             # [real] 真机外围
│   ├── servers/                    #   Flask franka_server + gripper server
│   ├── cameras/                    #   RealSense D455 读取器
│   ├── vision/                     #   WiLoR + YOLOv8 手检测 + ArUco TCP 追踪
│   └── trajectory_executor.py
├── rl/                             # [共通] 分布式训练框架
│   ├── core/                       #   actor / learner / learner_service / buffer / env_factory
│   ├── infra/                      #   process / queue / wandb / actor_utils / transport (gRPC)
│   └── supervisor/                 # [real] HierarchicalSupervisor + HomingController
├── policies/sac/                   # [共通] SAC 网络
├── processor/, teleoperators/, rewards/, optim/, datasets/, configs/, utils/
└── fault_injection.py              # [共通] EncoderBiasInjector（sim B+D / real 走 /set_encoder_bias 路由）
```

**Stage 1–2c 关键改动**（2026-04）：
- `scripts/` 按 sim/real/tools/hw_check 分类
- `frrl/rl/` 拆成 core / infra / supervisor 三层
- `frrl/controllers/` 合并到 `frrl/envs/sim/opspace.py`（只有 sim 消费）
- `frrl/envs/franka_real_env.py` → `frrl/envs/real.py`
- 外层 `configs/` 搬到 `scripts/configs/`（避开和 `frrl/configs/` 命名冲突）
- 废弃顶层 `demos/` 目录，真机 HIL pickle 路径统一到 `data/task_policy_demos/`
- [real] backup 旋转 scale 对齐：`BACKUP_ROTATION_SCALE=0.05 × LOOKAHEAD=2.0 = 0.10 rad/step`（= sim 的 `ROT_ACTION_SCALE=0.1`）

### 2.2 编码器偏差注入（一个偏差源 → 三重影响）

[共通] 偏差模型：

| 影响 | [sim] 实现 | [real] 实现 |
|------|-----------|-------------|
| ① 初始位置偏移 | `q_true = home - bias`（`sim/base.py::reset_robot`） | C++ 阻抗控制器收到 bias 后在跟踪 home 时自动偏离 |
| ② 执行偏差 (Jacobian 偏) | apply_action 前临时把 qpos 改写成 biased 值→ OSC → 还原（`sim/base.py::apply_action`） | serl_franka_controllers 内 `RealtimeBuffer<bias>` + biased FK/Jacobian |
| ③ 感知偏差 (FK 偏) | `qpos_measured = q_true + bias`, `tcp = biased_FK()`（`sim/base.py::get_robot_state`） | `/encoder_bias` topic → C++ 发布 `biased_state` → franka_server `/getstate` 返回 biased |

**[sim] 偏差类型**：固定（episode 内不变） / 随机 `U[-0.15, 0.15]` rad。
**[real] 当前目标关节 = Joint 0（J1）**：bias 范围 `[-0.15, 0.15]` rad，通过 Python 侧 HTTP → RT PC C++ 注入。
**[sim] PickCube 实验的历史目标关节 = Joint 4（J5）**，范围 `[0, 0.25]` rad（Exp 2 历史设置，保留用于论文 bias scan）。

### 2.3 观测设计（sim / real 已 diverge）

**[sim]** `PandaPickCube*` 任务：`agent_pos = 18D` + 可选 `environment_state` 附加通道（`block_pos(3) + noisy_real_tcp(3)` → 24D 变体）；wrist + front 双相机 128×128。

**[sim]** `PandaPickPlace*` 任务：27D full = `robot_state(18) + block(3) + plate(3) + real_tcp(3)`，支持 27/24/21D 三档消融（env 已实装，见 sim_exp_data.md Exp 1）。

**[real]** `FrankaRealEnv` observation：
- `agent_pos = 29D` = `joint_pos_biased(7) + joint_vel(7) + gripper(1) + tcp_biased(7) + tcp_true(7)`
- 其中 `tcp_true` 为 ArUco 追踪 TCP（privileged 作弊通道；sim 用 `noisy_real_tcp + σ=5mm` 高斯，真机当前无额外噪声）
- 相机：wrist + front 各 128×128，`shared_encoder=true`，冻结 **DINOv3-S** (2026-04-26 起；之前 ResNet10)

两端共通：
- Action unified 到 7D：`[dx, dy, dz, rx, ry, rz, gripper]`
- Vision encoder: frozen **DINOv3-S** (ViT-S/16, 22M frozen, 2026-04-26 起；之前 ResNet10)；约束双相机同尺寸由 `SACObservationEncoder.get_cached_image_features` 强制

### 2.4 训练框架

[共通]：
- 分布式 actor / learner via gRPC（`frrl/rl/infra/transport/`）
- Warmup pretraining：offline demo 上预训练 500–5000 步
- Offline+online 混合 replay（HIL-SERL 风格），干预 transition 同时写 offline
- RLPD：SAC + offline buffer，默认 50/50 online/offline 采样

### 2.5 真机部署（2026-04 新增）

两机系统（RT PC + GPU Workstation）详见
[`real_robot_deployment_plan.md`](real_robot_deployment_plan.md)。

**[real] RT PC 侧**
- PREEMPT_RT 5.15-rt83 内核、libfranka 0.9.1、franka_ros 0.9.1（源码编译）
- serl_franka_controllers 阻抗控制器 1 kHz 稳定
- 网络分离：`enp4s0 (172.16.0.1/24)` 直连 Franka，USB 网卡 `192.168.100.1/24` 连 GPU 机
- 一键启停：`~/start_franka_server.sh` / `~/kill_franka_server.sh`
- 启动细节：[`rt_pc_runbook.md`](rt_pc_runbook.md)

**[real] B+D 双注入点（2026-04-15 端到端验证）**
- 架构：[`fault_injection_architecture.md`](fault_injection_architecture.md)
- 实现/使用：[`fault_injection_realhw.md`](fault_injection_realhw.md)
- C++ 阻抗控制器加入 `RealtimeBuffer<std::array<double,7>>` + biased FK/Jacobian，通过 `/encoder_bias` topic 接收 Python 侧 bias，发布 `biased_state` 回传
- 上游 `serl_franka_controllers` 未 fork，改动存为 [`patches/serl_franka_controllers_bias_injection.patch`](../patches/serl_franka_controllers_bias_injection.patch)
- franka_server + gripper server 从 `~/hil-serl/` 搬入 `frrl/robots/franka_real/servers/`，启动方式改为 `python -m frrl.robots.franka_real.servers.franka_server`
- 新增 HTTP 路由：`/set_encoder_bias` / `/clear_encoder_bias` / `/get_encoder_bias`；`/getstate` 返回 biased `q`/`pose`
- **2026-04-15 验证**：`FrankaRealEnv.reset()` → `EncoderBiasInjector` → HTTP → C++ biased torque → J1 0.1 rad 约 7cm 物理扫动 → `/getstate` 返回 biased 观测

**[real] Backup Policy 真机部署**
- HierarchicalSupervisor（TASK / BACKUP / HOMING 三态）+ HomingController 已实装
- V2 几何：`ARM_SPHERE_RADIUS=0.10`、`ROTATION_COEFF=0.2`、`MAX_ROTATION=π`（软惩罚）
- **V3 全臂避障改造（2026-04-26 设计定稿）**：sim 加 5 球 (link3/4/5/6/hand) + obstacle r=0.10 + spawn (0.30,0.40) + hand_speed (0.015,0.030) + max_disp 0.40 + saturating proximity reward；FSM D_SAFE/D_CLEAR 升 0.40/0.45。解决 V2 "policy 只学 EE 避让，肘/前臂仍会撞"问题。env/config/test/script/docs 改完，训练待开始。详见 [`docs/sim_exp_data.md`](sim_exp_data.md) Backup S1 V3 段落。
- BiasMonitor 实时可视化 + 数据存盘，已集成进 `scripts/real/deploy_backup_policy.py`
- 相机-基座标定：`scripts/tools/calibrate_cam_to_robot.py` 默认用 SVD 点对齐，hand-eye 作为诊断
- 详见 [`backup_policy_deployment.md`](backup_policy_deployment.md)

**[real] 遥操作 / demo 采集**
- SpaceMouse 驱动 + 集成：[`spacemouse_teleop.md`](spacemouse_teleop.md)
- Task policy demo 采集：`scripts/real/collect_demo_task_policy.py` → `data/task_policy_demos/` pickle
- 四键键盘 reward 协议（S/Enter/Space/Backspace）+ `env.go_home()` 收尾

---

## 三、实验结果

详细原始输出与对比表见 [`sim_exp_data.md`](sim_exp_data.md)，本节只保留摘要。

### 3.1 [sim] Exp 2：PickCube 编码器偏差训练（已完成）

验证 H1/H2/H3：无偏差策略在 bias 下退化、随机偏差 + `real_tcp` 观测能学出全范围鲁棒策略。

**三策略 bias scan（J5 偏差，[0, 0.25] rad）**：

| 偏差 (rad) | 无偏差 18D | 固定偏差 24D | **随机偏差 24D** |
|-----------|-----------|-------------|------------------|
| 0.00 | 100% | 83% | **92%** |
| 0.05 | 100% | 87% | 100% |
| 0.10 | 80% | 99% | 99% |
| 0.15 | 62% | 100% | 96% |
| 0.20 | 51% | 100% | 97% |
| 0.25 | 5% | 100% | 96% |
| 0.30 | 0% | 99% | **99%** |

**关键发现**：随机偏差策略全范围 92–100%，泛化到训练分布外（0.3 rad 99%），无固定偏差策略的"无偏差下过度补偿"问题。

**失败实验（论文 negative results）**：18D 单步观测无法区分 bias → 42%；21D 仅加 block_pos 失败；24D 早期因干预 transition 误丢 99% → 失败。

### 3.2 [sim] Exp 3：Backup Policy S1（单移动障碍 + DR）

| 项 | 值 |
|---|---|
| env | `PandaBackupPolicyS1-v0`，28D obs，Action 6D，Episode 10 步 |
| utd=4，online=200k，discount=1.0，Frame stack 3 | |
| 训练时长 | ≈ 3h |
| **存活率** | **99.0%** (198/200) |
| 平均奖励 | +8.98 ± 2.53 |
| 平均最近距离 | 0.154 m |

四种运动模式（ARC / LINEAR / PASSING / STOP_GO）全部 ≥ 97.8%。Checkpoint：`checkpoints/backup_policy_s1/`。

### 3.3 [sim] Exp 5：Backup zero-shot 迁移到 BiasJ1（J1 ±0.15 rad）

| 指标 | S1 baseline | S1 + BiasJ1 zero-shot |
|------|-------------|----------------------|
| 存活率 | 99.0% | **98.5%** |
| 平均奖励 | +8.98 | **+9.13** |

0.5% 差距 ≈ 200ep 抽样噪声。Backup 使用真实 TCP sensor 而非 biased FK，所以 bias 对 Backup 几乎不可见 → 印证"DR 训练策略对观测扰动鲁棒"。

### 3.4 [sim] Exp 6：DR 策略在 NoDR env（反直觉发现）

| Env | 存活率 | hand_collision | 奖励方差 |
|-----|-------|----------------|---------|
| S1 (DR 训练分布) | **99.0%** | 0.5% | 2.53 |
| S1 + BiasJ1 | 98.5% | 0.5% | 2.11 |
| **S1 NoDR** | **96.5%** ↓ | **2.5%** ↑ | **3.61** ↑ |

DR 策略在"更简单"的 NoDR 环境反而退化（+43% 奖励方差、5× 碰撞）。结论：**DR 不是免费午餐，而是显式的感知先验注入**，需要匹配部署域的观测质量。

### 3.5 [sim] Exp 7：Backup S2 多障碍（1 移 + 1 静）

| 项 | 值 |
|---|---|
| env | `PandaBackupPolicyS2-v0`，114D (38×3 帧堆叠) |
| online=200k、**utd=4**（utd=8 100k 崩溃）、训练 3h 49min | |
| **存活率** | **83-91%**（两次 eval 方差较大） |
| hand_collision | 13.5%（S1 仅 0.5%） |
| 最弱模式 | ARC 75.0% / STOP_GO 81.8% |

从单障碍 99% 到多障碍 ~85%，碰撞率升 26×，**不是策略学不好而是多障碍几何本身更难**（逃生锥变窄 + 样本组合 50→2500）。PASSING 模式仍 91%。Checkpoint：`checkpoints/backup_policy_s2/`。

### 3.6 [sim] Backup S1 V2（新几何 + rotation budget 软惩罚）

V2 设计：`ARM_SPHERE_RADIUS=0.10`（配 R_GRIPPER=0.10 m）、`ROTATION_COEFF=0.2`、`MAX_ROTATION=π`，episode 20 步，`max_episode_steps=20`；引入 rotation budget soft penalty（替代旧的硬终止），eval 与训练同 env（之前 eval 错用 S1 v0）。

Checkpoints：
- `checkpoints/backup_policy_s1_v2_50k` — V2 50k 训练版
- `checkpoints/backup_policy_s1_v2_newgeom_35k` — 真机实验首选（小 ckpt）
- `checkpoints/backup_policy_s1_v2_newgeom_145k` — 训练峰值，真机备用

### 3.6.5 [sim] Backup S1 V3（全臂避障 + obstacle r=0.10 + proximity reward）— 设计定稿，训练待开始

V3 解决 V2 真机部署观察到的失效：policy 学会 EE 避让，但肘/前臂仍可能撞手。

改动总览（详见 `sim_exp_data.md` Backup S1 V3 段落）：
- arm collision: 单球 → 5 球 (link3:0.07, link4:0.07, link5:0.065, link6:0.065, hand:0.10)
- obstacle r: 0.035 → **0.10**（对齐真机 hand bbox 等效半径）
- spawn dist: (0.21, 0.30) → **(0.30, 0.40)**
- hand speed: (0.005, 0.015) → **(0.015, 0.030)** m/step
- max_displacement: 0.30 → **0.40**
- reward: 加 saturating proximity bonus（饱和 0.20 @ clearance 0.10）
- FSM `D_SAFE`/`D_CLEAR`: 0.30/0.35 → **0.40/0.45**

实施状态（2026-04-26）：
- ✅ env / 注册 / config / 启动脚本 / 单测 (21/21 pass) / FSM (CLI `--ckpt-version` 自动配阈值) / 文档
- ✅ 首训 300k 完成（6.5h, learner step）→ **71.5% 满存活率**（远低于 meta 预测 92-95%）
- ✅ Eval ckpt 落库 `checkpoints/backup_policy_s1_v3_300k_71pct/`
- ✅ Path A 重训中段（max_disp 0.40 → 0.50）：115k learner step → **89.5% 存活率**（+18%）。hand_collision 17.5% → 7%（multi-sphere 避让真学到了）。落库 `checkpoints/backup_policy_s1_v3b_115k_90pct/`
- ⏳ Path A 训完 300k（继续训中），最终 ckpt 落库 `backup_policy_s1_v3b_300k_*`

预期对比（V3 vs V2 145k）：

| 指标 | V2 (sim 100%, 真机存在肘撞) | V3 (sim 92-95%, 全臂避障) |
|---|---|---|
| Sim 满存活率 | 100% | 92-95%（worst-case 5-10% 不可解）|
| 真机肘/前臂撞 | **存在**（V2 collision 仅查 panda_hand 单球）| 根除（5 球全臂检测）|
| Reward 设计 | V4 (无 proximity) | V5 (saturating proximity，详见 `rl_reward.md`) |
| 真机部署阈值 | D_SAFE=0.30 / D_CLEAR=0.35 | D_SAFE=0.40 / D_CLEAR=0.45 |

V2→V3 sim 数字略降但**真机失效模式被根除**——质的改进。worst-case 推导见 `sim_exp_data.md` Backup S1 V3 段落。

### 3.7 [real] 真机实验（未开始）

真机训练首跑尚未进行。前置条件：workspace 标定、SpaceMouse demo 采集、真 dataset_stats。

---

## 四、当前系统参数

### 4.1 [共通] 训练配置（`scripts/configs/train_hil_sac_base.json`）

| 参数 | 值 |
|------|---|
| batch_size | 256 |
| utd_ratio | 2（sim backup S1/S2 单独用 4） |
| discount | 0.97（backup 用 1.0） |
| temperature_init | 0.01（sim） / 0.1（real，P0-2 防坍塌） |
| critic_lr / actor_lr / temp_lr | 3e-4 |
| critic_target_update_weight | 0.005 |
| num_critics | 2 |
| num_discrete_actions | 3（夹爪：开/关/保持） |
| online_buffer_capacity | 100,000 |
| offline_buffer_capacity | 100,000 |
| online_step_before_learning | 100（sim） / 500（real，P0-6） |
| warmup_steps | 500（真机 pretrain 目标 5000） |
| save_freq | 2,000 |
| vision_encoder | frozen DINOv3-S ViT-S/16（shared, 2026-04-26 起；之前 ResNet10）|
| state_encoder | Linear(D, 256) + LayerNorm + Tanh |
| actor / critic | [256, 256] |

### 4.2 环境配置

| 参数 | [sim] | [real] |
|------|-------|--------|
| 控制频率 | 10 Hz (`control_dt=0.1s`) | 10 Hz |
| 物理频率 | 500 Hz (`physics_dt=0.002s`) | libfranka 1 kHz 内层 |
| substeps | 50 | — |
| max_episode_steps | 200（task）/ 10（backup S1）/ 20（backup S1 V2）| 200 |
| action 空间 | 7D（`[dx,dy,dz,rx,ry,rz,gripper]`，历史 4D 已升级）| 7D |
| EE step size | 0.025 m/step（task）/ 0.03 m（backup） | `action_scale=0.03`，`max_cart_speed=0.30`（P0-3 后对齐：0.03×10Hz=0.30 m/s）|
| 旋转 scale | `ROT_ACTION_SCALE=0.1` | `BACKUP_ROTATION_SCALE=0.05 × LOOKAHEAD=2.0 = 0.10` |
| 偏差目标关节 | J5（PickCube 历史） / J1（PickPlace + Backup） | J1 |
| 偏差范围 | [0, 0.25] rad（J5） / [-0.15, 0.15] rad（J1） | [-0.15, 0.15] rad |
| noisy_real_tcp 噪声 | σ=5 mm | 无（`tcp_true` 作弊通道）|

---

## 五、可探索的改进方向

### 5.1 [sim] 观测改进

**显式偏差信号**（优先级高，改动小）：在 24D 上追加 `bias_signal = noisy_real_tcp - biased_tcp`（3D）→ 27D，直接告诉网络偏差方向。

### 5.2 [共通] 超参数调优

| 改动 | 当前 | 建议 | 理由 |
|------|-----|------|------|
| utd_ratio | 2（task） | 4 | WSRL，但注意 Backup S2 utd=8 已证明会崩 |
| num_critics | 2 | 10 | REDQ 风格，Q 稳定 |
| batch_size | 256 | 512 | 更稳 |

### 5.3 [共通] 网络结构

State encoder 加深到 2 层；考虑 DINOv2-Small 替换 ResNet10（约半天）。

### 5.4 [sim] 训练策略

课程学习：bias 范围按阶段扩大；offline 采样比例前期 80% 逐步调回 50%。

### 5.5 [共通] 算法层

Bias context encoder / GRU（从 K 步 action/Δstate 推断 bias，RMA 风格）；在线偏差估计器（最小二乘 from Jacobian）。

### 5.6 [共通] 多类型故障扩展

编码器噪声叠加、关节卡死、执行器退化、漂移偏差。

---

## 六、论文框架

### 核心假设

```
H1 [sim]: 编码器偏差显著降低操作任务成功率                    ✅ 已验证（Exp 2）
H2 [sim]: 无偏差策略在随机偏差下失效                          ✅ 已验证
H3 [sim]: RLPD + 外部定位观测 + 随机偏差训练 → 全范围鲁棒    ✅ 已验证（92-100%）
H4 [real]: 仿真训练策略能零样本/少样本迁移到真机              ⏳ 待验证
```

### 论文结构

第 1 章 Introduction／第 2 章 Problem Formulation (POMDP + 偏差因果链) ／第 3 章 Method (偏差注入 + 多维观测 + RLPD + HIL + Modular supervisor) ／第 4 章 Experimental Setup ／第 5 章 Results (sim bias scan + backup 迁移 + Real-world) ／第 6 章 Discussion。

### 核心贡献

1. 编码器偏差的精确因果链建模（B+D 双注入点）
2. 外部定位辅助观测（sim 24D+noisy_real_tcp / real 29D+tcp_true）
3. RLPD + HIL 在随机偏差下的鲁棒训练框架
4. Task + Backup + Homing 三态监督的 modular 部署架构
5. 仿真 + 真机双端验证

---

## 七、待办事项

### [共通] Vision encoder：ResNet10 → DINOv3-S（2026-04-26 切换）

动机：sim 分类任务用 ResNet10 失败，证明 ImageNet 预训 CNN 对 manipulation 任务特征不够分离。**升级 DINOv3-S** (ViT-S/16, 21M frozen, LVD-1689M 预训 1.69B 图)，dense feature 质量首次超过 weakly-supervised 模型。

实现：`frrl/policies/sac/modeling_sac.py::PretrainedImageEncoder` 加 ViT 适配（detect ViT vs CNN backbone, 丢 CLS+register tokens, reshape 成 4D feature map）；ResNet10 路径完全保留向后兼容。

参数对比（**完整 SACPolicy**：obs encoder + actor + 2 critics + targets + discrete critic + temperature；shared_encoder + freeze + 2 cam 128²）：

| 项 | ResNet10 (旧) | DINOv3-S (新) |
|---|---|---|
| trainable | 3.57M | 3.94M |
| total | 8.47M | 26.00M |
| frozen vision | 4.9M | 22.06M |

11 个 task/safe config 已切换到 `facebook/dinov3-vits16-pretrain-lvd1689m`。Backup policy 不用 vision，不受影响。

⚠️ **DINOv3 是 HF gated repo**：首次下载需 `huggingface-cli login` + 在 [HF 模型页](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) 上 accept license。

### [real] 真机部署（当前焦点）
- [x] RT PC 内核 + libfranka + franka_ros + serl_franka_controllers 部署
- [x] 两机网络隔离
- [x] franka_server + HTTP 路由
- [x] B+D 编码器偏差注入（C++ + server + biased_state topic）
- [x] `FrankaRealEnv` 端到端触发注入链路（2026-04-15）
- [x] SpaceMouse 遥操作 + demo 采集 pipeline
- [x] 相机-基座标定（`calibrate_cam_to_robot.py` SVD + hand-eye 诊断）
- [x] Task policy 阶段 1–5（FrankaRealEnv 29D / keyboard reward / actor discard hook / pickle adapter / offline pretrain）
- [x] Backup Policy 真机部署（HierarchicalSupervisor + HomingController + V2 + BiasMonitor）
- [ ] 标定 `abs_pose_limit`（工作空间边界）—— 需要手动引导采样
- [ ] ArUco 4 角 workspace 校准（`scripts/define_workspace_crop.py`）
- [ ] 50 条真机 demo + 真 dataset_stats（`scripts/tools/compute_dataset_stats.py`）
- [ ] 5000 步真 pretrain + wandb 收敛观察
- [x] Online HIL 训练接通（SpaceMouseTeleop 迟滞干预 + processor pipeline + `train_hil_sac_task_real.json`）
- [ ] 真机端到端首跑（硬件到位后跑 `bash scripts/real/train_hil_sac.sh task_real {learner,actor}` 联调）
- [ ] 真机训练首跑（`random_uniform(-0.15, 0.15)` J1 bias）
- [ ] 真机版 `eval_bias_curve_realhw.py`
- [x] Gripper penalty 激活（`train_hil_sac_task_real.json` 里 `gripper_penalty=-0.05`，新增 `FrankaGripperPenaltyProcessorStep` 对齐 sim 语义 action∈[-1,1]/state∈[0,1]，阈值 ±0.5/0.1/0.9；`step_env_and_process_transition` 把 `env.get_raw_joint_positions()` 塞进 `complementary_data["raw_joint_positions"]` 供 processor 读）
- [x] Learner offline_warmup/pretrain 防呆：`demo_pickle_paths=[]` + `dataset=null` + 非 resume 时直接 ValueError（避免 warmup 静默跳过）

### [sim] 优先级高
- [ ] 加 `bias_signal` 显式偏差信号 + UTD=4 重训
- [ ] PickPlace Task policy 观测消融：27D / 24D / 21D（env 已建，config 已建）
- [ ] 整理所有实验数据画正式图表

### [sim] 优先级中
- [ ] 课程学习实验
- [ ] State encoder 加深到 2 层
- [ ] `noisy_real_tcp` 噪声消融（σ=1/5/10/20 mm）
- [ ] Backup S2 reward shape（加 proximity signal，Kiemel 2024 风格，预期 +5–10%）

### [共通] 优先级低
- [ ] GRU context encoder
- [x] **DINOv3-S 替换 ResNet10** (2026-04-26 完成；ViT-S/16, 22M frozen, LVD-1689M 预训)
- [ ] 多关节偏差
- [ ] 多故障类型扩展

---

## 八、关键文件路径

| 文件 | 用途 | 标签 |
|------|------|------|
| `frrl/envs/sim/base.py` | 仿真环境基类 + 偏差注入 | [sim] |
| `frrl/envs/sim/panda_pick_cube_env.py` | PickCube 环境（Exp 2） | [sim] |
| `frrl/envs/sim/panda_pick_place_env.py` | PickPlace（Exp 1 消融） | [sim] |
| `frrl/envs/sim/panda_backup_policy_env.py` | Backup S1/S2/V2 | [sim] |
| `frrl/envs/real.py` | FrankaRealEnv（HTTP client） | [real] |
| `frrl/envs/real_config.py` | 相机 / action_scale / clip_box | [real] |
| `frrl/robots/franka_real/servers/franka_server.py` | RT PC Flask server | [real] |
| `frrl/rl/core/learner.py` | 训练循环 + warmup + offline-only mode | [共通] |
| `frrl/rl/core/actor.py` | 环境交互 + intervention 路径 | [共通] |
| `frrl/rl/core/env_factory.py` | env + processor factory | [共通] |
| `frrl/rl/supervisor/` | HierarchicalSupervisor + HomingController | [real] |
| `frrl/policies/sac/modeling_sac.py` | SAC 网络 | [共通] |
| `scripts/configs/train_hil_sac_base.json` | Task policy 训练配置 | [共通] |
| `scripts/configs/train_hil_sac_backup_s1_v2.json` | Backup V2 训练配置 | [sim] |
| `scripts/real/train_hil_sac.sh` | 训练启动脚本（conda 选 env） | [共通] |
| `scripts/real/deploy_backup_policy.py` | 真机 Backup 部署 | [real] |
| `scripts/real/collect_demo_task_policy.py` | SpaceMouse demo 采集 | [real] |
| `scripts/sim/eval_policy.py` | 单环境评估 | [sim] |
| `scripts/sim/eval_bias_curve.py` | bias scan | [sim] |
| `scripts/tools/compute_dataset_stats.py` | demo pickle 统计 | [real] |
| `scripts/tools/calibrate_cam_to_robot.py` | 相机-基座标定 | [real] |

---

## 九、实验检查点（Checkpoints）

### 9.1 仓内常驻（白名单入 git，`checkpoints/`）

| 目录 | 用途 | 状态 |
|------|------|------|
| `checkpoints/backup_policy_s1/` | Exp 3 Backup S1 DR，200k | 99.0% |
| `checkpoints/backup_policy_s1_v2_50k/` | V2 几何 50k | 训练中期 |
| `checkpoints/backup_policy_s1_v2_newgeom_35k/` | V2 新几何 | 部署用 `--ckpt-version v2` (D_SAFE=0.30/D_CLEAR=0.35) |
| `checkpoints/backup_policy_s1_v2_newgeom_145k/` | V2 新几何训练峰值 | 部署用 `--ckpt-version v2` (D_SAFE=0.30/D_CLEAR=0.35) |
| `checkpoints/backup_policy_s2/` | Exp 7 Backup S2 多障碍，200k | 83-91% |
| `checkpoints/backup_policy_s1_v3_300k_71pct/` | V3 首训（max_disp=0.40, 71.5%, ablation 对照） | 部署用 `--ckpt-version v3` |
| `checkpoints/backup_policy_s1_v3b_115k_90pct/` | V3 Path A 中段（max_disp=0.50, 89.5%, hand_collision 7%） | **真机首选**，`--ckpt-version v3` |
| `checkpoints/backup_policy_s1_v3b_300k_*` | V3 Path A 训完版（待训完后落库）| 同上 |

### 9.2 本地训练输出（`outputs/`，**不入 git**）

仅对 checkpoint 所有者有效，复现需从训练配置 + seed 重跑或联系所有者。

| 实验 | 路径 | 最佳 ckpt |
|------|------|----------|
| 无偏差 baseline | `outputs/train/2026-03-23/02-26-18_frrl_hil_sac_pick_cube/` | last |
| 固定偏差 0.2 rad | `outputs/train/2026-03-24/21-45-16_frrl_hil_sac_pick_cube_bias/` | last |
| 随机偏差 [0, 0.25] | `outputs/train/2026-03-25/19-26-27_frrl_hil_sac_pick_cube_bias_random/` | 010000 |

---

## 十、[real] Task Policy HIL-SERL 训练 Pipeline（阶段 1–5 完成，2026-04-23）

完整真机 pick-place task policy 训练框架已搭建，**阶段 1–5（观测升级 → keyboard reward → actor discard hook → SpaceMouse demo 采集 → offline pretrain）** 全部可工作。首次 smoke 100 步 pretrain `critic_loss=0.0135`，6M 可训参数 / 8M 总参，1s/step 真机吞吐验证通过。

详见 [`task_policy_training.md`](task_policy_training.md)。

### 阶段交付物

| 阶段 | 交付 | 说明 |
|------|------|------|
| 阶段 1 | `FrankaRealEnv` 29D 观测 + 双相机 | `joint_pos_biased(7) + joint_vel(7) + gripper(1) + tcp_biased(7) + tcp_true(7)`；`tcp_true` = ArUco 追踪 privileged 通道；双相机 128×128 shared encoder |
| 阶段 2 | Keyboard reward + `go_home` | S/Enter/Space/Backspace 四键；`KeyboardRewardListener` 状态机；`env.go_home()` 收尾 |
| 阶段 3 | Actor discard hook | `frrl/rl/infra/actor_utils.py::should_discard_episode` 按 `info["discard"]` 丢整条 rollout |
| 阶段 4 | SpaceMouse demo 采集 + pickle adapter | `scripts/real/collect_demo_task_policy.py` 输出 hil-serl schema pickle；`ReplayBuffer.from_pickle_transitions` 适配（key_map + HWC→CHW + resize + /255） |
| 阶段 5 | Offline-only pretrain | `SACConfig.offline_only_mode=True` 跳 gRPC actor，从 pickle 加载 `offline_replay_buffer` 跑 N 步 warmup；`scripts/tools/pretrain_task_policy.py` CLI |

### 4-agent code review 修复集合

session 末 4 agent（独立 review / vs sim / vs hil-serl 原仓库 / meta-review）发现并修复：

- **P0 Blocker**：HIL 链路三件套（intervene_action / complementary_info schema / learner 混样条件）、Actor teleop_action 接入、HTTP requests timeout=2.0、Image `/255` 归一化、Action speed cap `max_cart_speed=0.30`
- **P1（除 GripperPenaltyWrapper）**：`wait_for_start` 接 `shutdown_event`、pynput 缺失抛错、pickle `optimize_memory=False`、dataset_stats 默认值 + `compute_dataset_stats.py`
- **P2/P3**：reward `int→float`、`close()` 幂等、去重 deepcopy、RealSense reader sleep、`wandb.finish()`、rewards `__init__` re-export、resize_map 长度校验

### 测试覆盖

新增 8 个测试文件共 **115 个 pytest cases**，覆盖 config 默认字段、观测 shape layout（逐 slice 断言）、相机尺寸分发、键盘状态机（含并发压测）、pickle adapter（端到端 round-trip + 兼容性）、actor discard hook、offline_only_mode。执行时间 ≈ 2.6s。

### 关键决策

- **tcp_true 作弊通道**：真机无噪声（和 sim 的 `noisy_real_tcp` σ=5mm 不同），简化训练信号；future 消融可验证"无作弊"版本
- **双相机 128²**：来自 `modeling_sac.py::586` 硬编码同尺寸约束；future refactor 支持异构
- **shared_encoder=true**：frozen vision encoder（旧 ResNet10 / 现 DINOv3-S）无梯度，三副本浪费（ResNet10: 15.6M→8.47M；DINOv3-S: 70M→26M 总参）
- **action_scale 0.03（P0-3 后调整）**：原 0.04 × 10Hz = 0.40 m/s 会被 `max_cart_speed=0.30` 非线性压缩 0.75x，policy 学到的动作幅度与执行幅度不一致。改 0.03 后正好 0.30 m/s 触顶不削减；现阶段尚无 0.04 demo，安全切换

### 阶段 6 交付（2026-04-25 软件侧完结）

| 项 | 状态 | 交付物 |
|---|---|---|
| ArUco workspace 校准脚本 | ✅ 软件 | `scripts/real/define_workspace.py`（SpaceMouse 采样 + `make_workspace_roi_crop` factory） |
| BiasMonitor 可视化 | ✅ 软件 | `fault_injection.py::BiasMonitor` + `enable_bias_monitor` config flag |
| SpaceMouse 干预检测（迟滞） | ✅ 软件 | `SpaceMouseTeleop` enter=0.05 / exit=0.02 / persist=3 |
| Task Policy 真机 HIL config + shell | ✅ 软件 | `train_hil_sac_task_real.json` + `train_hil_sac.sh task_real` variant |
| GripperPenalty 激活 | ✅ 软件 | `gripper_penalty=-0.05` 对齐 sim |
| Phase 2 critic-only warmup（actor 冻结） | ✅ 软件 | `SACConfig.critic_only_online_steps` + `learner._should_freeze_actor` / `_should_run_actor_optimization`，配合 `__post_init__._validate_phase2()` 锁契约 |
| Resume 白名单超参覆盖 | ✅ 软件 | `learner.RESUMABLE_POLICY_OVERRIDES` + 覆盖后 `_validate_phase2()` 重校验 |
| ArUco 4 角 workspace 校准**执行** | ✅ 硬件（2026-04-26）| `scripts/hw_check/select_workspace_roi.py` 鼠标框选 → `roi_front_final=(200,171,408,379)` 208² 落档 `frrl/envs/real_config.py::image_crop["front"]` |
| `abs_pose_limit` 手动引导采样 | ✅ 硬件（2026-04-25）| 16 点采样 + `tour_workspace_corners.py` 实地巡 8 角 → `[0.386, -0.213, 0.175]` ~ `[0.709, 0.198, 0.315]` |
| Calibrate cam-to-robot **执行** | ✅ 硬件（2026-04-26）| 四元数根因 bug fix 后重标定，`calibration_data/T_cam_to_robot.npy` |
| 50 条真机 demo 采集 | ⏳ 硬件 | 当前 13 条（wipe 任务，gripper_locked='closed'）；继续 `--gripper closed -n 37` 凑齐 |
| 真 dataset_stats | ✅ 硬件（13 demos）| `dataset_stats_generated.json` 已落 `train_task_policy_franka.json`；新加 const-channel guard 防 wipe action[6] std=0 NaN |
| 5000 步真 pretrain + wandb | ⏳ 硬件 | smoke 500 步通过（critic_loss 0.26→0.014，2026-04-26）；50 demos 凑齐后跑正式 5000 步 |
| 真机端到端首跑 | ⏳ 硬件 | `task_real learner` + `task_real actor` 双终端 |
| `eval_bias_curve_realhw.py` | ⏳ 硬件 | 训完 ckpt 后再搭，现在搭无价值 |

### 阶段 6 后续 P0 风险审查（2026-04-25 round 4 修复）

经 3-reviewer + meta consolidator 审查，定位 6 个真机 blocker，已全部修复：

| ID | 议题 | 修复 |
|---|---|---|
| P0-1 | `demo_resize_images` 缺 wrist key 导致两路相机 cat shape 静默不一致 | JSON 加 `observation.images.wrist`；`get_cached_image_features` 加 spatial shape assert |
| P0-2 | `temperature_init=0.01` 起点过低 + pretrain 热身 α 漂移导致后续坍塌 | `temperature_init=0.1`；新增 `freeze_temperature_in_pretrain=True` gate；`target_entropy=-3.5`；`discrete_penalty` 进 continuous critic td_target |
| P0-3 | `action_scale[0]=0.04` × 10Hz = 0.40 m/s 被 `max_cart_speed=0.30` 非线性压缩 | `action_scale[0]→0.03`，policy 学到/执行幅度一致 |
| P0-4 | frozen ResNet10 BN 在 train mode 下 running_mean 仍漂移；shared encoder 中 state/env path 未对 actor detach | `freeze_image_encoder` 加 `.eval()`；`SACPolicy.train` override 强制 image encoder 永远 eval；`SACObservationEncoder.forward(detach=...)` 对 state/env 也加 detach 分支 |
| P0-5 | 真机 HTTP 单次失败直接 raise 中断 episode；reset 直线插值有撞撞风险 | 新增 `_http_post` 指数退避重试（pose/getstate/clearerr 等幂等端点）；`jointreset` 单次硬失败；`_go_to_reset` 加 z-lift 抬升避免工件下卡住（**round 4：升级为 lift→horizontal transit→descend 三段路径**）|
| P0-6 | NaN transition silent skip 让 critic 梯度污染；online buffer 起步过早导致 batch shrink | `check_nan_in_transition` 改 `if NaN: continue`（外层补 `optimization_step += 1` 避免 save_freq 卡死）；`online_step_before_learning=500`、`critic_only_online_steps=2000` |

### 阶段 6 真机部署 round 5 修复（2026-04-26 demo 采集前）

第二轮真机调试，解决 backup deploy + demo 采集中实测发现的问题：

| ID | 议题 | 修复 |
|---|---|---|
| R5-1 | WiLoR YOLO 把 franka gripper hand 误识别成人手，supervisor 永触 BACKUP 乱躲 | `HandDetector.detect()` 双参考点几何过滤：`flange_radius=0.10` 球（手掌+腕部）∨ `tcp_radius=0.06` 球（finger 末端），3D 距离任一命中视为 self-detection 丢弃 |
| R5-2 | HOMING 误差大（~20mm 稳态）+ 阻抗收尾抖动单帧瞬时切走 | `homing_action_scale=MAX_CART_SPEED/CTRL_HZ=0.03` 让 P 控制器算的 step 等于 deploy 实际能发的（kp=1 真正 deadbeat）；`is_done` 加 `done_consecutive_n=3` streak 闸门 |
| R5-3 | env reset 不处理夹爪，pick-place 上一 episode 抓的物体被带飞 | `_go_to_reset` 加根据 `gripper_locked` mode 强制 open/close + `gripper_sleep` 阻塞等机械到位再 lift |
| R5-4 | 双相机 live view 显示原始 640×480 BGR 两个独立窗口，对应不上 vision encoder 真实输入 | `_render_live_view` 改单窗 hstack：left front / right wrist，两路都是 image_crop+resize 后的 128² 图，3× upscale 让小目标可见 |
| R5-5 | 按 S 开始 episode 时 BiasMonitor 弹 "Save figure" 对话框（matplotlib 默认 's' 绑 save） | `_try_init_plot` 创建 figure 前清掉 keymap.save/quit/quit_all/fullscreen/home |
| R5-6 | 任务级夹爪锁定（wipe 海绵 / push 推动）需要全程闭合或张开 | `FrankaRealConfig.gripper_locked: "none" \| "closed" \| "open"`，env.step 锁定模式跳过 `_send_gripper_command`；collect_demo `--gripper closed` 屏蔽 SpaceMouse 按键，action[6]≡-1 |
| R5-7 | wipe 任务 action[6] 锁定 -1 → dataset_stats min=max → MIN_MAX 归一化 (x-min)/0=NaN | `compute_dataset_stats.py` 检测 (max-min) < 1e-6 const channel，自动 max=min+1，归一化恒等于 0 不爆 NaN |
| R5-8 | front workspace ROI 非正方形 412×326（z_plane 投影 4 角太扁），SAC encoder resize 失真 | 新增 `scripts/hw_check/select_workspace_roi.py` 鼠标拖拽框选 + auto-square + 写回 workspace.json，避免依赖几何投影 |
| R5-9 | BiasMonitor subplot 标题写死第一个 episode bias，后续 episode 切换不更新 | `mark_episode_boundary` 每次刷新 active subplot title 为当前 ep bias 值 |
