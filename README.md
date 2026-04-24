# FR-RL-lerobot: Fault-Resilient RL for Franka Panda

模拟工业机器人编码器校准偏差，用 RLPD + HIL（人类干预）训练对偏差鲁棒的操作策略。
仿真用 MuJoCo，真机用 Franka Panda (libfranka 0.9.1 + franka_ros)。

## 项目结构

```
FR-RL-lerobot/
├── frrl/                    # 主 Python 包
│   ├── envs/                # MuJoCo 仿真环境（pick-cube / pick-place / arrange-boxes /
│   │                        #   backup-policy / safe）和真机环境 (franka_real_env)
│   ├── robot_servers/       # RT PC 上跑的 Flask server（franka_server.py + gripper）
│   ├── controllers/         # OSC 控制器（仿真）
│   ├── fault_injection.py   # EncoderBiasInjector
│   ├── rl/                  # Actor + Learner + Replay Buffer
│   ├── policies/sac/        # SAC 网络与训练逻辑
│   ├── processor/           # 观测/动作 transition 处理器
│   ├── teleoperators/       # SpaceMouse + 键盘输入
│   ├── cameras/             # RealSense 相机
│   └── trajectory_executor.py
├── scripts/                 # 训练/评估/真机自检脚本
├── demos/                   # 早期 sim 环境的硬编码 pick-and-place 演示（见 demos/README.md）
├── configs/                 # 训练 JSON + bias YAML
├── docs/                    # 详细文档（见下方索引）
├── patches/                 # 对外部 catkin 包的改动（见 patches/README.md）
├── assets/                  # MuJoCo 场景/模型
└── checkpoints/             # 预训练策略 checkpoint（不完全入 git）
```

## 快速开始（仿真）

```bash
conda activate lerobot

# 硬编码 pick-and-place 验证仿真环境
python demos/demo_continuous_loop.py --episodes 10

# 训练（带编码器偏差的 RLPD）
bash scripts/real/train_hil_sac.sh        # 具体任务/配置见脚本内部
```

## 快速开始（真机）

1. RT PC：启动 Flask server
   ```bash
   ~/start_franka_server.sh   # 内部 cd 到本仓库并跑 python -m frrl.robot_servers.franka_server
   ```
   启动流程/FCI/网络配置见 [`docs/rt_pc_runbook.md`](docs/rt_pc_runbook.md)。

2. GPU 机：构造 `FrankaRealEnv` 跑训练或评估
   ```python
   from frrl.envs.real_config import FrankaRealConfig
   from frrl.envs.real import FrankaRealEnv
   from frrl.fault_injection import EncoderBiasConfig

   cfg = FrankaRealConfig(
       server_url="http://192.168.100.1:5000/",
       encoder_bias_config=EncoderBiasConfig(
           enable=True, target_joints=[0],
           bias_mode="random_uniform", bias_range=(0.0, 0.25),
       ),
   )
   env = FrankaRealEnv(cfg)
   obs, _ = env.reset()   # 会自动 reset + 注入 bias
   ```

3. 真机编码器偏差注入的完整细节见
   [`docs/fault_injection_realhw.md`](docs/fault_injection_realhw.md)。

## 核心思路

编码器校准偏差 → 一个故障源，两处影响：

1. **控制偏差**：控制器基于错误的关节角度计算 Jacobian/FK → 力矩错 → 执行偏差
2. **感知偏差**：FK(q_measured) 得到错误的末端位姿 → 观测错

仿真里通过修改 `q_measured = q_true + bias` 注入；真机通过 **B+D 双注入点**
（C++ 阻抗控制器 + franka_server HTTP）实现等价效果。

## 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/project_progress.md`](docs/project_progress.md) | 实验结果、待办、当前聚焦 |
| [`docs/real_robot_deployment_plan.md`](docs/real_robot_deployment_plan.md) | 两机（RT PC + GPU）整体部署方案（历史规划） |
| [`docs/rt_pc_runbook.md`](docs/rt_pc_runbook.md) | RT PC 从开机到 server 启动的完整流程 |
| [`docs/fault_injection_architecture.md`](docs/fault_injection_architecture.md) | 为什么选 B+D 注入点（设计决策） |
| [`docs/fault_injection_realhw.md`](docs/fault_injection_realhw.md) | 真机 B+D 实现、编译、验证、训练配置 |
| [`docs/fault_simulation_design.md`](docs/fault_simulation_design.md) | 仿真侧 bias 注入设计 |
| [`docs/data_flow.md`](docs/data_flow.md) | 观测/动作/reward 数据流 |
| [`docs/rl_reward.md`](docs/rl_reward.md) | Reward 设计 |
| [`docs/backup_policy.md`](docs/backup_policy.md) | Backup Policy + 人手避让（阶段五） |
| [`docs/spacemouse_teleop.md`](docs/spacemouse_teleop.md) | SpaceMouse 遥操作部署 |
| [`patches/README.md`](patches/README.md) | 对外部 catkin 包 serl_franka_controllers 的改动 |

## 研究进度

- ✅ 仿真环境 + 编码器偏差注入（Joint 4 固定/随机）
- ✅ RLPD + HIL 训练框架（24D 观测 + backup policy）
- ✅ 真机部署：RT PC 控制栈 + GPU 训练机 + franka_server HTTP
- ✅ 真机 B+D 编码器偏差注入（2026-04-15 端到端验证）
- ⏳ 真机训练首跑（工作空间标定、demo 采集、首轮训练）
- ⏳ 评估与论文
