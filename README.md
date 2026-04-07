# FR-RL: Fault-Resilient Reinforcement Learning

模拟工业机器人编码器校准偏差，用 RL 训练鲁棒操作策略。

## 项目结构

```
FR-RL/
└── mujoco_sim/                    # MuJoCo 仿真环境
    ├── fault_injection.py         # 编码器偏差注入
    ├── trajectory_executor.py     # 轨迹执行器
    ├── envs/                      # Gymnasium 环境
    ├── robot_control/             # 机器人控制（OSC, IK, FK）
    ├── demos/                     # 演示程序
    ├── configs/                   # 偏差配置
    └── assets/                    # MuJoCo 场景/模型
```

## 快速开始

```bash
conda activate lerobot
cd FR-RL/mujoco_sim

# 无故障 baseline
python demos/demo_continuous_loop.py --episodes 10

# 编码器偏差训练
python demos/demo_continuous_loop.py --episodes 100 \
    --enable-encoder-bias \
    --bias-config configs/encoder_bias_joint4_random.yaml
```

## 核心思路

编码器校准偏差 → 一个故障源，两个影响：
1. **控制偏差**: OSC 基于错误的关节角度计算力矩
2. **感知偏差**: FK 得到错误的末端位姿

详见 [mujoco_sim/README.md](mujoco_sim/README.md)

## 研究进度

- 阶段 1-3: 仿真环境 + 轨迹跟踪 + 编码器偏差注入 ✅
- 阶段 4: HIL-SERL 集成 ⏳
- 阶段 5: 评估和论文 ⏳
