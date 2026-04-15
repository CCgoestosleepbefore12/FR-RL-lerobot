# demos/ — Hardcoded pick-and-place validation scripts

Four **非-RL**、纯脚本化的 `PandaPickPlaceEnv` 演示程序。它们在项目早期用于验证
仿真环境 + OSC 控制器 + 轨迹执行器的基础链路是否正常，目前主要作为：

1. **环境冒烟测试** —— 改了 `frrl/envs/panda_pick_place_env.py` 或
   `frrl/controllers/opspace.py` 之后，快速验证仿真还能跑、抓取逻辑还正常
2. **对比基线** —— 需要一个"非 RL 但能完成任务"的对照时使用

训练用的是 `scripts/train_hil_sac.sh`，评估用的是 `scripts/eval_policy.py` /
`scripts/eval_bias_curve.py` / `scripts/eval_backup_policy.py`，**不要**用这些
demo 做正式评估。

---

## 脚本对比

| 脚本 | 行为 | Episode 数 | 支持 bias | 何时用 |
|---|---|---|---|---|
| `demo_hardcoded_pick_place.py` | 硬编码阶段分解，逐步打印 stage | 1 | ❌ | 调 OSC 参数时想看每一步的 TCP 位姿 |
| `demo_optimized_trajectory.py` | 手调了停留时间 / 下降高度 / 速度 | 1 | ❌ | 验证 pick-and-place 在当前 env 参数下稳定 |
| `demo_repeat_trajectory.py` | 跑 N 次相同轨迹，统计抓取成功率 | N（默认 10） | ❌ | 评估硬编码策略的稳定性上限 |
| `demo_continuous_loop.py` | 连续循环直到 Ctrl+C，支持 YAML bias config | 无限 | ✅ | 带编码器偏差看硬编码策略失败率（对照实验） |

---

## 运行

所有脚本必须从**仓库根目录**运行（`cd ~/FR-RL-lerobot`），因为它们用相对路径
加载 `configs/`、`assets/`、等等：

```bash
# 单次硬编码演示（viewer 打开）
python demos/demo_hardcoded_pick_place.py

# 连续循环 + 偏差注入
python demos/demo_continuous_loop.py --episodes 100 \
    --enable-encoder-bias \
    --bias-config configs/encoder_bias_joint4_random.yaml
```

---

## 取舍说明

4 个脚本功能高度重叠，理论上可以合成 1 个参数化脚本。暂时保留全部是因为：

- 它们都很短（< 400 行），维护成本低
- 每个脚本内部的"硬编码轨迹"都是独立调参的结果，合并后反而难读
- 没人实际反复用它们，合并收益不大

如果未来修改 `PandaPickPlaceEnv` 的观测/动作格式，**最先会坏的就是这些**。
改完记得至少跑一次 `demo_hardcoded_pick_place.py` 确认还能完成任务。
