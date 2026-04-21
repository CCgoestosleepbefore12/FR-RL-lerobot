# Backup Policy S1 V2 — 50k steps (旧几何训练版)

## 用途
Franka Panda 真机实验用的 backup policy checkpoint。

## 训练信息
- **环境**：`gym_frrl/PandaBackupPolicyS1V2-v0`（V2 防作弊：腕+手单球避障 r=10cm + 旋转预算 ≤0.5rad）
- **Config**：`configs/train_hil_sac_backup_s1_v2.json`（port 50054, seed 1003, 300k steps 计划）
- **Step**：50,000 / 300,000（~17%，训练仍在继续）
- **训练起始 commit**：几何修复后 V2 重启（18:51 run）

## 训练时的几何配置（旧）
- `max_displacement = 0.20m`
- `TCP_INIT_X=(0.30,0.45) / Y=(-0.22,0.00) / Z=(0.15,0.30)`（为预算收紧）
- `enforce_cartesian_bounds = True`（工作空间 clamp 激活）
- `ARM_SPAWN_DIST = (0.21, 0.30)` / `ARM_COLLISION_DIST = 0.135m`

> 注：commit `a5df0b8` 之后 S1V2 env 已改为"直线退让几何"（0.30 预算 +
> `enforce_cartesian_bounds=False` + TCP_INIT 放宽）。**本 ckpt 是在旧几何下
> 训练得到的**，策略学到的是"尊重工作空间边界 + 小幅退让"，而不是沿
> -hand_dir 的直线退让。

## 真机部署注意
1. 真机本身带工作空间 safety clamp，本 ckpt 学到的"尊重边界"行为与真机约束一致
2. 观测维度：84D = (18 robot + 10 obstacle) × 3 stack
3. 动作维度：6D（3D pos delta + 3D rot delta），经 OSC 跟踪
4. Episode 长度：20 步（100Hz 控制 → 每 episode 2s）

## Eval 基准（sim，新几何 env 50 ep）
- 满存活率 82% / 平均奖励 +7.98 / 平均最近距离 0.196m
- 失败分布：rotation 10% / collision 6% / displacement 2%

## 加载示例
```python
from frrl.policies.sac.modeling_sac import SACPolicy
from frrl.configs.policies import PreTrainedConfig

ckpt = "checkpoints/backup_policy_s1_v2_50k"
cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.pretrained_path = ckpt
cfg.device = "cuda"
policy = SACPolicy.from_pretrained(ckpt, config=cfg)
policy.eval()
```
