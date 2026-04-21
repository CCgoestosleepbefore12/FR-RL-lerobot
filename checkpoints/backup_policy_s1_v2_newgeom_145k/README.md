# Backup Policy S1 V2 — 新几何 145k steps（真机部署首选，新版）

## 用途
Franka Panda 真机实验用的 backup policy checkpoint。**推荐使用此 ckpt**，它在新几何下训练至 learner 145k 步，对应 actor 训练曲线的新峰值段（190-195k，98.4% 在线存活率）。

## 训练信息
- **环境**：`gym_frrl/PandaBackupPolicyS1V2-v0`（V2 防作弊：腕+手单球避障 r=10cm）
- **Config**：`configs/train_hil_sac_backup_s1_v2.json`（port 50054, seed 1003, 300k steps 计划）
- **Learner Step**：145,000 / 300,000（~48%，训练仍在继续）
- **对应 Actor Step**：~197,000（actor/learner 比例 ≈ 1.36）
- **训练起始 commit**：`a5df0b8` 之后新几何版（21:29 run，后续由 23:06:55 run 继续）
- **训练时间**：2026-04-21 23:06 – 2026-04-22 05:29（~6.5h，145k learner steps）

## 训练时的几何配置（新，直线退让）
- `max_displacement = 0.30m`（比旧版 0.20 宽松 50%，允许更大幅度的直线退让）
- `enforce_cartesian_bounds = False`（关闭工作空间 clamp，让策略自由沿 -hand_dir 退让）
- `TCP_INIT_X=(0.30,0.55) / Y=(-0.30,0.10) / Z=(0.15,0.40)`（初始范围放宽以适配新预算）
- `ARM_SPAWN_DIST = (0.21, 0.30)` / `ARM_COLLISION_DIST = 0.135m`

## 旋转预算（仍是旧版，训练启动时尚未放松）
- `MAX_ROTATION = 0.5 rad`（严格预算，非常保守）
- `ROTATION_COEFF = 0.5`（预算内就有线性惩罚）

> 注：23:06:55 run 启动时 Python 模块常量仍是上述旧值。之后代码把默认值改成
> `MAX_ROTATION=π / ROTATION_COEFF=0.2`（取消预算，仅保留软惩罚），以解决"手从上方来
> 无法躲避"的问题 —— 这个改动对**下一次训练**生效，本 ckpt 不受影响。

## Eval 基准（sim，新几何 env 50 ep，20 steps/episode）
- 满存活率 **100.0%**（50/50）⭐
- 平均累计奖励 **+10.62 ± 0.77**（方差最低）
- 平均最近距离 0.204m
- 平均终止位移 0.186m
- 失败分布：无失败

在相邻 ckpt 中 145k 为最佳：140k (98%/±2.21) / 145k (**100%/±0.77**) / 150k (100%/±0.95) / 155k (98%/±2.40)。

## 与旧 ckpt 对照
| Ckpt | 几何 | Learner Steps | 存活率 | 奖励方差 | 真机推荐 |
| --- | --- | --- | --- | --- | --- |
| `backup_policy_s1_v2_50k` | 旧（0.20 预算 + 工作空间 on） | 50k | 82% | — | 备选 |
| `backup_policy_s1_v2_newgeom_35k` | 新（0.30 预算 + 工作空间 off） | 35k | 94% | — | 旧版首选 |
| `backup_policy_s1_v2_newgeom_145k` | 新（同上） | **145k** | **100%** | **±0.77** | **新版首选** ⭐ |

## 真机部署注意
1. 真机本身带工作空间 safety clamp，会裁剪越界动作 —— 但由于本 ckpt 学到"沿手反方向退让"，
   正常避障时不会触碰边界，实际动作被 clamp 的概率低
2. 观测维度：84D = (18 robot + 10 obstacle) × 3 stack
3. 动作维度：6D（3D pos delta + 3D rot delta），经 OSC 跟踪
4. Episode 长度：20 步（100Hz 控制 → 每 episode 2s）

## 加载示例
```python
from frrl.policies.sac.modeling_sac import SACPolicy
from frrl.configs.policies import PreTrainedConfig

ckpt = "checkpoints/backup_policy_s1_v2_newgeom_145k"
cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.pretrained_path = ckpt
cfg.device = "cuda"
policy = SACPolicy.from_pretrained(ckpt, config=cfg)
policy.eval()
```
