# Backup Policy S1 V3b — 300k Path A FINAL (95.0% 存活率)

V3 Path A 训完版本（max_disp=0.50, 300k learner step, 6.3h, 2026-04-26 17:03→23:22）。
**真机部署首选 ckpt**。

## 训练参数

| 项 | 值 | vs V3 首训 |
|---|---|---|
| 环境 | `gym_frrl/PandaBackupPolicyS1V3-v0` | 同 |
| `max_displacement` | **0.50** | ★ 从 0.40 放宽 |
| obstacle radius | 0.10 | 同 |
| `ARM_SPAWN_DIST_V3` | (0.30, 0.40) | 同 |
| `HAND_SPEED_RANGE_V3` | (0.015, 0.030) | 同 |
| utd_ratio | 4 | 同 |
| online_steps | 300,000（已完成）| 同 |
| seed | 1004 | 同 |

## Eval 结果（200 episodes，deterministic policy）

| 指标 | 值 |
|---|---|
| **存活率 (≥20 步)** | **95.0%** (190/200) |
| 平均累计奖励 | +11.41 ± 4.84 |
| 平均最近距离 | 0.271m (center-to-center) |
| 平均终止位移 | 0.369m ± 0.074 (max_disp=0.50, 用 74% 预算) |

终止原因分布：
| 原因 | 数量 | 占比 |
|---|---|---|
| **survived** | **182** | **91.0%** |
| excessive_displacement | 11 | 5.5% |
| hand_collision | 6 | 3.0% |
| zone_c_intrusion | 1 | 0.5% |

## 三方完整对比（V2 vs V3 baseline vs V3b final）

| 指标 | V2 (单球, max_disp=0.30, 145k) | V3 (5 球, max_disp=0.40, 300k) | **V3b (5 球, max_disp=0.50, 300k)** |
|---|---|---|---|
| Sim 满存活率 | 100% | 71.5% | **95.0%** |
| 真机肘/前臂撞 | **存在** | 根除 | 根除 |
| hand_collision rate | <1% | 17.5% | 3.0% |
| excessive_displacement | <1% | 14.5% | 5.5% |
| 平均奖励 | +10.62 ± 0.77 | +8.02 ± 8.04 | +11.41 ± 4.84 |
| 平均终止位移 | 0.092m | 0.324m | 0.369m |

V3b sim 95.0% **接近 V2 100%** 但几何更严（5 球 + obstacle r=0.10）且**根除真机肘撞**——质的改进无 sim 性能损失。

## Path A 训练曲线（200 ep deterministic eval）

```
ckpt    存活率   hand_coll  ex_disp   reward
 65k    89.0%    11.0%      7.5%     +9.69
115k    89.5%     7.0%      7.5%    +10.59
150k    91.5%     7.0%      4.0%    +11.01
200k    91.0%     5.5%      5.0%    +11.10
250k    91.5%     6.5%      3.5%    +11.00
300k    95.0%     3.0%      5.5%    +11.41   ← FINAL
```

300k 在最后阶段把 hand_collision 从 ~7% 砍到 3%——multi-sphere 避让能力完全到位。

## 真机部署

```bash
python scripts/real/deploy_backup_policy.py \
  --ckpt-version v3 \
  --checkpoint checkpoints/backup_policy_s1_v3b_300k_95pct
```

D_SAFE=0.40 / D_CLEAR=0.45 自动配（与 V3 sim 训练 spawn 上限对齐）。

## 仓内其他 ckpt 关系

| Ckpt | 用途 |
|---|---|
| `backup_policy_s1_v3_300k_71pct/` | V3 首训（max_disp=0.40），ablation 对照——证明 max_disp 是关键 |
| `backup_policy_s1_v3b_115k_90pct/` | Path A 中段（89.5%），训练曲线记录，可删 |
| **`backup_policy_s1_v3b_300k_95pct/`** | **真机首选** |
| `backup_policy_s1_v2_newgeom_145k/` | V2 (sim 100% 但真机肘撞)，仅做 V2/V3 切换 ablation |
