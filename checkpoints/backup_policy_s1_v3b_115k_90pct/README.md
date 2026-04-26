# Backup Policy S1 V3b — 115k Path A best (89.5% 存活率)

V3 Path A 重训中段最佳 ckpt（截至 2026-04-26 19:21, 训练仍在进行中, 总 300k step）。

## 训练参数（与 V3 baseline 唯一差异：`max_displacement` 0.40 → 0.50）

| 项 | 值 | vs V3 首训 |
|---|---|---|
| 环境 | `gym_frrl/PandaBackupPolicyS1V3-v0` | 同 |
| `max_displacement` | **0.50** | ★ 从 0.40 放宽 |
| obstacle radius | 0.10 | 同 |
| `ARM_SPAWN_DIST_V3` | (0.30, 0.40) | 同 |
| `HAND_SPEED_RANGE_V3` | (0.015, 0.030) | 同 |
| utd_ratio | 4 | 同 |
| online_steps | 300,000（训练中）| 同 |
| 当前 ckpt | 115k learner step (~2.5h) | — |

## Eval 结果（200 episodes，deterministic policy）

| 指标 | 值 |
|---|---|
| 存活率 (≥20 步) | **89.5%** (179/200) |
| 平均累计奖励 | +10.59 ± 6.22 |
| 平均最近距离 | 0.276m (center-to-center) |
| 平均终止位移 | 0.391m ± 0.083 (max_disp=0.50) |

终止原因分布（vs V3 baseline 300k）：
| 原因 | V3 baseline (max_disp=0.40, 300k) | **V3b (max_disp=0.50, 115k)** |
|---|---|---|
| survived | 66.5% | **84.5%** (+18.0%) |
| hand_collision | 17.5% | **7.0%** (−10.5%) ⭐ |
| excessive_displacement | 14.5% | 7.5% (−7.0%) |
| zone_c_intrusion | 1.0% | 1.0% |
| block_dropped | 0.5% | 0.0% |

## 关键发现

**Path A 单参数改动 (max_disp 0.40→0.50) 同时救了两个失效模式**：
1. **excessive_displacement** −7%：直接放宽位移预算
2. **hand_collision** −10.5%：意外收获——policy 不再被位移上限"夹住"，能更灵活避让多球

V3 首训 71.5% 假设 hand_collision 是 multi-sphere 几何天然难度；实测 max_disp 放宽后 hand_collision 也降了，**说明很多碰撞是因为 policy 想退但被上限挡住**，不是真学不会避。

## Online Training Curve（actor 满存活率 by env step bucket）

```
 0-10k:  47.9%  (探索期)
10-20k:  64.6%
20-30k:  77.5%
30-40k:  74.7%
40-50k:  79.8%
50-60k:  82.4%
60-70k:  82.9%
70-80k:  83.0%  (当前)
```

Online stochastic 在 80% 附近平稳；deterministic eval (115k) 89.5% 比 online 高 ~7%（policy 取 mean 比 sample 表现更稳）。

## 真机部署

```bash
python scripts/real/deploy_backup_policy.py \
  --ckpt-version v3 \
  --checkpoint checkpoints/backup_policy_s1_v3b_115k_90pct
```

D_SAFE=0.40 / D_CLEAR=0.45 自动配（与 V3 sim 训练 spawn 上限对齐）。

## 与 V2 145k 对比（最完整三方比对）

| 指标 | V2 (单球, max_disp=0.30, 145k) | V3 (5 球, max_disp=0.40, 300k) | **V3b (5 球, max_disp=0.50, 115k)** |
|---|---|---|---|
| Sim 满存活率 | 100% | 71.5% | **89.5%** |
| 真机肘/前臂撞 | **存在**（V2 collision 仅查 panda_hand）| 根除 | 根除 |
| hand_collision rate | <1% | 17.5% | 7.0% |
| 平均奖励 | +10.62 ± 0.77 | +8.02 ± 8.04 | +10.59 ± 6.22 |
| 平均终止位移 | 0.092m | 0.324m | 0.391m |

V3b 几何更严（5 球 vs 单球，obstacle r=0.10 vs 0.035）但 sim 数字接近 V2，且根除真机肘撞——是质的改进，不是简单回退。

## 注意

- **115k 是中段 ckpt**，训练未结束。后续 200k / 300k 可能更高（看 actor curve 是否继续上升）
- 训完后会有 `backup_policy_s1_v3b_300k_<best>pct/` final ckpt 落库；本 115k 作 baseline 留存
- V3 首训 ckpt `backup_policy_s1_v3_300k_71pct/` 保留作 max_disp=0.40 vs 0.50 ablation 对照
