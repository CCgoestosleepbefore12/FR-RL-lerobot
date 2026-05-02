# Backup Policy S1 V3 — 300k 全臂避障 baseline

V3 全臂避障 (5 球 link3/4/5/6/hand) + obstacle r=0.10 + saturating proximity reward 的首训 baseline。

## 训练参数（与 `scripts/configs/train_hil_sac_backup_s1_v3.json` 对齐）

| 项 | 值 |
|---|---|
| 环境 | `gym_frrl/PandaBackupPolicyS1V3-v0` |
| 训练 step | 300,000 (learner step) |
| 训练时长 | 6.5h (09:36 → 16:04, 2026-04-26) |
| utd_ratio | 4 |
| seed | 1004 |
| Port | 50055 |
| obstacle radius | 0.10 (对齐真机 hand bbox) |
| `ARM_SPAWN_DIST_V3` | (0.30, 0.40) |
| `HAND_SPEED_RANGE_V3` | (0.015, 0.030) m/step |
| **`max_displacement`** | **0.40** |
| `PROXIMITY_REWARD_MAX` / `PROXIMITY_SAFE_DIST` | 0.20 / 0.10 |

## Eval 结果（200 episodes，deterministic policy）

| 指标 | 值 |
|---|---|
| 存活率 (≥20 步) | **71.5%** (143/200) |
| 平均累计奖励 | +8.02 ± 8.04 |
| 平均最近距离 | 0.247m (center-to-center) |
| 平均终止位移 | 0.324m ± 0.063 |

终止原因分布：
- survived: 66.5%
- hand_collision: 17.5% ← 多球检测让 link3/4/5 几何避让更难
- excessive_displacement: 14.5% ← `max_disp=0.40` 偏紧，policy 撞顶
- zone_c_intrusion / block_dropped: 1.5%

## 已知 Limitation

`max_displacement=0.40` 对 V3 几何（hand 加速到 30cm/s + 多球检测）偏紧——policy 想退更远但被罚，14.5% episode 因此失败。

**Path A 重训计划**：`max_displacement=0.40 → 0.50`，预期把 14.5% 位移失败大半救回，目标存活率 → 80%。下一版 ckpt：`backup_policy_s1_v3b_*`。

## 真机部署

```bash
python scripts/real/deploy_backup_policy.py \
  --ckpt-version v3 \
  --checkpoint checkpoints/backup_policy_s1_v3_300k_71pct
```

D_SAFE=0.40 / D_CLEAR=0.45 自动配（与 V3 sim 训练 spawn 上限对齐）。

## 与 V2 145k 的对比

| 指标 | V2 (单球, 145k) | V3 (5 球, 300k) |
|---|---|---|
| Sim 存活率 | 100% | 71.5% |
| 真机肘/前臂撞 | **存在**（V2 collision 仅查 panda_hand）| 根除（5 球全臂检测）|
| 平均奖励 | +10.62 ± 0.77 | +8.02 ± 8.04 |
| 平均终止位移 | 0.092m | 0.324m |

V3 sim 数字降低，但真机失效模式被根除——质的改进。
