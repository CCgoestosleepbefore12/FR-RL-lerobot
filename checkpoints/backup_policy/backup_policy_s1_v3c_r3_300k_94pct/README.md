# Backup Policy S1 V3c r3 — 300k (93.5% min cross-eval)

V3c 单参数 ablation 最终版（r3）训完版本。**论文 ablation 对照**，性能略低于 V3b 但可作 sim2real 几何对齐 fallback。

## 核心设计：和 V3b 唯一差异 = `tracking_target`

| 参数 | V3b 300k_95pct | **V3c r3 300k** |
|---|---|---|
| `tracking_target` | "tcp" | **"arm_center"** ★ 唯一差异 |
| 追的目标 | pinch_site (TCP) | panda_hand body (= flange = collision 球心) |
| `D_TIGHT` (effective) | 0.08m (距 TCP) | 0.08m (距 arm_center) |
| `HAND_SPEED_RANGE` | (0.015, 0.030) | (0.015, 0.030) |
| `max_displacement` | 0.50 | 0.50 |
| `max_episode_steps` | 20 | 20 |
| `online_steps` | 300k | 300k |
| `utd_ratio` | 4 | 4 |

## V3c 演进史（三版）

| 版本 | tracking_target | D_TIGHT_ARM | hand_speed | episode | min cross-eval | 结论 |
|---|---|---|---|---|---|---|
| r1 (dwell-based) | arm_center | 0.23 (dwell 可达) | (0.015, 0.030) | 20 | **65%** | benign 训练 → brittle |
| r2 (加压) | arm_center | 0.08 (无 dwell) | (0.020, 0.040) | 25 | **62%** | 太严 → 不收敛 |
| **r3 (本设计)** | **arm_center** | 0.08 (= V3 等价) | (0.015, 0.030) | 20 | **93.5%** | 接近 V3b，单参数 ablation |

## Cross-eval 结果（200 ep deterministic）

| ckpt \ env | V3 env (TCP) | V3c env (arm_center) | min |
|---|---|---|---|
| **V3b 300k_95pct** ⭐ 真机首选 | 95.0% | 96.0% | **95.0%** |
| **V3c r3 300k** | **96.5%** | 93.5% (own) | **93.5%** |

详细分布：

| 指标 | V3 env | V3c env (own) |
|---|---|---|
| 满存活率 | 96.5% (193/200) | 93.5% (187/200) |
| 平均累计奖励 | +12.33 ± 3.87 | +11.48 ± 5.43 |
| 平均最近距离 | 0.272m | 0.271m |
| 平均终止位移 | 0.363m | 0.377m |
| hand_collision | 3.5% | 5.0% |
| excessive_displacement | 2.0% | 4.0% |

## 关键观察

1. **V3c r3 在 V3 env 反超 V3b**（96.5% vs 95.0%）——arm_center tracking 训练的 policy 在 TCP-tracking eval 上略好
2. **V3c r3 在 V3c env 略输 V3b**（93.5% vs 96.0%）——反直觉但稳定结论
3. **V3b 以 1.5% 优势保持真机首选**（min 95.0% vs 93.5%）

## 论文 ablation 结论

V3c 三版完整结果证明：
- tracking_target 选择对 policy 质量影响**微小**（~1-2%）
- sim 训练 target (TCP/flange) 不必跟真机 FSM 反推点 1:1 对齐
- V3b TCP-tracking 的"几何错位"可由 FSM 反推 `hand_body_equiv = TCP - 0.1034 × gripper_z` 完美弥补
- **训练分布的"balanced harshness"** 比 sim2real 几何对齐更重要：r1 太 benign brittle，r2 太严不收敛，r3/V3b 适中收敛好

## 真机部署

```bash
# 默认仍用 V3b（min 高 1.5%）
python scripts/real/deploy_backup_policy.py --ckpt-version v3

# Fallback / ablation 验证：用 V3c r3
python scripts/real/deploy_backup_policy.py \
  --ckpt-version v3 \
  --checkpoint checkpoints/backup_policy_s1_v3c_r3_300k_94pct
```

D_SAFE=0.40 / D_CLEAR=0.45 与 V3b 共用（sim ARM_SPAWN_DIST_V3 一致）。

## 仓内 ckpt 关系

| 路径 | 用途 |
|---|---|
| `backup_policy_s1_v2_newgeom_145k/` | V2 单球（sim 100% 但真机肘撞）|
| `backup_policy_s1_v3_300k_71pct/` | V3 baseline (max_disp=0.40)，max_disp ablation 对照 |
| `backup_policy_s1_v3b_115k_90pct/` | V3 Path A 中段，训练曲线记录 |
| **`backup_policy_s1_v3b_300k_95pct/`** | **V3b 真机首选** ⭐ |
| **`backup_policy_s1_v3c_r3_300k_94pct/`** | **V3c r3 单参数 ablation** (本目录) |
