# Pickandplace Task — 真机命令清单

把勺子/叉子从盘子上抓起来移到旁边放下，多阶段：抓 → 横向移动 → 放下。

- 数据目录：`data/{no_bias|with_bias}/pickandplace/`
- ckpt 目录：`checkpoints/{no_bias|with_bias}/pickandplace/`
- Config：`scripts/configs/train_hil_sac_pickandplace_real.json`
- 部署：`scripts/real/deploy_pickandplace_with_backup.py`
- bias_range 默认 ±0.1 rad（task factory 内置；细长餐具 yaw 对齐对 bias 敏感）

---

## 上手清单（pickandplace 真机从零到 iterN）

```bash
# 1. 收 50 条 cold-start demo（操作员示范 pick → 横移 → place）
python scripts/real/collect_demo_task_policy.py --task pickandplace -n 50

# 2. cold-start BC
python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_pickandplace_real.json \
    --demo-paths "data/no_bias/pickandplace/pickandplace_demos_*.pkl" \
    --steps 20000 \
    --output-dir checkpoints/no_bias/pickandplace/pickandplace_bc_$(date +%Y%m%d_%H%M%S)

# 3. 部署测一下，记新 ckpt 路径
python scripts/real/deploy_bc_inference.py \
    --ckpt <step2 出的路径>/checkpoints/020000/pretrained_model \
    --task pickandplace --lift-threshold 0.04

# 4. 收 30 条 dagger iter1（基于 cold-start ckpt）
python scripts/real/deploy_bc_with_dagger.py \
    --ckpt <step2 出的路径>/checkpoints/020000/pretrained_model \
    --task pickandplace --iter 1 -n 30

# 5. 训 iter1（demos + dagger 介入帧合并）
python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_pickandplace_real.json \
    --demo-paths "data/no_bias/pickandplace/*.pkl" \
    --steps 20000 --intervention-only \
    --output-dir checkpoints/no_bias/pickandplace/pickandplace_bc_iter1_$(date +%Y%m%d_%H%M%S)

# 6. 重复 3-5 拿 iter2、iter3...
```

---

## 上手清单 — With bias 版本

```bash
# 1. 收 50 条 cold-start demo（带 bias）
python scripts/real/collect_demo_task_policy.py \
    --task pickandplace -n 50 \
    --bias --bias-range -0.2 0.2

# 2. cold-start BC
python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_pickandplace_real.json \
    --demo-paths "data/with_bias/pickandplace/pickandplace_demos_*.pkl" \
    --steps 20000 \
    --output-dir checkpoints/with_bias/pickandplace/pickandplace_bc_$(date +%Y%m%d_%H%M%S)

# 3. 部署测一下（带 bias，跟训练分布一致）
python scripts/real/deploy_bc_inference.py \
    --ckpt <step2 出的路径>/checkpoints/020000/pretrained_model \
    --task pickandplace --lift-threshold 0.04 \
    --bias --bias-range -0.2 0.2 --bias-monitor

# 4. 收 30 条 dagger iter1（带 bias）
python scripts/real/deploy_bc_with_dagger.py \
    --ckpt <step2 出的路径>/checkpoints/020000/pretrained_model \
    --task pickandplace --iter 1 -n 30 \
    --bias --bias-range -0.2 0.2

# 5. 训 iter1（demos + dagger 介入帧合并）
python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_pickandplace_real.json \
    --demo-paths "data/with_bias/pickandplace/*.pkl" \
    --steps 20000 --intervention-only \
    --output-dir checkpoints/with_bias/pickandplace/pickandplace_bc_iter1_$(date +%Y%m%d_%H%M%S)

# 6. 重复 3-5 拿 iter2、iter3...
```

---

## 联合 backup 部署

```bash
# with bias + monitor
python scripts/real/deploy_pickandplace_with_backup.py \
    --bc-ckpt checkpoints/with_bias/pickandplace/<your_ckpt>/checkpoints/020000/pretrained_model \
    --ckpt-version v3 --lift-threshold 0.04 \
    --bias --bias-range -0.2 0.2 --bias-monitor

# no bias
python scripts/real/deploy_pickandplace_with_backup.py \
    --bc-ckpt checkpoints/no_bias/pickandplace/<your_ckpt>/checkpoints/020000/pretrained_model \
    --ckpt-version v3 --lift-threshold 0.04
```

---

## Pickandplace 特殊事项

- **多阶段任务**：抓（pick）→ 横向 transit → 放下（place）。比 pickup 长一倍，`max_episode_length=200`（20s @ 10Hz），`control_time_s=20.0`。
- **gripper_locked="none"**：BC 学开/关 gripper 时机（pick 时关，place 时开）。`num_discrete_actions=3`，跟 pickup 一致——actor 输出 6D continuous + 1D discrete gripper head（CE loss）。
- **bias_range 默认 ±0.1**：餐具是细长形（勺子/叉子），yaw 对齐关键。bias 在 J1 → 末端 xyz 偏移 ~5cm @ ±0.1 rad，餐具放置精度本就要求高，所以 task factory 默认偏小。**部署 / 收数时保持 train-deploy 一致即可**。
- **reset_pose**：悬停在盘子 + 放置区中心上方（`[0.4608, -0.0935, 0.2575]`），跟 wipe 共享同一盘子位置。
- **--lift-threshold 用于自动 success 判定**：跟 pickup 一致，pickandplace 也用 z 抬升 + gripper 状态判定 episode 结束。
