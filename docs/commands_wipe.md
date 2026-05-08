# Wipe Task — 真机命令清单

擦盘子任务：海绵全程被夹住（gripper_locked="closed"），BC 学连续涂抹轨迹。

- 数据目录：`data/{no_bias|with_bias}/wipe/`
- ckpt 目录：`checkpoints/{no_bias|with_bias}/wipe/`
- Config：`scripts/configs/train_hil_sac_wipe_real.json`
- 部署：`scripts/real/deploy_wipe_with_backup.py`
- bias_range 默认 ±0.2 rad（task factory 内置；wipe 涂抹幅度大对 bias 较敏感）

---

## 上手清单（wipe 真机从零到 iterN）

```bash
# 1. 收 50 条 cold-start demo（操作员全程 SpaceMouse 涂抹示范）
python scripts/real/collect_demo_task_policy.py --task wipe -n 50

# 2. cold-start BC
python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_wipe_real.json \
    --demo-paths "data/no_bias/wipe/wipe_demos_*.pkl" \
    --steps 20000 \
    --output-dir checkpoints/no_bias/wipe/wipe_bc_$(date +%Y%m%d_%H%M%S)

# 3. 部署测一下，记新 ckpt 路径
python scripts/real/deploy_bc_inference.py \
    --ckpt <step2 出的路径>/checkpoints/020000/pretrained_model \
    --task wipe

# 4. 收 30 条 dagger iter1（基于 cold-start ckpt）
python scripts/real/deploy_bc_with_dagger.py \
    --ckpt <step2 出的路径>/checkpoints/020000/pretrained_model \
    --task wipe --iter 1 -n 30

# 5. 训 iter1（demos + dagger 介入帧合并）
python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_wipe_real.json \
    --demo-paths "data/no_bias/wipe/*.pkl" \
    --steps 20000 --intervention-only \
    --output-dir checkpoints/no_bias/wipe/wipe_bc_iter1_$(date +%Y%m%d_%H%M%S)

# 6. 重复 3-5 拿 iter2、iter3...
```

---

## 上手清单 — With bias 版本

```bash
# 1. 收 50 条 cold-start demo（带 bias）
python scripts/real/collect_demo_task_policy.py \
    --task wipe -n 50 \
    --bias --bias-range -0.2 0.2

# 2. cold-start BC
python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_wipe_real.json \
    --demo-paths "data/with_bias/wipe/wipe_demos_*.pkl" \
    --steps 20000 \
    --output-dir checkpoints/with_bias/wipe/wipe_bc_$(date +%Y%m%d_%H%M%S)

# 3. 部署测一下（带 bias，跟训练分布一致）
python scripts/real/deploy_bc_inference.py \
    --ckpt <step2 出的路径>/checkpoints/020000/pretrained_model \
    --task wipe \
    --bias --bias-range -0.2 0.2 --bias-monitor

# 4. 收 30 条 dagger iter1（带 bias）
python scripts/real/deploy_bc_with_dagger.py \
    --ckpt <step2 出的路径>/checkpoints/020000/pretrained_model \
    --task wipe --iter 1 -n 30 \
    --bias --bias-range -0.2 0.2

# 5. 训 iter1（demos + dagger 介入帧合并）
python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_wipe_real.json \
    --demo-paths "data/with_bias/wipe/*.pkl" \
    --steps 20000 --intervention-only \
    --output-dir checkpoints/with_bias/wipe/wipe_bc_iter1_$(date +%Y%m%d_%H%M%S)

# 6. 重复 3-5 拿 iter2、iter3...
```

---

## 联合 backup 部署

```bash
# with bias + monitor
python scripts/real/deploy_wipe_with_backup.py \
    --bc-ckpt checkpoints/with_bias/wipe/<your_ckpt>/checkpoints/020000/pretrained_model \
    --ckpt-version v3 \
    --bias --bias-range -0.2 0.2 --bias-monitor

# no bias
python scripts/real/deploy_wipe_with_backup.py \
    --bc-ckpt checkpoints/no_bias/wipe/<your_ckpt>/checkpoints/020000/pretrained_model \
    --ckpt-version v3
```

---

## Wipe 特殊事项

- **gripper_locked="closed"**：env.reset 不会强制张爪，海绵在 episode 之间持续夹住。recovery 路径里 `go_home` 内部仍调 `/open_gripper` 释放海绵，让操作员重新摆好海绵起点。
- **action.shape=[7]**：wipe 不用 discrete gripper head（`num_discrete_actions=null`），actor 输出 7D 全 continuous。所以 BC 训练 log 里**不会有** `disc_ce` 字段，只有 `nll_loss`。
- **max_episode_length=300**（30s @ 10Hz）：比 pickup 长 3 倍，连续涂抹任务。
- **`--lift-threshold` 不适用**：wipe 没有 z 抬升 success 判定，全程 keyboard reward（操作员按 Enter=success / Space=fail）。所以单独部署命令里**不需要** `--lift-threshold`。
