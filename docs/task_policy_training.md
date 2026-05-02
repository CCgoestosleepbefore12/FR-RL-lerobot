# Task Policy 真机训练 Pipeline

本文档覆盖 **pick-place task policy 在 Franka Panda 真机上的 HIL-SERL 训练全流程**：从相机双路采集到 SpaceMouse demo、offline pretrain、online actor+learner 训练。针对性区别 backup policy 的简单部署，这里是一整套训练+部署闭环。

设计决策源自项目约定：`tcp_true` 作弊通道、29D 观测、J1 编码器偏差注入、keyboard reward。

---

## 目录

1. [观测 + 动作空间](#观测动作空间)
2. [双相机硬件 + 命名](#双相机硬件命名)
3. [Keyboard Reward 协议](#keyboard-reward-协议)
4. [Demo 采集](#demo-采集)
5. [算法说明：BC 与 SAC 的关系](#算法说明bc-与-sac-的关系)
6. [Offline Pretrain](#offline-pretrain)
7. [Online HIL 训练](#online-hil-训练)
8. [HG-DAgger 介入数据迭代](#hg-dagger-介入数据迭代)
9. [配置文件参考](#配置文件参考)
10. [常见问题](#常见问题)
11. [相关文件](#相关文件)

---

## 观测 / 动作空间

### Observation `agent_pos` (29D)

| idx | 字段 | 维度 | 来源 |
|-----|------|------|------|
| 0-6 | `joint_pos_biased` | 7 | `/getstate`（含 J1 bias） |
| 7-13 | `joint_vel` | 7 | `/getstate`（bias 不影响 Δq/dt） |
| 14 | `gripper_binary` | 1 | 夹爪开合 `[0, 1]` |
| 15-17 | `tcp_pos_biased` | 3 | `/getstate`（biased FK） |
| 18-21 | `tcp_quat_biased_xyzw` | 4 | `/getstate`（biased FK，scipy xyzw 顺序）|
| 22-24 | `tcp_pos_true` | 3 | `/getstate_true`（**privileged 作弊通道**）|
| 25-28 | `tcp_quat_true_xyzw` | 4 | `/getstate_true`（**privileged 作弊通道**）|

**核心设计**：proprio 字段读 `/getstate`（含 J1 bias），但 `tcp_pos_true` / `tcp_quat_true` 从 `/getstate_true` 拉无 bias 的真值作为"作弊通道"——让 policy 自己对比 biased 关节感 vs 真 TCP 位置，学会识别并补偿 bias。配套消融实验可以遮掉这两路 privileged 通道验证"纯视觉学 bias" 的能力。

和 backup policy 的观测策略**正好相反**：backup policy 读 `/getstate_true`（因为它训练时无 bias），task policy 读 `/getstate`。

### 图像

| 相机 | 硬件分辨率 | 策略输入 | Crop 策略 |
|------|-----------|---------|-----------|
| `front` | 640×480 @ 15fps RGB | **128×128×3** | 中心正方形 crop（ArUco 4 角校准待做）|
| `wrist` | 640×480 @ 15fps RGB | **128×128×3** | 中心正方形 crop |

两路相机**必须同尺寸**（SAC 架构限制，`modeling_sac.py:586` 硬编码 `torch.cat` 所有相机）。硬件 serial 固定：`234222303420=front`、`318122303303=wrist`（见 `frrl/envs/real_config.py`）。

### Action (7D)

`[dx, dy, dz, rx, ry, rz, gripper]`，归一化到 `[-1, 1]`，在 env 内通过：
- `action_scale[0] = 0.03` m/step（xyz delta；P0-3 后从 0.04 调整）
- `action_scale[1] = 0.2` rad/step（rotvec）
- `action_scale[2] = 1.0`（gripper 二值阈值）

**安全网**：`max_cart_speed = 0.30 m/s`。`action_scale[0] × hz = 0.03 × 10 = 0.30 m/s`，恰好等于 cap，保证极限动作不会被 `clip_safety_box` 非线性压缩，policy 学到的动作幅度与执行幅度一致。**调大 action_scale[0] 必须同步调 max_cart_speed**，否则会触发 sim2real 行为漂移。

### Bias 注入

训练期 `J1 ~ U(-0.2, 0.2)` rad 每 episode 随机采样；`EncoderBiasConfig(target_joints=[0], bias_range=(-0.2, 0.2), bias_mode='random_uniform')`。

---

## 双相机硬件 / 命名

两台 D455 **必须挂在 Bus 04 (20000M Gen 2x2) 控制器**，Port 1 和 Port 2。Bus 06 (10000M Gen 2) 端口不稳会挂一台。详见 `docs/rt_pc_runbook.md` 或 memory entry `reference_dual_realsense_usb_setup.md`。

验证脚本：
```bash
python scripts/hw_check/test_d455.py             # 枚举 + 打开所有 D455
python scripts/hw_check/test_d455.py --show      # 实时预览每路
python scripts/hw_check/test_dual_camera.py      # 30s 带宽压测（FPS / drop / USB topology）
```

---

## Keyboard Reward 协议

真机 task policy 不训 reward classifier，直接人工按键给 reward（HIL-SERL paper 里这叫 `HumanClassifier`，我们简化实现为 `frrl/rewards/keyboard_reward.py`）。

| 键 | 时机 | 效果 |
|----|------|------|
| **`S`** | Episode idle（reset 完成后）| 开始 episode，状态机 IDLE → RUNNING |
| **`Enter`** | Episode running | `reward=1`, `terminal=True`，本条 episode 保存 |
| **`Space`** | Episode running | `reward=0`, `terminal=True`，作为失败 rollout（训练时有用）|
| **`Backspace`** | Episode running | `reward=0`, `terminal=True`, **`info["discard"]=True`**，整条 episode 不进 replay buffer |
| timeout | `max_episode_length=300` 步（30s）| `reward=0`, `truncated=True` |

**不使用 `Esc`**：该键保留给上游 `frrl/utils/control_utils.py:init_keyboard_listener()` 做全局停止记录。

协议在 `FrankaRealEnv.reset()` 中阻塞等 S，`FrankaRealEnv.step()` 中 poll 得到 outcome 后构造 info dict：
- `info["succeed"]`: bool, terminal reward == 1
- `info["discard"]`: bool, 整条丢弃
- `info["teleop_action"]`: 实际执行的 action（供 actor 走 RLPD intervention 路径）
- `info["is_intervention"]`: bool，标记 action 是否来自人工（demo 采集阶段恒为 True）

---

## Demo 采集

SpaceMouse 全程遥操作（不用 policy）。每步都是人的 action，每条成功 episode 的 transitions 存成 hil-serl 兼容的 pickle。

### 脚本

```bash
python scripts/real/collect_demo_task_policy.py -n 50                      # 采 50 条 pick-place demo
python scripts/real/collect_demo_task_policy.py -n 5 --no-bias             # 调试用，不注入 bias
python scripts/real/collect_demo_task_policy.py -n 50 --gripper closed     # wipe 海绵任务，夹爪全程闭合
python scripts/real/collect_demo_task_policy.py -n 50 --gripper open       # push 推动任务，夹爪全程张开
```

锁定模式下 SpaceMouse 夹爪按键无效（action[6] 强制 ±1），env.step 内部跳过 `_send_gripper_command`，env.reset 也强制夹爪到锁定状态（避免上一 episode 抓的海绵掉落）。

### 操作流程

```
启动 → EncoderBiasInjector 初始化 → 等待 KeyboardRewardListener
     → [ Episode loop ]
        1. 机器人复位到 reset pose
        2. 每 episode 采样新 bias（J1 随机）
        3. 终端提示 "等待操作者按 S 开始 episode"
        4. 按 S → SpaceMouse 接管
        5. 完成任务 → Enter (保存) / Space (失败丢弃) / Backspace (作废)
     → 达到 N 条 → 汇总存 pickle → env.go_home() → 干净退出
```

### Pickle schema（hil-serl 兼容）

```python
[
  {
    "observations": {"agent_pos": np.ndarray(29,), "environment_state": ..., "pixels": {"front": ..., "wrist": ...}},
    "actions": np.ndarray(7,),
    "next_observations": { ...same structure... },
    "rewards": float,
    "masks": float,   # 1.0 - done
    "dones": bool,
    "infos": {"succeed": ..., "discard": ..., "teleop_action": ..., "is_intervention": True, ...},
  },
  ...
]
```

输出文件名格式：`data/task_policy_demos/task_policy_demos_{N}_{complete|aborted}_{YYYY-MM-DD_HH-MM-SS}.pkl`。

### 验证

```bash
python scripts/tools/inspect_demo_pickle.py data/task_policy_demos/*.pkl
```

打印 transition 数量、observation shape / dtype、reward/done 统计，并尝试 load 进 `ReplayBuffer`（走 adapter 全链路）。

---

## 算法说明：BC 与 SAC 的关系

代码里 `bc_pretrain_task_policy.py` / `pretrain_task_policy.py` / `learner.py` 看起来都跟 SAC 网络绑在一起，但实际跑的算法可以是纯 BC、纯 SAC、或 SAC + BC pretrain 混合。先拆开两者的算法本质，再说为什么共享一套网络代码。

### 算法对比

| | BC (Behavior Cloning) | SAC (Soft Actor-Critic) |
|---|---|---|
| 类型 | 纯监督学习 | Off-policy RL |
| 数据来源 | 人示范的 (obs, action) 对 | 自采 transition (obs, action, reward, next_obs) |
| Loss | `-log π(a_demo \| obs)` （NLL on tanh-Gaussian） | actor: `-Q(s, π(s)) - α·H(π)`; critic: Bellman TD error |
| 目标 | 模仿示范分布 | 最大化 expected return |
| 需要 reward | ❌ | ✅ |
| 需要 critic | ❌ | ✅ 双 Q-net + target net |
| 需要 entropy/temperature | ❌ | ✅ α 温度 + entropy bonus |
| 失败模式 | OOD 状态行为未定义 | actor 跟着错误 critic gradient 漂飞 |
| 调参面 | std_min/max，lr，image aug | 上面那些 + critic_lr + utd + γ + α + temperature_init + warmup |

BC 的极简性正是它对小数据真机场景的优势：没 critic 这一层，就没"critic 还没 calibrate 但 actor 已经在跟它学"的问题。

### 为什么 codebase 里 BC 和 SAC 共享 `SACPolicy` 类

工程取巧，不是算法设计。`scripts/tools/bc_pretrain_task_policy.py` 里你能看到：

```python
policy = SACPolicy(config=cfg.policy)   # ← 用 SAC 网络结构容器
...
loss_actor = -log_prob.mean()            # ← 但 loss 是 BC 的 NLL
optimizer.step()                         # ← 只更新 actor + encoder（+ 可选 discrete_critic）
```

三个理由：
1. **复用网络定义**：`SACPolicy` 已经把 encoder（DINOv3 / ResNet）+ actor MLP + mean/std layer + discrete_critic 都封装好了，BC 直接用就行，省得另写一份 ImitationPolicy 类
2. **ckpt schema 兼容**：保存时按 SAC 格式（含 critic_ensemble 的 random init state、target_critic、temperature 等），未来想 `--resume` 接 SAC online 训练时直接加载，actor 是 BC 训过的、critic 从零起手在线 calibrate
3. **inference 路径统一**：deploy 脚本调 `policy.select_action(batch)`，不管 ckpt 是 BC 训的还是 SAC 训的，前向逻辑一样（critic 在 deploy 时根本不参与决策）

### BC pretrain 期间动哪些参数

```
训练（requires_grad=True 且 optimizer 收）：
  policy.actor.encoder.spatial_embeddings    ← 学视觉空间到 latent 的投影
  policy.actor.encoder.post_encoders          ← per-camera 后处理 MLP
  policy.actor.encoder.state_encoder          ← proprio MLP
  policy.actor.network                        ← actor 主 MLP
  policy.actor.mean_layer                     ← 输出 action 均值
  policy.actor.std_layer                      ← 输出 action log_std（NLL 关键，否则 std 停 random init）
  policy.discrete_critic                      ← 可选，开 --discrete-bc-weight 时用 CE loss 学 gripper 三态

冻结（requires_grad=False 或不进 optimizer）：
  policy.actor.encoder.image_encoder          ← DINOv3 backbone, freeze_vision_encoder=True 默认冻
  policy.critic_ensemble                      ← SAC critic（BC 不用 reward 学不了 Q）
  policy.critic_target                        ← target critic
  policy.log_alpha                            ← temperature
```

⚠️ **BC 期间一个关键修正**：默认 SAC 训练时 actor forward 走 `detach=True`（让 encoder 由 critic loss 更新，actor loss 不回传到 encoder）。BC 没有 critic loss，必须把 `policy.actor.encoder_is_shared=False` 临时关掉，让 actor NLL loss 直接训 encoder。否则 encoder 停在 random init，actor MLP 头只能在 random feature 上拟合，部署时输出近常数。`bc_pretrain_task_policy.py:148-150` 显式做了这个修正，保存 ckpt 前再恢复成 `True` 让 SAC online resume 行为正确。

### 训练流水线全景

```
Demo 采集 (collect_demo_task_policy.py)
   ↓ data/{no_bias|with_bias}/{task}/*.pkl  ← 2026-04-30 数据按 bias 状态分流
   │
   ├─→ 路径 A: BC pretrain (bc_pretrain_task_policy.py)
   │       ↓ checkpoints/{task}_bc_*/checkpoints/N/pretrained_model
   │       │     (SAC-schema ckpt：actor + encoder 已训，critic 全 random init)
   │       │
   │       ├─→ deploy_bc_inference.py     纯 BC 推理，critic 不参与
   │       ├─→ deploy_bc_with_dagger.py    BC + 介入采集，纯推理
   │       │       ↓ data/{no_bias|with_bias}/{task}/*.pkl  ← demo 同目录
   │       │       └─→ 回到 BC pretrain --intervention-only 重训 (HG-DAgger 迭代)
   │       │
   │       └─→ (放弃) HIL-SAC online       --resume 加载 BC ckpt → 解冻 critic + actor
   │                                         在线 SAC 训练，actor 沿 critic gradient 优化
   │
   └─→ 路径 B: SAC pretrain (pretrain_task_policy.py)
           ↓ 同样 SAC-schema ckpt 但 critic 已被 demos 训过（Bellman bootstrap on demos）
           └─→ HIL-SAC online 路径
```

**当前推荐路径**：路径 A 的 BC pretrain → `deploy_bc_with_dagger.py` 介入采集 → BC 重训迭代。路径 B 的 SAC pretrain + HIL-SAC online 在 50–200 demo 小数据 + sparse reward 场景下 actor 解冻后大幅漂移已多次复现（`fix(learner): resume 时保留 cfg.validate() 设的 pretrained_path` 等多个 commit 是为这条路径修的 bug），不推荐用。

### Bug 修复历史与算法路径的对应

之前出现的"actor 解冻后第二个 episode 乱动"系列 bug，**全部发生在路径 B 的 BC pretrain → HIL-SAC online 切换那一步**：

- `2e939c5` learner resume 时丢了 pretrained_path → BC actor 没加载，用 random init 当 actor
- `edd64eb` warmup 期 actor 必须冻 → 防 BC 权重被 random critic 错误梯度推飞
- `4ad1297` warmup 期 shared encoder 也要冻 → 防 critic loss 改 actor 用的 encoder
- `712b74c` critic_only_online_steps 期间 encoder 也要冻 → 同上理由扩展到 actor 解冻前

BC 自己的训练循环（`bc_pretrain_task_policy.py` 内部，没切到 SAC online）从来没出过这类 bug——因为没有 critic 这一层，不存在"critic 还没好但 actor 已经在跟它学"的状态。

---

## Offline Pretrain

基于 hil-serl `train_rlpd.py` 的 demo-in-offline-buffer 模式。复用 `frrl/rl/core/learner.py` 的 `offline_warmup_steps` 作为 pretrain loop，跑 N 步 critic + actor + temperature 梯度更新，然后保存 checkpoint 退出。

**核心改动**：`offline_only_mode=True` 时：
1. 跳过 gRPC actor server 启动
2. 用 `ReplayBuffer.from_pickle_transitions` 从 pickle 加载 demos 到 `offline_replay_buffer`
3. warmup loop 跑 `offline_pretrain_steps`（默认 5000）
4. 保存 checkpoint + wandb.finish() + return，不进 online 主循环

**HIL-SERL 混合模式（S6 方案 A 放开）**：`offline_only_mode=False + demo_pickle_paths=[...]` 时：
1. Learner 启动仍起 gRPC actor server（走正常 HIL 通路）
2. 同样用 `from_pickle_transitions` 把 demos 灌进 `offline_replay_buffer`
3. `offline_warmup_steps` 做 warmup（默认 500；resume 时跳过，pretrain 已经做过）
4. 进 online 主循环，RLPD 50/50 mix offline buffer + online buffer

这条路径支撑 `train_hil_sac_task_real.json`：先用 pretrain ckpt `--resume`，再带着 demos 进 online，避免冷启动。如果 `demo_pickle_paths=[]` 且 `dataset=None` 但 `offline_warmup_steps>0`，learner 直接 ValueError（避免静默退化成冷启动）。

### Phase 2：真机 critic-only warmup（actor 冻结）

从 pretrain ckpt 恢复后进真机，actor 已经是预训练过的 policy，但 critic 面对的是真机新分布。直接放开 actor + critic 联合更新会让两者同时被稀疏 online 数据冲击。解法：**前 N 真机 step 内冻结 actor/temperature，只训 critic**（50/50 mix 让 critic 渐进适配真机分布），N 步后解冻。

`SACConfig.critic_only_online_steps`（默认 0 = 关闭）控制这个窗口。配合 `online_step_before_learning`（buffer 至少要够一个 online batch 才能开训）使用：

| 真机交互步 | 阶段 | Critic | Actor/Temperature |
|-----------|-----|--------|---|
| 0 ~ `online_step_before_learning` | 冷启动 | ❌ | ❌ |
| `online_step_before_learning` ~ `critic_only_online_steps` | Phase 2 | ✅ 50/50 mix | ❌ 冻结 |
| `critic_only_online_steps` 之后 | Phase 3 | ✅ | ✅ |

`train_hil_sac_task_real.json` 的默认配置：`online_step_before_learning=500, critic_only_online_steps=2000`，即真机 500~1999 步 critic-only（约 2.5 分钟 @ 10 FPS），2000 步后 actor 解冻。这个窗口在 P0-6 真机风险审查后从 100/500 上调，原因：(a) 等 online buffer 够 1 个独立 batch 再开训，避免 batch shrink 把同一 transition 反复梯度踩；(b) 真机 6+ episode 让 critic 在 RLPD 50/50 下充分适配 online 分布后再解冻 actor。

关键契约：`critic_only_online_steps > online_step_before_learning`，否则 Phase 2 窗口为空。

### 运行

```bash
# 先算 dataset stats（真实 min/max/std）
python scripts/tools/compute_dataset_stats.py --demos "data/task_policy_demos/*.pkl"
# 复制输出 JSON 到 scripts/configs/train_task_policy_franka.json 的 policy.dataset_stats 下

# Pretrain
python scripts/tools/pretrain_task_policy.py \
    --config scripts/configs/train_task_policy_franka.json \
    --demo-paths "data/task_policy_demos/*.pkl" \
    --steps 5000 \
    --output-dir checkpoints/task_policy_pretrain_$(date +%Y%m%d_%H%M%S)
```

### Pickle → Buffer 适配

`ReplayBuffer.from_pickle_transitions` 负责：
1. **展平嵌套 dict**：`obs["pixels"]["front"]` → 键 `"pixels.front"`
2. **重命名键**（`demo_key_map`）：`"pixels.front" → "observation.images.front"`, `"agent_pos" → "observation.state"`（匹配 SAC 要求的 `observation.*` 前缀）
3. **HWC → CHW 转置**（`demo_transpose_hwc_to_chw`）：`(H, W, C) → (C, H, W)`
4. **Resize**（`demo_resize_images`）：bilinear 缩放，用于历史 224² front demo 转成 128²
5. **/255 归一化**（`demo_normalize_to_unit`）：uint8 `[0, 255]` → float32 `[0, 1]`，匹配 online 端 `VanillaObservationProcessorStep` 的归一化
6. **Complementary_info schema 预分配**：保留 `discrete_penalty` + `is_intervention` 字段，避免 online 阶段加进来的新字段被静默丢弃

### 参数量验证

**ResNet10 baseline**（`shared_encoder=true`, `freeze_vision_encoder=true`, 2 相机 128²，含 actor + 2 critics + targets + discrete critic + temperature）：
```
num_learnable_params ≈ 3.57M
num_total_params ≈ 8.47M  (含 4.9M frozen ResNet10)
```

**DINOv3-S baseline**（2026-04-26 起切换；ViT-S/16, 22M frozen）：
```
num_learnable_params ≈ 3.94M  (略多于 ResNet10：ViT 投影 + actor/critic 不变)
num_total_params ≈ 26.00M  (含 22.06M frozen DINOv3)
```

obs encoder 单独：trainable 2.08M / total 24.14M（含 frozen ViT），SAC 算法侧 actor+critics+targets 占额外 ~1.86M trainable。

切换 DINOv3-S 的动机：sim 分类任务用 ResNet10 失败，证明 ResNet10 ImageNet 预训特征对 manipulation 任务不够分离。DINOv3 用 LVD-1689M (1.69B 张) 自监督预训，dense feature 质量首次超过 weakly-supervised 模型。`PretrainedImageEncoder` 已加 ViT 适配（patch token sequence → 4D feature map），ResNet 路径不破坏。

⚠️ **DINOv3 是 HuggingFace gated repo**：首次下载需 (1) `huggingface-cli login` (2) 在 https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m 上 accept license。否则 SAC 启动时 OSError。

数量级不符请先检查：
- `shared_encoder` 是否 true（false 会导致 3× 参数）
- 图像分辨率（224 会比 128 多 4× encoder latent，DINOv3 patch 16 → 14×14=196 patch）
- `num_discrete_actions` 是否为 null（非 null 会加 discrete critic 头）

---

## Online HIL 训练

**状态：阶段 6，待完成**。基础设施已就绪（actor 支持 discard hook、learner 主循环混样条件修对、env 写 `teleop_action`/`is_intervention`）。剩余工作：

1. 写 `InterventionWrapper` 包住 `FrankaRealEnv`，SpaceMouse 推动时 override policy action + 标记 `set_intervention(True)`（参考 hil-serl `SpacemouseIntervention` wrapper）
2. actor.py 从 `info["is_intervention"]` 路由 transition 到 prior_buffer（hook 已在）
3. Resume from pretrain checkpoint：`--resume path/to/pretrain/checkpoints/last`
4. Online 训练 config：拷贝 `train_task_policy_franka.json`，改 `offline_only_mode=false`，设 `online_steps=100000`

⚠️ **小数据真机场景的实际推荐路径见下一节 HG-DAgger**。Online HIL（SAC critic + actor 联合优化）在 50 demo / sparse reward / 短 horizon 下容易出现 actor 解冻后大幅漂移（critic Q surface 平坦，gradient 任意方向都"合法"）。SAC 适合"已经有 1k+ 真机交互且 reward 信号清晰"的阶段；50–200 条 demo 起家时，HG-DAgger 的迭代式 BC 收敛更稳、调参更少。

---

## HG-DAgger 介入数据迭代

针对小数据真机场景的替代路径：BC 部署 + 人工介入纠错 + 介入帧合并重训。每轮迭代直接补策略的失败模式，无需 critic / reward / replay buffer 调度，调参面比 SAC 小一个数量级。

**适用场景**：50–200 条 demo 训出的 BC 已经能完成部分任务（≥ 30% 自主成功率），但存在 OOD 失败模式。**不适用场景**：iter 0 BC 完全不会做（介入率 > 70%）→ 先回去补 demo，DAgger 拯救不了零起点的策略。

### 工作流

| 阶段 | 命令 | 输出 |
|---|---|---|
| Iter 0 demo 采集 | `collect_demo_task_policy.py --task pickup --no-bias -n 50` | `data/no_bias/pickup/*.pkl` |
| Iter 0 BC 训练 | `bc_pretrain_task_policy.py --demo-paths 'data/no_bias/pickup/*.pkl' --steps 20000` | `checkpoints/pickup_bc_iter0_*` |
| Iter N 介入采集 | `deploy_bc_with_dagger.py --ckpt <iter N-1 ckpt> --task pickup --no-bias --iter N -n 30` | `data/no_bias/pickup/{task}_dagger_iterN_*.pkl` |
| Iter N BC 重训 | `bc_pretrain_task_policy.py --demo-paths 'data/no_bias/pickup/*.pkl' --intervention-only --steps 20000` | `checkpoints/pickup_bc_iterN_*` |

数据目录 layout（2026-04-30 重组）：

```
data/
├── no_bias/{task}/    ← 用 --no-bias 采的 demo + dagger 全在这
└── with_bias/{task}/  ← 默认（bias=ON）采的早期数据
```

demo 和 dagger 共享同一 task 目录，单 glob `data/no_bias/pickup/*.pkl` 一次取全。`--intervention-only` 在 dagger pkl 上过滤介入帧，原 demo 因为没 `is_intervention` key 默认全保留（向后兼容）。

每轮重训仍是从零初始化 SACPolicy（不复用上轮 ckpt 权重），在合并数据集上跑 20000 步。

### `deploy_bc_with_dagger.py` 的状态机

部署时纯 BC 推理；SpaceMouse 推杆 / 按键 → 介入触发，覆盖 policy action：

| 状态 | 触发 | Action 来源 | `is_intervention` 标签 |
|---|---|---|---|
| IDLE | sm 静止 | policy | False |
| IDLE → ACTIVE | `||sm[:6]|| > enter_threshold (0.05)` 或任意按键 | sm | True |
| ACTIVE | sm 仍活动 | sm | True |
| ACTIVE → IDLE | sm 静止持续 `tail_k=10` 帧 | sm（near-zero） | True |

**tail_k 的设计意图**：操作员的"释放瞬间"通常发生在恢复未完成时（看到要稳了就松手）。tail_k=10（≈1s @ 10Hz）让释放后的 follow-through 帧仍标为介入，保留恢复尾巴。tail_k 过短会切掉恢复轨迹，过长会把"已经完全回到正常"的帧也污染成介入帧。

操作协议同 `collect_demo_task_policy.py`：S 启动 episode，Enter=success（整 episode 全帧落盘），Space/Backspace=discard，Ctrl+C=退出（已存的 success 仍 dump）。

### Pickle schema：`is_intervention` per-frame

DAgger pkl 与原始 demo 共用 hil-serl transition dict schema，唯一新增是 `infos["is_intervention"]: bool`：

```python
{
    "observations": {...},
    "actions": np.ndarray(7,),
    "next_observations": {...},
    "rewards": float,
    "masks": float,
    "dones": bool,
    "infos": {
        "succeed": bool,
        "discard": bool,
        "teleop_action": np.ndarray(7,),
        "is_intervention": bool,    # ← 新增：True=人介入帧，False=BC 自主帧
    },
}
```

**向后兼容**：`buffer.py` 的 `from_pickle_transitions` 在读取时 `.get("is_intervention", True)`，老 demo（无该 key，全程 teleop）默认全是 intervention=True。所以 `--intervention-only` flag 在合并 demo + dagger 数据时：原 demo 全保留，dagger 数据只取人介入帧。

### `--intervention-only` 过滤逻辑

```python
ReplayBuffer.from_pickle_transitions(
    pickle_paths=[...],
    intervention_only=True,   # 仅保留 infos.is_intervention=True 的 transition
    ...
)
```

典型 iter 1 数据流：50 demo（≈2500 transitions, 全保留）+ 30 dagger episodes（≈1800 transitions, ~15% 介入率 → 留 ~280 帧）= 合并 ~2800 transitions。介入帧只占 10%，主要修复来自这 ~280 帧对失败模式的精准覆盖。

### 判停准则（人工）

每轮 dagger 采集结束看 log 中三个指标：

1. **`intvn (Y%)` 趋势**：连续 5+ episode `Y ≤ 5%` → 策略自主能力够了，停
2. **zero-intvn episode 占比**：≥ 80% → 大部分场景已无需介入
3. **DROPPED (fail/timeout) 占比**：还在下降说明仍有可改进空间

脚本本身不自动判停，到 `-n` 数后退出。一般 1–3 轮迭代足够 pickup 类简单任务。

### 与 hil-serl 关键 config 对齐

完全一致：`reset_pose / abs_pose_limit / compliance_param / precision_param / max_episode_length=100 / gripper_sleep=0.6 / hz=10 / 相机 serial / IMAGE_CROP / discount=0.97 / utd=2`。

不一致（已知）：
- **action_scale[0]**：frrl=0.05 m/step，hil-serl pickup=0.015 m/step（3.3× 大）。`real_config.py:88` 注释自称"对齐 hil-serl"是错的；hil-serl pickup 实际更紧。**已采的 demo 是用 0.05 训的，iter 0 BC 也是 0.05 训的，暂不要改**——下次完整重做时考虑切回 0.015 提升精细操控
- **proprio_keys**：frrl=14D `[tcp_pose_true(7), tcp_vel(6), gripper(1)]`，hil-serl pickup=19D `[tcp_pose, tcp_vel, tcp_force(3), tcp_torque(3), gripper_pose]`。**缺 force/torque 6 维**——pickup 接触类任务里 force/torque 是关键反馈，TODO 是回头加上
- **encoder**：DINOv3 ViT-S/16（frrl）vs frozen ResNet10 ImageNet（hil-serl）——刻意为之

### 实测数据（pickup iter 1, 30 episodes）

```
total transitions: 1819
  is_intervention=True : 288 (15.8%)
  is_intervention=False: 1531

episode 长度: min=36, max=97, mean=60.6
zero-intvn episodes: 18/30 (60%)
有介入 episodes: 12/30, 介入率 20–44%
```

iter 0 BC（用 50 demo 训）已能 60% 自主成功；介入主要发生在"卡住的拐点"（有介入的 episode 长度 67–97 显著高于无介入的 36–60）。iter 1 重训后预期 zero-intvn 升到 80%+。

### 已知瓶颈：gripper close 信号稀疏

dagger 数据 1819 帧只有 34 个 close (-1) 信号（1.87%），跟 50 原始 demo 同样稀疏。这是 SpaceMouse "按键瞬间→发 -1" 的固有问题（每帧 `get_action` 只返回当下 buttons 状态，按住期间不持续输出）。

如果 iter 1+ 后发现 close 学不稳（比如夹爪反复开合），需要：(a) sticky-close 模式重采 demo，按键松开后继续输出 -1 共 N 帧；或 (b) latching 模式，按 close 后每帧都发 -1 直到按 open。这会破坏现有 ckpt（demo 分布变了），是大改。

---

## 配置文件参考

**主配置**：`scripts/configs/train_task_policy_franka.json`

关键字段（pretrain 特定）：

```jsonc
{
    "policy": {
        "shared_encoder": true,          // 共享 encoder 省 3× 参数
        "freeze_vision_encoder": true,   // 冻结 DINOv3-S backbone（之前是 ResNet10）
        "offline_only_mode": true,       // 切换 pretrain 模式
        "offline_pretrain_steps": 5000,
        "demo_pickle_paths": [],         // CLI 覆盖
        // 以下 4 项专供 pickle adapter
        "demo_key_map": { ... },
        "demo_transpose_hwc_to_chw": [ ... ],
        "demo_resize_images": { ... },   // 老 224² demo 才需要
        "demo_normalize_to_unit": [ ... ],
        "dataset_stats": {
            "observation.state": { min: 29 floats, max: 29 floats },
            "action": { min: [-1]*7, max: [1]*7 },
            "observation.images.*": { mean: ImageNet 默认, std: ImageNet 默认 }
        }
    }
}
```

---

## 常见问题

**Q: Pretrain 跑完后 checkpoint 在哪？**
A: `{output_dir}/checkpoints/last/pretrained_model/` 和 `training_state/`，格式和 online 训练一致，可直接 `--resume`。

**Q: 相机 serial 变了怎么办？**
A: 编辑 `frrl/envs/real_config.py` 的 `cameras` dict 里对应 `serial_number`，或设为占位符 `"000000000000"` 让 `_resolve_camera_serials()` 按枚举顺序自动分配。

**Q: 为什么图像两路都 128²？能不能把 front 调回 224²？**
A: SAC `SACObservationEncoder` (`modeling_sac.py:586`) 硬编码把所有相机 `torch.cat` 到同一个 encoder forward，要求同尺寸。要支持不同分辨率得 refactor 成 per-key encoder（tech debt backlog）。

**Q: `info["teleop_action"]` 是什么时候被 actor 读的？**
A: `frrl/rl/core/actor.py:323-326`，每步 step 后从 info 拿出来作为 `executed_action`（代替 policy 的 action），进 replay buffer。HIL-SERL 的 "H" 就靠这个通路实现 intervention replay。

**Q: dataset_stats 的 29D state min/max 怎么估？**
A: 跑 `scripts/tools/compute_dataset_stats.py --demos "data/task_policy_demos/*.pkl"` 自动从真实 demo 算。也可以保留 config 里现有的理论边界（Franka 关节极限、workspace bounding box 等）。

**Q: 想要 ArUco workspace ROI crop？**
A: 已做（2026-04-26）。`scripts/hw_check/select_workspace_roi.py` 鼠标拖拽框选 + S 自动调正方形 + W 写回 `calibration_data/workspace.json`。当前落档 `(200, 171, 408, 379)` = 208² 正方形，已 wired 到 `frrl/envs/real_config.py::image_crop["front"]` 用 `make_workspace_roi_crop(...)`。

**Q: 训练别的任务（如 wipe 海绵）需要夹爪锁定怎么办？**
A: `FrankaRealConfig.gripper_locked` 字段：`"none"`（默认 pick-place）/ `"closed"`（wipe）/ `"open"`（push）。锁定模式下 env.reset 时强制夹爪到锁定状态，env.step 跳过夹爪 HTTP 调用。collect_demo 加 `--gripper closed` 同时屏蔽 SpaceMouse 按键、action[6] 强制 -1。⚠️ wipe 任务 action[6] ≡ -1 const → `compute_dataset_stats.py` 自动 const-channel guard（max=min+1），避免 MIN_MAX 归一化 NaN。

**Q: live view 看不清两路相机？**
A: `_render_live_view` 单窗 hstack：左 front / 右 wrist，**显示的是 image_crop + resize 后送 vision encoder 的真实输入**（128²，3× upscale 让小目标可见）。和原始 640×480 不一样，看的就是 policy 视角。

---

## 相关文件

| 用途 | 文件 |
|------|------|
| Env 类 | `frrl/envs/real.py`, `frrl/envs/real_config.py` |
| Reward | `frrl/rewards/keyboard_reward.py` |
| Buffer 适配器 | `frrl/rl/core/buffer.py::ReplayBuffer.from_pickle_transitions` |
| Actor discard hook | `frrl/rl/infra/actor_utils.py` + `actor.py` |
| Learner pretrain | `frrl/rl/core/learner.py::add_actor_information_and_train` (offline_only_mode 分支) |
| 配置 | `scripts/configs/train_task_policy_franka.json` |
| 脚本 | `scripts/real/collect_demo_task_policy.py`；`scripts/tools/{inspect_demo_pickle,pretrain_task_policy,compute_dataset_stats}.py`；`scripts/hw_check/test_dual_camera.py` |
| 测试 | `tests/test_*.py`（115 cases 覆盖 pickle adapter / 键盘状态机 / 观测布局 / config 字段等）|
