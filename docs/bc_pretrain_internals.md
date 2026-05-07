# BC Pretrain 实现细节

本文档聚焦 BC pretrain 的**实现层细节**：loss 数学、关键模块、demo pipeline。
高层"BC vs SAC 算法关系"、调参建议、训练流水线放在 [`task_policy_training.md`](task_policy_training.md#算法说明bc-与-sac-的关系)，本文不重复。

代码定位：
- 训练入口：`scripts/tools/bc_pretrain_task_policy.py`
- 网络结构：`frrl/policies/sac/modeling_sac.py`（BC 复用 `SACPolicy.actor` + `discrete_critic`）
- Demo loader：`frrl/rl/core/buffer.py:ReplayBuffer.from_pickle_transitions`

---

## 目录

1. [Loss：NLL on tanh-Gaussian](#1-lossnll-on-tanh-gaussian)
2. [SpatialLearnedEmbeddings：从 ViT patch grid 到固定 latent](#2-spatiallearnedembeddings从-vit-patch-grid-到固定-latent)
3. [Demo Pipeline：pickle → ReplayBuffer](#3-demo-pipelinepickle--replaybuffer)
4. [关键设计决策](#4-关键设计决策)
5. [跟标准 BC 的差异](#5-跟标准-bc-的差异)
6. [复现命令](#6-复现命令)

---

## 1. Loss：NLL on tanh-Gaussian

### 1.1 为什么不是 MSE

最 naive 的 BC loss 是

```
L_MSE = E_(s,a)~D [ ‖ μ(s) − a ‖² ]
```

只训练 `mean_layer`，`std_layer` 停在 random init。部署时 `select_action` 走 `dist.rsample()` 抽样，`std` 没被监督过，可能漂到 `[1e-3, 5]` 任意位置（依赖 init 种子）。后果：

- `std` 过小 → 部署接近 deterministic，OOD 状态完全没探索余地
- `std` 过大 → 部署样本散布 [-1, 1] 全域，policy 行为完全乱

要让 BC 同时学 `μ` 和 `σ`，得用 likelihood-based 的 NLL。

### 1.2 输出分布

Actor 输出层是 mean-std diagonal Gaussian，再过 tanh squash 把动作约束到 `(-1, 1)`：

```
μ(s)     = mean_layer(actor_MLP(encoder(s)))      # ∈ ℝ^6
log σ(s) = std_layer(actor_MLP(encoder(s)))       # ∈ ℝ^6
σ(s)     = clamp(exp(log σ(s)), σ_min=1e-5, σ_max=5.0)

z ~ N(μ, diag(σ²))
a = tanh(z) ∈ (−1, 1)^6
```

实现在 `modeling_sac.py:TanhMultivariateNormalDiag`：用 `torch.distributions.TransformedDistribution` 包一个 `MultivariateNormal(μ, diag(σ²))` 做 base，套一个 `TanhTransform` 做 push-forward。

### 1.3 NLL 的精确公式

设 demo action `a_demo ∈ (−1, 1)^6`（脚本里 `clamp(-0.999, 0.999)` 防止 atanh 边界）。
NLL 通过 change-of-variable 公式从 base Gaussian 算回来：

```
log p(a | s) = log N(z; μ, diag(σ²)) − Σᵢ log |dtanh/dz|_zᵢ

其中 z = atanh(a), |dtanh/dz| = 1 − tanh²(z) = 1 − a²

带 batch 后：
L_BC^cont = −E_(s, a_demo)~D [ log p(a_demo | s) ]
          = E [ ½‖(z − μ)/σ‖² + Σ log σ + Σ log(1 − a²) + const ]
```

PyTorch 一行：

```python
dist = TanhMultivariateNormalDiag(loc=μ, scale_diag=σ)
log_prob = dist.log_prob(a_demo)        # shape (B,)
loss_actor = -log_prob.mean()
```

`TransformedDistribution.log_prob` 自动处理 jacobian 项。

### 1.4 离散 head（gripper）loss

Gripper 走独立的 `discrete_critic` head，输出 3 个 logit（`{close=0, no-op=1, open=2}`）。
demo 里 `action[-1] ∈ {−1, 0, +1}`，转换到 class index：

```
gripper_idx = round(a[-1]) + 1, clamp [0, 2]
```

Loss 是标准 cross-entropy：

```
L_BC^disc = E_(s, a_demo)~D [ CE(discrete_critic(s), gripper_idx) ]
```

### 1.5 总 loss

```
L_BC = L_BC^cont + λ · L_BC^disc      # λ = 0.5（discrete_bc_weight 默认）
```

⚠️ **discrete head 不复用 actor MLP**。`discrete_critic` 有自己的 `[256, 256]` MLP + `output_layer(256, 3)`，跟 actor 的 mean/std layer 并列。两条路径共享前面的 encoder。

---

## 2. SpatialLearnedEmbeddings：从 ViT patch grid 到固定 latent

### 2.1 问题

DINOv3-S/16 输出 patch tokens shape `(B, num_patches, 384)`。`modeling_sac.py:1117-1127` 把它 reshape 成 `(B, 384, 8, 8)` 用作 spatial feature map（128/16=8 patch grid）。

但 actor MLP 需要固定长度 latent vector。常见做法是 global average pooling → `(B, 384)`，但这样**丢掉所有空间信息**——pickup 任务里物块的 xy 位置是关键，平均池化把"物块在左上 vs 右下"混成同一个表示。

### 2.2 SpatialLearnedEmbeddings 做什么

给每个空间位置 `(c, h, w)` 学 `F=8` 个权重，把 spatial map 通过加权和压成 `(B, C·F)` 向量。代码（`modeling_sac.py:SpatialLearnedEmbeddings`）：

```python
class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        # learnable kernel: shape (C, H, W, F)
        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))
        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        # features: (B, C, H, W)
        # 沿 H, W 加权求和（每 F 一组权重）
        out = (features.unsqueeze(-1) * self.kernel.unsqueeze(0)).sum(dim=(2, 3))
        # out: (B, C, F) → flatten → (B, C·F)
        return out.view(out.size(0), -1)
```

数学上：

```
out_{b, c, f} = Σ_{h, w} features_{b, c, h, w} · kernel_{c, h, w, f}
```

每个 `(c, f)` 对应一个对 spatial 加权的"概念"——比如 kernel `(c=front_red_channel, f=0, h, w)` 训练后可能在物块常出现的 xy 位置取大正值，相当于学了一个"物块在视野中央"的 spatial detector。

### 2.3 为什么是 `C·F` 而不是别的输出维度

- **保持 C 维**：直接 reduce H, W 但保留 C，避免把 backbone 学到的 channel 间相关丢失
- **F=8** 让网络学 8 个独立 spatial pattern（前景 / 背景 / 边缘 / 中心 等粗糙分组）
- 输出 `384 × 8 = 3072`，再过 `post_encoder = Dropout + Linear(3072, 256) + LayerNorm + Tanh` 压到 latent_dim=256

### 2.4 跟其他 spatial pooling 对比

| 方法 | 输出维度 | 保留空间信息 | 参数量 |
|---|---|---|---|
| Global Avg Pool | C | ❌ | 0 |
| Spatial Softmax (Levine 2016) | 2C | ✓ 每 channel 学一个 (x, y) 关键点 | 0 |
| Flatten | C·H·W | ✓ 保全 | 0 |
| **SpatialLearnedEmbeddings** | C·F | ✓ 学 F 个 spatial pattern | C·H·W·F |

跟 hil-serl `serl_launcher.networks.mlp.SpatialLearnedEmbeddings` 一致。比 spatial softmax 更灵活（能学复杂 attention pattern 而非单一 keypoint），比 flatten 参数量小（`C·H·W·F` vs `C·H·W` 投影到 latent 还要 `C·H·W·D`）。

---

## 3. Demo Pipeline：pickle → ReplayBuffer

完整链路在 `buffer.py:from_pickle_transitions` (~line 525-700)。

### 3.1 输入：HIL-SERL 兼容 pickle 格式

```python
[
  {
    "observations":      {"agent_pos": (14,), "pixels": {"front": (128,128,3) uint8, "wrist": ...}},
    "actions":           (7,) float32,                        # [dx,dy,dz,rx,ry,rz,gripper]
    "next_observations": 同上,
    "rewards":           float,
    "masks":             float,                                # 1 - done，BC 不用
    "dones":             bool,
    "infos":             {"is_intervention": bool, ...}        # is_intervention 给 HG-DAgger 过滤用
  },
  ...
]
```

跟 `rail-berkeley/hil-serl` `examples/record_demos.py` 输出的格式完全一致。
我方采集脚本 `scripts/real/collect_demo_task_policy.py` 和 `scripts/real/deploy_bc_with_dagger.py` 也都按这个 schema 写。

### 3.2 Pipeline 全景

```
[input]  list of pickle files (支持 glob: "data/no_bias/pickup/*.pkl")
   │
   ▼
[1] 文件展开 + 加载
   - sorted(glob.glob(p)) per pickle_path
   - 顺序拼接 transitions（不 shuffle）
   │
   ▼
[2] action_norm_min 过滤  (optional, 默认 0.0 = 不过滤)
   - 丢掉 ‖a‖₂ ≤ action_norm_min 的近零 action 帧
   - 用途：操作员停手观察、按 Enter 期间的静止帧会让 model 偏向"原地不动"
   - hil-serl train_bc.py:183 同款 trick
   │
   ▼
[3] intervention_only 过滤  (optional, 默认 False)
   - 仅保留 infos.is_intervention=True 的帧
   - 用途：HG-DAgger iter，只用人类介入帧训练
   - 老 demo（collect_demo 全程介入）没这个 key，.get(..., default=True) 默认保留
   │
   ▼
[4] 创建 ReplayBuffer 容量 = len(transitions)（或 caller 指定 capacity ≥ len）
   │
   ▼
[5] for each transition:
     5a. _flatten_obs_to_tensor_dict
         嵌套 dict 按点号展开：obs["pixels"]["front"] → "pixels.front"
         所有 numpy → torch tensor，前置 batch dim=1
     5b. _remap_and_transform per state key:
         (i)   key_map rename：pixels.front → observation.images.front
         (ii)  HWC→CHW transpose（如果 key 在 transpose_set）
         (iii) bilinear resize（如果 key 在 resize_map，128×128）
         (iv)  uint8 [0,255] → float32 [0,1]（如果 key 在 normalize_set）
     5c. _drop_extra：只保留 state_keys 里声明的 key（去掉 demo 里多余的 environment_state placeholder）
     5d. complementary_info：
         is_intervention: bool（默认 True，老 demo 全保留）
         discrete_penalty: float (默认 0.0)
     5e. replay_buffer.add(state, action, reward, next_state, done, truncated=False, complementary_info=...)
   │
   ▼
[output]  ReplayBuffer 实例
   - .sample(batch_size) 返回 dict:
       "state": {"observation.state": (B,14), "observation.images.{front,wrist}": (B,3,128,128)}
       "action": (B, 7)
       "reward", "done", "next_state", "complementary_info"
   - 自动支持 DrQ image augmentation（采样时调 random_shift）
```

### 3.3 几个工程细节

**为什么 next_observations 也要存？** BC 用不到，但 ckpt 要兼容 SAC online resume，存了 next_state online 阶段才能算 Q-target。

**state_keys 过滤**：online actor 的 transition 只含 `policy.input_features` 的 key（如 `observation.state` + `observation.images.{front,wrist}`），但 demo pickle 里可能有 `environment_state` placeholder（block_pos / plate_pos）。如果 buffer 预分配了多余 key，online transition 进 buffer 时 `for key in self.states: state[key]` 会 KeyError。`_drop_extra` 在加载时统一过滤。

**HWC vs CHW**：demo pickle 里相机帧是 HWC uint8（直接从 OpenCV 拿的），但 `image_encoder` 期望 CHW float。在 buffer load 时一次性转换，避免每 batch sample 时再做。

**resize 在 transpose 之后**：transpose 让 tensor 是 `(B, C, H, W)`，`F.interpolate` 默认期望这个 layout。

---

## 4. 关键设计决策

### 4.1 为什么不用 MSE（详见 §1.1）
MSE 只训 mean_layer，std_layer 停 random init → 部署时随机性失控。NLL 同时学 (μ, σ)，σ 自然收敛到 demo 的真实条件方差（典型 0.05-0.2）。

### 4.2 BC 期间临时关 encoder_is_shared
`SACPolicy.actor.forward(obs, detach=encoder_is_shared)`：默认 `True`，actor 拿 encoder 输出做 detach 防止梯度回传。SAC 训练时是必需的（让 critic 单独训 encoder）。

但 BC 没有 critic loss，必须开梯度让 actor NLL 训 encoder。否则 encoder 停在 random init，部署时 actor 输出近常数（典型 OOD bug：BC ckpt 真机上 gripper 一直张爪不下降）。

修法（`bc_pretrain_task_policy.py:148-150`）：
```python
_original_encoder_is_shared = policy.actor.encoder_is_shared
policy.actor.encoder_is_shared = False    # BC 期间打开梯度
# ... 训练 ...
policy.actor.encoder_is_shared = _original_encoder_is_shared   # 保存 ckpt 前还原
```

ckpt 保存的是 SAC schema，online resume 时 encoder_is_shared 自动是 True，行为正确。

### 4.3 为什么 freeze DINOv3 backbone
- **数据效率**：30-50 demo × ~50 帧 = ~2000 transitions，远不够训 21M 参数 ViT
- **OOD 鲁棒**：frozen pretrained features 比 BC fine-tune 出来的 features 在 unseen state 更稳
- **训练速度**：frozen 时 backbone forward 一次缓存（`get_cached_image_features`），可训部分（spatial_embeddings + post_encoder + state_encoder + actor MLP + heads）只 ~3-5M 参数

### 4.4 为什么 NLL clamp `a_demo` 到 ±0.999
TanhTransform 的 inverse 是 `atanh(a)`，在 `a → ±1` 时 jacobian `log(1-a²) → -∞` 让 loss 数值爆炸。clamp 0.999 留出安全边距。SpaceMouse 输入有时会接近 ±1 触发这个 corner case。

### 4.5 为什么 image augmentation 用 random_shift
hil-serl 标准 DrQ trick（`batched_random_crop` padding=4）。让 encoder 学到 spatial-translation invariance：物块向左偏 1 个 pixel 的图像也应该输出相似 action。小数据 BC 关键正则化手段，没有的话 BC 在物块位置稍偏的状态下立刻 OOD。

实现：每 batch 独立采样 `(offset_h, offset_w) ∈ [-pad, +pad]`，pad-replicate 后 crop 回原尺寸。`buffer.py:random_shift`，跟 hil-serl Jax 版数学等价。

### 4.6 为什么 std_min=1e-5, std_max=5
注意是 **std 边界**（直接 clamp），不是 log_std 边界。
- std_min 太大 → policy 噪声底过高，部署抖动
- std_min 太小（如 1e-9）→ NLL log σ 项数值不稳
- std_max 太小 → demo 中段方差大的状态学不下来（loss 撞天花板）
- std_max 太大（>10） → 部署 sample 撒到 [-1, 1] 全域

`(1e-5, 5)` 是 hil-serl `serl_launcher.networks.actor_critic_nets.Policy:123-124` 同款值。**历史 bug**：曾把 JAX 风格的 log_std 边界 `(-5, 2)` 误用作 std 边界，下界 -5 因 `exp` 永正而无效，effective range 变成 `(0, 2]`，BC 训完 std 漂到 1e-3，部署 deterministic，online SAC resume 时 entropy bonus 失效。

---

## 5. 跟标准 BC 的差异

| | 标准 BC | 本仓库（hil-serl 风格）|
|---|---|---|
| Loss (continuous) | MSE on action | NLL on tanh-Gaussian（同时训 μ + σ）|
| Loss (discrete) | 跟 continuous 合并 | 独立 CE head + Q-style 网络（部署 argmax → {-1, 0, +1}） |
| 网络是否复用 SAC | 不一定 | 必须复用 `SACPolicy`（为 online resume 兼容）|
| Encoder 训练 | 单独 BC loss 直接训 | 默认 SAC 走 detach；BC 时临时关 detach |
| Vision encoder | 通常从零训或全 fine-tune | DINOv3-S frozen + SpatialLearnedEmbeddings 适配头 |
| Spatial pooling | Global avg / flatten | SpatialLearnedEmbeddings（C·F=384·8） |
| 数据增强 | 通常没有 | DrQ random_shift pad=4 |
| 静止帧过滤 | 不做 | `action_norm_min` 阈值 |
| 介入帧过滤 | 不做 | `intervention_only` flag（HG-DAgger iter 用）|

---

## 6. 复现命令

```bash
# 路径 A: 纯 BC pretrain（5000-20000 步）
python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_pickup_real.json \
    --demo-paths "data/no_bias/pickup/*.pkl" \
    --steps 20000 \
    --output-dir checkpoints/pickup_bc_$(date +%Y%m%d_%H%M%S) \
    --action-norm-min 1e-3 \
    --image-aug-pad 4 \
    --discrete-bc-weight 0.5

# 路径 B: HG-DAgger iter（用上一轮 BC ckpt 部署 + 介入采集，重训）
python scripts/real/deploy_bc_with_dagger.py \
    --bc-ckpt checkpoints/pickup_bc_<iter0_TS>/.../pretrained_model \
    --output-dir data/no_bias/pickup        # 新数据存到同目录

python scripts/tools/bc_pretrain_task_policy.py \
    --config scripts/configs/train_hil_sac_pickup_real.json \
    --demo-paths "data/no_bias/pickup/*.pkl" \
    --steps 20000 \
    --intervention-only                       # 只用人类介入帧
    --output-dir checkpoints/pickup_bc_iter1_$(date +%Y%m%d_%H%M%S)
```

---

## 引用关系

| 概念 | 代码位置 | 行号参考 |
|---|---|---|
| BC 训练 main loop | `scripts/tools/bc_pretrain_task_policy.py` | `main()` 全文 |
| TanhMultivariateNormalDiag | `frrl/policies/sac/modeling_sac.py` | `class TanhMultivariateNormalDiag` |
| RescaleFromTanh | `frrl/policies/sac/modeling_sac.py` | `class RescaleFromTanh` |
| Policy.forward (actor) | `frrl/policies/sac/modeling_sac.py` | `class Policy.forward` |
| SpatialLearnedEmbeddings | `frrl/policies/sac/modeling_sac.py` | `class SpatialLearnedEmbeddings` |
| SACObservationEncoder | `frrl/policies/sac/modeling_sac.py` | `class SACObservationEncoder` |
| PretrainedImageEncoder（DINOv3 加载）| `frrl/policies/sac/modeling_sac.py` | `class PretrainedImageEncoder` |
| from_pickle_transitions | `frrl/rl/core/buffer.py` | `ReplayBuffer.from_pickle_transitions` |
| _flatten_obs_to_tensor_dict | `frrl/rl/core/buffer.py` | `_flatten_obs_to_tensor_dict` |
| _remap_and_transform | `frrl/rl/core/buffer.py` | `_remap_and_transform` |
| random_shift (DrQ) | `frrl/rl/core/buffer.py` | `random_shift` |
| Demo schema | `scripts/real/collect_demo_task_policy.py` | 输出 pickle 格式 |
| HG-DAgger 介入采集 | `scripts/real/deploy_bc_with_dagger.py` | `main()` |
