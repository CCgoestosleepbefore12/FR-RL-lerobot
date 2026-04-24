# Backup Policy 真机部署指南

Backup policy 是仿真训练、真机部署的安全策略：检测到人手接近时接管控制，引导机械臂闪避。
本文档是从**零开始**的完整部署流程。

---

## 0. 前置条件

### 0.1 硬件

| 设备 | 位置 | 备注 |
|---|---|---|
| Franka Panda 机械臂 + 控制柜 | 工作台 | 固件 4.2.1，libfranka 0.9.1 |
| Franka Hand 夹爪（原装）| 机械臂末端 | 标准 80×90×120mm |
| Intel RealSense D455（前置）| 固定支架俯视工作台 | USB 3.0 到 GPU 站 |
| 3Dconnexion SpaceMouse Compact | GPU 站 USB | teleop 占位用 |
| GPU 工作站 RTX 3090 Ti | — | Ubuntu 20.04，标准内核 |
| RT PC（笔记本）| — | Ubuntu 20.04 + PREEMPT_RT，装 libfranka |
| AprilTag 或 ArUco 标定板 | A4 纸印出来贴夹爪 | 标定用，尺寸量准 |

### 0.2 网络

GPU 站 ↔ RT PC 直连或经交换机，**同一子网**：
- GPU 站 `enp13s0`：静态 IP `192.168.100.2/24`
- RT PC（连 Franka 的另一块网卡）：静态 IP `192.168.100.1/24`

验证：
```bash
ping -c 2 192.168.100.1
```

### 0.3 RT PC 准备

Flask server 必须在跑：
```bash
# RT PC
~/start_franka_server.sh
```

验证：
```bash
# GPU 站
curl -X POST http://192.168.100.1:5000/getstate_true
```
应返回带 `q`, `dq`, `pose`, `gripper_pos` 的 JSON。

**Franka Desk 网页**（浏览器访问 `https://172.16.0.2`）确认：
- 面板灯 **白色**
- 关节已解锁
- FCI 已激活

### 0.4 GPU 站依赖

```bash
conda activate lerobot
pip install pyrealsense2 ultralytics==8.1.34 opencv-python scipy requests torch
# SpaceMouse（项目内 bundled pyspacemouse）
pip install easyhid
```

系统层一次性：
```bash
# HID 设备权限（SpaceMouse）
sudo apt install -y libhidapi-dev libhidapi-libusb0 libhidapi-hidraw0
sudo tee /etc/udev/rules.d/99-spacemouse.rules <<'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", MODE="0666"
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
# 拔插一次 SpaceMouse
```

### 0.5 Backup Policy Checkpoint

位置：`checkpoints/backup_policy_s1/`
内容：
- `model.safetensors` — SAC 权重
- `config.json` — 策略配置 + 归一化参数
- `train_config.json` — 训练超参

---

## 1. 相机标定（T_cam_to_robot）

目标：求出 D455 相机坐标系到机器人基座坐标系的 4×4 变换矩阵。

### 1.1 打印 ArUco Marker

- 用 `DICT_5X5_100`，任意 ID（后面都用这个 ID）
- 推荐尺寸：**15cm × 15cm** 易检测但挡抓取，或 **7cm × 7cm** 小但需要近距离
- 打印后**量准黑框实际边长**（mm），这是标定精度关键
- 平整贴在夹爪**侧面**或夹指间的平面

### 1.2 启动 SpaceMouse 标定脚本

```bash
conda activate lerobot
cd ~/FR-RL-lerobot
python scripts/real/calibrate_cam_to_robot.py --marker-id 55 --marker-size 0.15
```

### 1.3 采集流程

1. Flask server 必须在跑（进入 impedance mode）
2. 用 **SpaceMouse** 缓慢移动机械臂到 **10-15 个不同姿态**：
   - 位置分散（工作空间前后左右上下）
   - **旋转多样化**（手腕转角变化大，不能只平移）
3. 每到一个姿态：**松开 SpaceMouse 等静止 → 按 Enter 采集**
4. 画面显示 `MARKER OK` 才有效
5. 按 **q** 结束解算

### 1.4 结果

保存在 `calibration_data/`：
- `handeye_pairs.json` — 原始数据（可 `--load` 重算）
- `T_cam_to_robot.npy` — 4×4 矩阵，**后续所有步骤共用**

### 1.5 求解方法

脚本用 **SVD 点对齐作为主求解**（只用平移信息），hand-eye（`cv2.calibrateRobotWorldHandEye`）
作为诊断对比。

SVD 的优势：在小样本（<15 组）或旋转多样性不足时更鲁棒。多次实验显示 SVD 的 mean error
稳定在 ~15-20mm，而 hand-eye 在某些数据上会给出 >300mm 的错误解。

脚本自动两种方法都跑，只有 hand-eye 明显更好（<0.5× SVD）时才覆盖 SVD 结果。

### 1.6 精度判断

reprojection error 参考：
- `mean < 10mm` → 优秀
- `mean < 30mm` → 可用（backup policy 训练 DR 是 σ=30mm 所以 OK）
- `mean > 50mm` → 重采集（注意分散位置 + **多变旋转**）

---

## 2. 手部检测验证

### 2.1 下载 WiLoR Hand Detector

WiLoR 自带一个 YOLOv8 手部检测器（`detector.pt`），我们只用这个检测部分，不用完整 3D 重建。

```bash
cd /home/lab1
git clone https://github.com/rolpotamias/WiLoR.git
cd WiLoR
wget -q https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
     -O pretrained_models/detector.pt
```

`HandDetector` 类硬编码路径为 `/home/lab1/WiLoR/pretrained_models/detector.pt`。

### 2.2 实时手部检测测试

```bash
conda activate lerobot
cd ~/FR-RL-lerobot
python scripts/hw_check/test_hand_detector.py
```

期望看到：
- 手在 D455 视野内 → 绿色框 + 3D 位置文字
- 手握拳、侧面、背面 → 都能检测（WiLoR 特点）
- 多只手 → 只取离图像中心最近的一只
- FPS ~15

### 2.3 内部原理

```
D455 RGB → WiLoR YOLOv8 → bbox (u1,v1,u2,v2)
  ↓ bbox 中心像素 (u,v)
D455 depth @ (u,v) → 3D 点（相机系）
  ↓ T_cam_to_robot
hand_pos（机器人基座系）
  + bbox_size_3d = ((u2-u1)*depth/fx, (v2-v1)*depth/fy)
```

---

## 3. Backup Policy 部署

### 3.1 部署脚本

```bash
python scripts/real/deploy_backup_policy.py
```

### 3.2 状态机（HierarchicalSupervisor 三态 FSM）

由 `frrl/rl/supervisor/hierarchical.py` 驱动，**三态**：

```
  ┌─────────────────────────────────────────┐
  │                                         │
  │   hand returns: dist < d_safe           │
  │   (tcp_start retained)                  │
  │                                         │
  ▼                                         │
┌────────┐  dist < d_safe   ┌──────────┐    │
│  TASK  │ ───────────────→ │  BACKUP  │    │
│(green) │  (record tcp_    │  (red)   │    │
│        │   start pos+quat)│          │    │
└────────┘                  └──────────┘    │
    ▲                           │           │
    │                           │ dist > d_clear  │
    │ pos_err<2cm AND           │ for 3 steps     │
    │ rot_err<2.9°              ▼           │
    │                     ┌──────────┐      │
    │                     │  HOMING  │──────┘
    └─────────────────────│ (orange) │
                          │          │
                          │ P 控制回到
                          │ tcp_start 6D 位姿
                          └──────────┘
```

**关键参数**（scripts/real/deploy_backup_policy.py 顶部）：
- `D_SAFE = 0.15m` — surface-to-surface 距离，触发 BACKUP
- `D_CLEAR = 0.20m` — 允许 BACKUP → HOMING 的阈值（滞回）
- `CLEAR_N_STEPS = 3` — 连续 3 帧 `dist > d_clear` 才切 HOMING（防抖动）
- `HOMING_POS_TOL = 0.02m` / `HOMING_ROT_TOL = 0.05rad` — HOMING → TASK 收敛阈值

**关键设计点**：
- `tcp_start_pos/quat` 在 **TASK → BACKUP** 那一瞬间记录**一次**
- BACKUP ↔ HOMING 之间来回切换**不更新** tcp_start（手反复进出时 homing 目标不变）
- HOMING → TASK 完成后 tcp_start 清空

### 3.2.1 HomingController（6D 位姿回归）

`frrl/rl/supervisor/homing.py` 的 6D clipped-P 控制器：

```
position error = tcp_start_pos - tcp_current_pos
rotation error = axis_angle(q_cur⁻¹ · q_start)  ← 局部系右乘约定

action_pos = clip(kp_pos * error / action_scale, ±1)
action_rot = clip(kp_rot * error / rot_action_scale, ±1)
is_done: ||pos_err|| < pos_tol AND ||rot_err|| < rot_tol
```

`kp=1` 是 clipped-deadbeat（无超调）。配合 deployer 的 rate-limit（0.30m/s），典型 5-10 步收敛。

### 3.3 观测构造（关键）

| 项 | 来源 | 维度 |
|---|---|---|
| q | `/getstate_true`（**真值**）| 7 |
| dq | `/getstate_true` | 7 |
| gripper_pos | `/getstate_true` | 1 |
| tcp | `/getstate_true` + 5mm Gaussian 噪声 | 3 |
| obstacle.active | HandDetector 检测到就是 1 | 1 |
| obstacle.pos | **虚拟手点**（Minkowski 球变换后）| 3 |
| obstacle.vel | raw bbox 中心的帧间差 | 3 |
| obstacle.rel_pos | virtual_hand - tcp | 3 |

**28D 每帧 × 3 帧堆叠 = 84D 输入**

### 3.4 Minkowski 球变换（框对框避障）

Backup policy 训练时 obstacle 是一个点，部署要做**两个球之间的避障**。技巧：

```
膨胀人手球（r_hand + r_gripper）
收缩夹爪到 TCP 点
→ 两点距离 = 两球表面间距
```

实现：`compute_virtual_hand_pos()` 函数。

**`R_GRIPPER = 0.10m`**（2026-04-22 起）——**匹配 V2 训练的 `ARM_SPHERE_RADIUS`**，
覆盖 wrist + hand 整个区域的单球，球心 = mocap weld 点（= TCP）。

人手球半径 = `0.5 × bbox_对角线` （从 WiLoR bbox 动态推断，一般 6-14cm）。

效果：policy 的"点对点"距离语义映射到"表面对表面"距离。Policy 不用重训，但观测分布相比
训练有一定偏移（训练用 center-to-center，部署用 surface-to-surface），属于**保守安全**
的有意选择——真人手比仿真 obstacle（3.5cm）大很多，保留 Minkowski 让 policy 感知"更早"。

> 替代方案：严格匹配训练分布用 `r_hand = 0.035`（固定）+ 不做投影。当前为安全优先保留
> 当前实现。详见 commit 中 "方案 B" 讨论。

### 3.5 动作执行（关键）

#### 四元数约定

**Franka Server 使用 scipy 约定 `[x, y, z, qx, qy, qz, qw]`（w 在最后）**。

⚠️ 仿真代码的 `mat_to_quat` 返回 `[w, x, y, z]`（w 在前），**千万不要混用**。

deployer 里保持 scipy 约定：
```python
target_quat_xyzw = list(state["pose"][3:])  # 直接当 xyzw 用
pose7 = [*target_xyz, *target_quat_xyzw]    # 发给 /pose
```

#### Target 帧选择（关键）

不同目的用不同端点：

| 用途 | 端点 | 原因 |
|---|---|---|
| 策略观测 | `/getstate_true` | 匹配训练分布（训练时无 bias）|
| Target 起点 | `/getstate` | 和 C++ 控制器内部"当前"同一帧，不然有 bias 时会产生幽灵误差 |

代码：
```python
# 观测路径
state_true = get_state_true()
obs = build_obs28(state_true, ...)

# 执行路径
state_biased = get_state_biased()
actual_xyz = state_biased["pose"][:3]
target_xyz = actual_xyz + action_scaled_xyz  # delta 来自 policy
send_pose(target_xyz, target_quat)
```

#### 累积方式

`target = actual + delta`（每步重算），**不是** `target += delta`。原因：累加会让 target 跑偏，impedance 弹簧力爆炸，触发 libfranka reflex。

#### Look-ahead

```python
action_scaled = action × ACTION_SCALE × LOOKAHEAD  # LOOKAHEAD = 2.0
```

因为"每步重算"模式下阻抗控制器跟不上，多加一倍 delta 补偿滞后。

#### Rate Limit

```python
MAX_CART_SPEED = 0.30 m/s
```

单步 delta 向量模长超过 `MAX_CART_SPEED * dt` 就缩放。

### 3.6 工作空间边界

```python
WORKSPACE_MIN = [0.20, -0.30, 0.10]  # 保守的笛卡尔 bounds
WORKSPACE_MAX = [0.70,  0.30, 0.60]
```

Target 发送前 clip。根据你的桌面布局调整。

**设计理念**：训练侧关闭 workspace clamp（`enforce_cartesian_bounds=False`），让位移
penalty 的梯度天然偏好沿 `-hand_dir` 直线退让；部署侧再加 clamp 作为真机硬约束。两者正交。

**`--no-workspace-clamp` 标志**：临时关闭 clip 用于调试/测量可达空间。**使用时手放急停**：
```bash
python scripts/real/deploy_backup_policy.py --no-workspace-clamp ...
```
启动时会打印 `!! WORKSPACE CLAMP DISABLED` 提醒。

### 3.7 关键默认参数

| 参数 | 值 | 说明 / 可调范围 |
|---|---|---|
| `BACKUP_ACTION_SCALE` | 0.025 m/step | 0.015-0.030 |
| `BACKUP_ROTATION_SCALE` | 0.020 rad/step | 0-0.03（为 0 则锁姿态）|
| `TASK_ACTION_SCALE` | 0.025 m/step | SpaceMouse 手感 |
| `TASK_ROTATION_SCALE` | 0.040 rad/step | SpaceMouse 手腕灵敏度 |
| `LOOKAHEAD` | 2.0 | 1.5-3.0；补阻抗跟踪滞后 |
| `MAX_CART_SPEED` | 0.30 m/s | 单步 delta 速度硬上限 |
| `D_SAFE` | 0.15m | 触发 BACKUP 的 surface-to-surface 距离 |
| `D_CLEAR` | 0.20m | 允许 BACKUP→HOMING 的滞回阈值 |
| `CLEAR_N_STEPS` | 3 | 连续清除帧数（防抖动）|
| `HOMING_POS_TOL` | 0.02m | HOMING → TASK 位置容差 |
| `HOMING_ROT_TOL` | 0.05rad | HOMING → TASK 姿态容差 (≈2.9°) |
| `R_GRIPPER` | **0.10m** | V2 `ARM_SPHERE_RADIUS` 对齐 |

### 3.8 启动流程

```bash
# 1. 清僵尸进程 + 重启 Flask server（RT PC）
pkill -9 -f franka ; pkill -9 -f roslaunch ; pkill -9 -f roscore ; sleep 2
~/start_franka_server.sh

# 2. Franka Desk 确认白灯

# 3. GPU 站启动 deployer
conda activate lerobot
cd ~/FR-RL-lerobot
python scripts/real/deploy_backup_policy.py
```

Deployer 会自动：
- `/clearerr` → `/stopimp` → `/clearerr` → `/startimp`
- 加载 checkpoint 到 CUDA
- 启动 D455 + WiLoR + SpaceMouse

### 3.9 测试与验证

```
阶段 1 — Dry run（不发 /pose，零风险）
python scripts/real/deploy_backup_policy.py --dry-run
确认：策略加载 OK、观测维度正确、推理不报错

阶段 2 — 纯触发测试（SpaceMouse 别动）
python scripts/real/deploy_backup_policy.py
手伸近 TCP → 机械臂应主动后退 → 手离开 → 静止
观察 MODE 颜色切换、backup_step 计数、避让方向

阶段 3 — SpaceMouse + Backup 联动
用 SpaceMouse 驱动机械臂做任务 → 伸手 → backup 接管 → 退出后恢复 SpaceMouse
```

---

## 4. 偏差注入验证（可选）

### 4.1 注入原理

C++ 阻抗控制器内部加 bias：
- `q_measured` 作为控制器"眼中的当前" = `q_true + bias`
- Jacobian / FK 计算都基于 biased 值
- 物理力矩指令方向偏 → 执行有偏差

### 4.2 端点

| 端点 | 作用 |
|---|---|
| `/set_encoder_bias` `{"bias": [7 floats]}` | 设置偏差向量 |
| `/clear_encoder_bias` | 清零 |
| `/get_encoder_bias` | 读当前值 |

### 4.3 注入测试

```bash
# 无 bias 基线（启动时强制清零）
python scripts/real/deploy_backup_policy.py --clear-bias

# Joint 1 注入 0.1 rad（≈5.7°）
python scripts/real/deploy_backup_policy.py --bias "0.1,0,0,0,0,0,0"

# Joint 4 注入 0.2 rad（末端偏移 ~13cm，大幅偏差）
python scripts/real/deploy_backup_policy.py --bias "0,0,0,0.2,0,0,0"

# 手动清除（脚本正常退出或 Ctrl-C 会自动清，这条只在崩溃遗留 bias 时需要）
curl -X POST http://192.168.100.1:5000/clear_encoder_bias
```

**Deployer 退出时自动清零**：finally 块里会调 `/clear_encoder_bias`，防止 bias 在多次
run 之间残留。崩溃退出（如 libfranka reflex 中断）可能来不及清，需手动 curl。

### 4.4 预期行为

- 策略观测不受影响（用 `/getstate_true`）
- 执行端物理运动偏离指令方向 ~bias 对应的角度
- Backup policy **基于真实感知做决策**，但**执行被扭曲**
- 验证问题：policy 的决策余量够不够抵消执行偏差？

### 4.5 验证实验设计

| 实验 | Bias (J1) | 记录指标 |
|---|---|---|
| Baseline | 0 | 最小 surface-dist、触发-释放时间、避让方向 |
| Small | 0.05 rad | 同上 |
| Medium | 0.10 rad | 同上 |
| Large | 0.15 rad | 同上（可能开始失败）|

每组跑 N 次人手接近，统计成功率。

---

## 5. 故障排查

### 5.1 libfranka `server closed connection`

**症状**：RT PC 终端报 `franka::NetworkException: libfranka: server closed connection`，控制栈崩溃。

**常见原因（按频率）**：

| 原因 | 解决 |
|---|---|
| 四元数约定错误 | 确认 deployer 用 scipy `[qx, qy, qz, qw]` |
| Target pose 累加 drift 触发 reflex | 使用 `target = actual + delta` |
| Bias 下观测和执行参考系不一致 | 策略用 `/getstate_true`，target 用 `/getstate` |
| Target 太远 / 速度过快 | 降低 ACTION_SCALE、加 rate limit |
| 同名节点二次注册 | `pkill -9 -f franka/roslaunch/roscore` 后重启 |

**恢复步骤**：
```bash
# RT PC
pkill -9 -f franka ; pkill -9 -f roslaunch ; pkill -9 -f roscore ; sleep 2
~/start_franka_server.sh
# Franka Desk 解锁关节 + 激活 FCI
```

### 5.2 `/pose` 返回 "Move command rejected, not possible in current mode"

**原因**：阻抗控制器没启动。Flask server 默认不自动启，需要手动 `/startimp`。

**解决**：deployer 启动时会自动调 `stopimp → clearerr → startimp`。如果仍失败：

```bash
curl -X POST http://192.168.100.1:5000/stopimp
curl -X POST http://192.168.100.1:5000/clearerr
curl -X POST http://192.168.100.1:5000/startimp
```

`/startimp` 需要**长超时**（≥15s），不是 2s。

### 5.3 机械臂"表现麻木"——/pose 返回 200 但不动

**症状**：`TGT_ERR` 持续增大，`ACTUAL_TCP` 不变。

**原因**：`/pose` 写入 topic 但没控制器消费（`/startimp` 超时了）。

**解决**：加大 `/startimp` 超时 + 启动后调 `/getstate_true` 验证。

### 5.4 机械臂"朝奇怪方向跑"

**原因**：Bias 下 target 和控制器 current 参考系不一致。

**解决**：确保 deployer 的 target 构造用 `/getstate`（biased）而不是 `/getstate_true`。

### 5.5 ROS 同名节点冲突

**症状**：`new node registered with same name`。

**原因**：前一次 libfranka 崩溃留下僵尸进程，新的 launch 和它冲突。

**解决**：`pkill -9` 后重启。

### 5.6 D455 `Frame didn't arrive within N000`

**症状**：`pipeline.wait_for_frames()` 超时。

**原因**：USB 口带宽/供电不够。

**解决**：
- 换用主板**后面板直出的 USB 3.0 口**（不是前面板延长线）
- 避开 USB Hub
- 一次 `hardware_reset()`：
  ```python
  import pyrealsense2 as rs
  rs.context().devices[0].hardware_reset()
  ```

### 5.7 ArUco 检测不到

**常见原因**：
- 字典错误（检查 `DICT_5X5_100` 匹配打印来源）
- Marker 尺寸参数和实际打印不符
- 光照反光

**调试**：
```bash
python scripts/real/debug_aruco.py  # 保存一帧原图 + 尝试所有字典
```

---

## 6. 关键设计决策

### 6.1 为什么 backup policy 观测用真值，目标用 biased？

- 策略在仿真里没见过 bias，喂它 biased 观测会分布外 → 决策乱
- 但 C++ 控制器内部用 biased q 算 current，target 必须同一帧才不会补幽灵误差

### 6.2 为什么 target = actual + delta 不累加？

- 累加模式下 target 可能比真实位置超前几十 cm
- Impedance 弹簧力 = K × (target - current) 爆炸
- libfranka 认为机械臂"疯了"，触发 reflex 切断连接

### 6.3 为什么 Minkowski 球而不是真 bbox-vs-bbox？

- 训练用的是点对点距离，observation 形状固定为 10D
- 框对框需要增加观测维度（+bbox 尺寸），意味着重训
- Minkowski 球是**数学等价的简化**：两点距离 = 两球表面间距
- 不改观测维度，不用重训，语义正确

### 6.4 为什么 R_GRIPPER = 0.06m？

- Franka Hand 标称 85×90×120mm
- 包围球半径 ≈ sqrt(85² + 90² + 120²)/2 ≈ 88mm
- 取 60mm 是一个保守折衷：既覆盖主体又不会太夸张

### 6.5 为什么用 SpaceMouse 当 task policy 占位？

- Task policy 未训练完成
- SpaceMouse 天然是 6D + gripper，和 action space 对齐
- 人作为"任务策略"能灵活产生各种场景测试 backup

---

## 7. 文件清单

### 仿真侧（已完成）

```
frrl/envs/panda_backup_policy_env.py   ← S1/S2 训练环境
configs/train_hil_sac_backup_s1.json   ← 训练配置
checkpoints/backup_policy_s1/          ← 训练好的 checkpoint
```

### 真机侧（本次部署）

```
frrl/vision/hand_detector.py           ← WiLoR + D455 手部检测
scripts/real/calibrate_cam_to_robot.py      ← 相机标定（SpaceMouse 驱动）
scripts/real/deploy_backup_policy.py        ← 主部署脚本
scripts/hw_check/test_d455.py                   ← 相机 smoke test
scripts/hw_check/test_hand_detector.py          ← 手部检测 smoke test
scripts/hw_check/test_rtpc_link.py              ← RT PC 网络 + 基础端点 smoke test
scripts/hw_check/test_spacemouse.py             ← SpaceMouse 驱动 smoke test
calibration_data/T_cam_to_robot.npy    ← 标定结果
```

### RT PC 侧

```
frrl/robot_servers/franka_server.py    ← Flask server（加了 bias 端点和 /getstate_true）
patches/serl_franka_controllers_bias_injection.patch ← C++ 阻抗控制器的 bias 补丁
```

---

## 8. 每次使用的最短流程

```bash
# RT PC
pkill -9 -f franka ; pkill -9 -f roslaunch ; pkill -9 -f roscore
~/start_franka_server.sh

# 浏览器访问 https://172.16.0.2 → 确认白灯 / 解锁关节 / 激活 FCI

# GPU 站
conda activate lerobot
cd ~/FR-RL-lerobot

# 验证链路（一次性）
python scripts/hw_check/test_rtpc_link.py
python scripts/hw_check/test_d455.py
python scripts/hw_check/test_hand_detector.py

# 部署
python scripts/real/deploy_backup_policy.py

# 做完清理
curl -X POST http://192.168.100.1:5000/clear_encoder_bias
curl -X POST http://192.168.100.1:5000/stopimp
```
