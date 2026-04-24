# SpaceMouse 遥操作部署与使用

在 GPU 工作站上通过 3Dconnexion SpaceMouse 遥操作 Franka Panda 真机。
SpaceMouse 6D 输入 → 按帧累加到目标位姿 → HTTP POST 到 RT PC 的 `franka_server.py` → libfranka 阻抗控制器执行。

---

## 1. 硬件

| 设备 | 接线 | 说明 |
|---|---|---|
| 3Dconnexion SpaceMouse Compact | USB 口 → GPU 工作站 | USB ID `256f:c635` |
| Franka Panda + 控制柜 | 以太网 → RT PC 的 Franka 直连口 | libfranka 1kHz 控制 |
| GPU 工作站 ↔ RT PC | 以太网（直连或交换机）| GPU `192.168.100.2/24`, RT PC `192.168.100.1/24` |

SpaceMouse 插在 **GPU 工作站**（不是 RT PC），因为 teleop 程序在 GPU 站上跑。

---

## 2. 系统依赖（GPU 工作站）

一次性安装，后续不用再动：

```bash
# HID 库（SpaceMouse 通过 HID 通信）
sudo apt update
sudo apt install -y libhidapi-dev libhidapi-libusb0 libhidapi-hidraw0

# 允许非 root 用户读 SpaceMouse HID 设备
sudo tee /etc/udev/rules.d/99-spacemouse.rules <<'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", MODE="0666"
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**udev 规则生效后，拔掉 SpaceMouse 再插上**（或重启），否则首次打开设备会报 permission denied。

---

## 3. Python 依赖（在 `lerobot` conda 环境里）

```bash
conda activate lerobot
pip install pyspacemouse easyhid
```

两个库的作用不同：

| 库 | 用途 | 被谁用 |
|---|---|---|
| `pyspacemouse`（pip 新版，device-object API）| 独立驱动烟雾测试 | `scripts/hw_check/test_spacemouse.py` |
| `easyhid` | 项目内 bundled `frrl/teleoperators/spacemouse/pyspacemouse.py` 的后端 | `frrl.teleoperators.spacemouse.SpaceMouseExpert`，实际 teleop 用这个 |

实际遥操作用的是项目内的 `SpaceMouseExpert`（多进程后台轮询 + 统一轴映射），pip 的 `pyspacemouse` 仅用于驱动自检。

---

## 4. 项目内相关文件

```
frrl/teleoperators/spacemouse/
├── __init__.py
├── configuration_spacemouse.py       # SpaceMouseConfig
├── pyspacemouse.py                    # bundled HID 读取（依赖 easyhid）
├── spacemouse_expert.py               # 多进程后台读取 → 共享内存
└── teleop_spacemouse.py               # Teleoperator 抽象实现（给 RL 用）

scripts/hw_check/
├── test_spacemouse.py                 # 独立驱动自检（不碰机器人）
└── test_rtpc_link.py                  # 包含 --teleop 模式
```

---

## 5. 首次配置后的自检流程

### 5.1 驱动自检（无机器人，零风险）

```bash
conda activate lerobot
cd ~/FR-RL-lerobot
python scripts/hw_check/test_spacemouse.py
```

期望输出：连续打印 6 个浮点 + buttons 数组。推动/旋转 SpaceMouse 球头应看到数值变化，按钮状态在 `[0,0]` 和 `[1,0]`/`[0,1]` 之间切换。

常见错误：
- `[FAIL] could not open SpaceMouse` → udev 规则没生效，拔插一次 USB
- `ModuleNotFoundError: pyspacemouse` → conda env 不对，检查 `which python`

### 5.2 网络和 Flask server 自检

**RT PC 上**启动 Flask server（建议用 tmux，防止关窗口挂掉）：

```bash
# 在 RT PC 上
tmux new -s franka
# tmux 里启动 franka_server.py （具体命令按你 RT PC 的部署而定）
# 启动后 Ctrl+B D 脱离
# 以后查看：tmux attach -t franka
```

**GPU 工作站上**验证 HTTP 链路：

```bash
conda activate lerobot
cd ~/FR-RL-lerobot
python scripts/hw_check/test_rtpc_link.py
```

期望看到一段 JSON，包含 `q`、`dq`、`pose`、`gripper_pos` 等字段。
报 `Connection refused` 就是 RT PC 上 Flask server 没起。

---

## 6. 遥操作

### 6.1 启动

```bash
conda activate lerobot
cd ~/FR-RL-lerobot
python scripts/hw_check/test_rtpc_link.py --teleop
```

默认参数（已调好的手感）：

| 参数 | 默认值 | 物理含义 |
|---|---|---|
| `--action-scale` | `0.015` | 球头推到底 = 15 cm/s 平移 |
| `--rotation-scale` | `0.035` | 球头扭到底 ≈ 20°/s 旋转 |
| `--rate-hz` | `10.0` | 控制循环 10 Hz |

需要更快/慢时加参数覆盖：

```bash
# 更快
python scripts/hw_check/test_rtpc_link.py --teleop --action-scale 0.020 --rotation-scale 0.045
# 更慢
python scripts/hw_check/test_rtpc_link.py --teleop --action-scale 0.010 --rotation-scale 0.025
```

### 6.2 按键映射

| SpaceMouse | 动作 |
|---|---|
| 球头平移（前/后、左/右、上/下）| TCP 平移增量 |
| 球头旋转（roll/pitch/yaw）| TCP 旋转增量（当前 quat 上右乘 delta）|
| 左按钮 `button[0]` | close_gripper |
| 右按钮 `button[1]` | open_gripper |
| Ctrl-C | 退出 teleop 循环 |

轴映射来自 `frrl/teleoperators/spacemouse/spacemouse_expert.py` 里的 `[-s.y, s.x, s.z, -s.roll, -s.pitch, -s.yaw]`——假设操作员坐在机器人正对面。如果坐姿不同导致推拉方向反，修改那里的符号即可（影响全局）。

### 6.3 运行时注意

- **没有 safety clip**：当前脚本不做笛卡尔边界裁剪，推久了会把 TCP 推出工作区或撞桌面。手始终放在急停按钮上。
- **启动瞬间无跳变**：脚本第一步 `getstate` 读当前位姿作为累加起点，所以启动时不会跳。
- **旋转会累积漂移**：长时间扭动后姿态会偏离初始值，松手不会回正，这是预期行为。
- **退出不会回 home**：Ctrl-C 只是停止循环，机器人停在最后的目标位姿。需要复位就单独跑 `python scripts/hw_check/test_rtpc_link.py --reset`（会走 `/jointreset`）。

---

## 7. 每次使用的最短流程

假设系统依赖、udev、pip 包都已装好：

```bash
# RT PC 上（保持 tmux 里 franka_server.py 一直跑）
tmux attach -t franka   # 或确认它还活着

# GPU 工作站上
conda activate lerobot
cd ~/FR-RL-lerobot
python scripts/hw_check/test_rtpc_link.py           # 1) 网络自检
python scripts/hw_check/test_rtpc_link.py --teleop  # 2) 进入遥操作
```

---

## 8. 故障排查对照表

| 现象 | 原因 | 处置 |
|---|---|---|
| `could not open SpaceMouse` / `permission denied` | udev 规则未生效 | 拔插 USB；`ls -l /dev/hidraw*` 应看到 `666` 权限 |
| `ModuleNotFoundError: pyspacemouse` / `easyhid` | 不在 lerobot env | `conda activate lerobot` |
| `Connection refused` to `192.168.100.1:5000` | RT PC Flask server 没起 | 登录 RT PC 重启 `franka_server.py`（用 tmux 保活）|
| ping 通但 curl refused | Flask 绑定到别的网卡 | 检查 `franka_server.py` 启动时的 `host` 参数 |
| SpaceMouse 有信号但机器人不动 | `/pose` 返回非 200 / 阻抗控制器未启 | 看 RT PC 终端 `franka_server.py` 的日志 |
| 推前 TCP 却往后走 | 坐姿和默认轴映射假设不符 | 修改 `spacemouse_expert.py` 的轴符号 |
| 动作抖动/跟不上 | 10 Hz 控制环在网络繁忙时超时 | 观察 teleop 循环是否有 `requests.Timeout`，降低 `--rate-hz` |

---

## 9. 与 RL 训练的关系（后续工作）

当前 `test_rtpc_link.py --teleop` 只是**自检用的直连脚本**，不经过 Gym env。真正的 RL 训练流程中：

```
SpaceMouseTeleop (frrl/teleoperators/spacemouse/teleop_spacemouse.py)
    ↓ get_action() -> 7D ndarray
Actor (frrl/rl/core/actor.py)
    ↓ 结合策略动作，干预时覆盖
FrankaRealEnv (待实现)
    ↓ apply_action() -> HTTP POST /pose
franka_server.py → libfranka
```

也就是说现在验证的 SpaceMouse 驱动 + `SpaceMouseExpert` 类是**完全复用**的，后续只需要把它接到 `FrankaRealEnv.step()` 的干预路径里。
