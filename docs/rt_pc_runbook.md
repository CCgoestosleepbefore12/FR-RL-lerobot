# RT PC 启动运行手册

这台笔记本作为 Franka Panda 的实时控制 PC，负责跑 `franka_server.py` (Flask HTTP :5000)，给 GPU 训练机上的 `FrankaRealEnv` 提供远程控制接口。

## 硬件拓扑

```
[Franka 控制柜] ──网线── [笔记本自带以太网 enp4s0] 172.16.0.1/24
                                                    ↕ 172.16.0.2 (Franka)

[GPU 训练机]   ──网线── [笔记本 USB 网卡 enx...]   192.168.100.1/24
                                                    ↕ 192.168.100.2 (GPU)
```

## 软件栈

| 组件 | 版本 / 路径 |
|---|---|
| 内核 | Linux 5.15-rt83 (PREEMPT_RT) |
| ROS | Noetic |
| libfranka | 0.9.1 (本地源码编译, `/home/yuliang/libfranka/build`) |
| franka_ros | 0.9.1 (源码编译, `~/serl_ws/src/franka_ros`) |
| serl_franka_controllers | 源码编译, `~/serl_ws/src/serl_franka_controllers` |
| franka_server.py | `~/hil-serl/serl_robot_infra/robot_servers/franka_server.py` |
| Conda env | `frankaserver` (Python 3.8) |

固件要求：Panda system version ≥ 4.2.1

---

## 开机启动流程

### 物理检查

- [ ] 笔记本自带以太网口 → 网线 → Franka 控制柜 **LAN 口**（不是 shop 口）
- [ ] 笔记本 USB 网卡 → 网线 → GPU 机以太网口
- [ ] Franka 控制柜电源开
- [ ] E-stop 在手边且已复位

### Step 1：网络自检

```bash
ip -4 addr show enp4s0          # 应显示 172.16.0.1/24
ip -4 addr show enx00e04c1c0c08 # 应显示 192.168.100.1/24
ping -c 3 172.16.0.2            # 应通
```

**不通怎么办**：

- `enp4s0` 没 IP → `nmcli con up franka`
- USB 网卡没 IP → `nmcli con up 027d068b-6ed9-4851-8c44-35e0c80627c4`
- `ip addr show | grep 172.16` 里 USB 网卡又抢了 172.16.0.x → 回去改 NM 配置到 192.168.100.0/24

### Step 2：激活 FCI（**每次上电必做**）

1. 浏览器打开 `https://172.16.0.2` 登录 Desk
2. 有红色错误 → 点 **Error Recovery**（或按机器人基座蓝色按钮）
3. 机器人 **unlock joints**（白灯常亮）
4. 右上角 **FCI 图标 → 点击激活**（图标变蓝）
5. Desk 提示 `FCI active. Desk features are limited.` = 成功

⚠️ 断电重启后 FCI 自动关闭，每次开机必须手动激活。

### Step 3：启动 franka_server

```bash
~/start_franka_server.sh
```

脚本内部会自动：

1. 激活 `frankaserver` conda env
2. `source /opt/ros/noetic/setup.bash`
3. `source ~/serl_ws/devel/setup.bash`
4. `cd ~/hil-serl/serl_robot_infra/robot_servers`
5. 启动 `python franka_server.py --robot_ip=172.16.0.2 --gripper_type=Franka --flask_url=0.0.0.0`

**成功日志的关键行**：

```
process[master]: started with pid [...]
process[franka_control-X]: started
Loaded controllers: cartesian_impedance_controller
 * Serving Flask app 'franka_server'
 * Running on http://192.168.100.1:5000
```

之后终端阻塞不动，**不要按 Ctrl+C**。机器人进入 impedance 保持状态（手推会软绵绵地让开，松手回弹）。

### Step 4：自检

另开终端：

```bash
# 本机读状态
curl -X POST http://127.0.0.1:5000/getstate

# GPU 机远程（在 GPU 机上跑）
curl -X POST http://192.168.100.1:5000/getstate
```

返回 JSON 包含 `q`, `dq`, `pose`, `gripper_pos`, `jacobian` 等字段即成功。

---

## 正常关机

1. 终端 **Ctrl+C**（不要 Ctrl+Z！）停 `franka_server.py`
2. `~/kill_franka_server.sh` 确认无残留进程
3. （可选）Desk 里 Deactivate FCI → lock joints → 关 Franka 控制柜

---

## 异常处理速查

### A. Desk 进不去（https://172.16.0.2 无响应）

```bash
ping -c 3 172.16.0.2
```

- 不通 → 网线接错口（Franka 必须接笔记本**自带**以太网口）
- 或 USB 网卡 IP 被设成了 172.16.0.x（和 Franka 冲突，回去改 NM 配置）

### B. 启动时报 `Connection to FCI refused`

→ Desk 里 FCI 没激活，回 Step 2。

### C. 启动时报 `Move command rejected ... Reflex`

→ 机器人在 Reflex 保护态。Desk 里 Error Recovery 或按机器人基座蓝色物理按钮清错。

### D. 启动时报 `roscore already running`

→ 上次没关干净：

```bash
~/kill_franka_server.sh
~/start_franka_server.sh
```

### E. 终端 Ctrl+Z 挂起（进程变成 T 状态）

```bash
# 新终端执行
~/kill_franka_server.sh
```

### F. 任何崩溃 / 一键重启

```bash
~/kill_franka_server.sh
# Desk: Error Recovery + 重新激活 FCI
~/start_franka_server.sh
```

### G. `communication_constraints_violation` 持续刷屏

→ RT kernel 抖动。检查：

- `uname -r` 输出里必须有 `rt`
- `ulimit -r` ≥ 90
- Wi-Fi 是否关闭（`nmcli radio wifi off`，无线网卡中断会干扰 RT 循环）
- BIOS 里是否关闭了 Hyper-Threading / C-States / SpeedStep

---

## 关键参数速查

| 项目 | 值 |
|---|---|
| Franka IP | `172.16.0.2` |
| RT PC Franka 口 IP | `172.16.0.1` |
| RT PC GPU 口 IP | `192.168.100.1` |
| GPU 机 IP | `192.168.100.2` |
| Franka Desk | `https://172.16.0.2` |
| Server URL | `http://192.168.100.1:5000` |
| 启动脚本 | `~/start_franka_server.sh` |
| 清场脚本 | `~/kill_franka_server.sh` |
| hil-serl 源码 | `~/hil-serl` |
| catkin workspace | `~/serl_ws` |
| Conda env | `frankaserver` (Python 3.8) |

---

## 启动脚本内容参考

**`~/start_franka_server.sh`**：

```bash
#!/bin/bash
set -e
source /home/yuliang/miniconda3/etc/profile.d/conda.sh
conda activate frankaserver
source /opt/ros/noetic/setup.bash
source /home/yuliang/serl_ws/devel/setup.bash
cd /home/yuliang/hil-serl/serl_robot_infra/robot_servers
exec python franka_server.py \
    --robot_ip=172.16.0.2 \
    --gripper_type=Franka \
    --flask_url=0.0.0.0
```

**`~/kill_franka_server.sh`**：

```bash
#!/bin/bash
pkill -9 -f franka_server       || true
pkill -9 -f roslaunch           || true
pkill -9 -f rosmaster           || true
pkill -9 -f franka_control_node || true
pkill -9 -f franka_gripper_node || true
pkill -9 -f franka_state_controller || true
pkill -9 -f controller_spawner  || true
sleep 1
echo "=== Surviving processes ==="
ps aux | grep -E "franka_server|ros(master|launch)" | grep -v grep || echo "(clean)"
```
