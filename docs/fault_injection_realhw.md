# 真机编码器偏差注入：实现、验证与使用

本文是 [`fault_injection_architecture.md`](fault_injection_architecture.md) 的真机落地文档。
架构设计（为什么选 B+D 而不是 A/C/E）请看前者，本文覆盖**代码改动、编译、
真机验证步骤和训练/评估/部署时的实用配置**。

**状态**：B+D 注入管线已在真实 Franka Panda 上端到端验证（2026-04-15）。
机械臂在 0.1 rad J1 bias 下产生预期的 ~7cm 水平扫动，清除后平滑复位，
`/getstate` 返回的 biased 观测与注入值一致。

---

## 1. 架构回顾

单条数据流，所有路径都在一张图里：

```
 GPU 侧 FrankaRealEnv.reset()
     │
     └─ bias_injector.on_episode_start() → 采样 bias (7D)
     │
     └─ _set_encoder_bias(bias)
            │
            ↓ HTTP POST /set_encoder_bias   (JSON: {"bias": [7 floats]})
  ┌──────────────────────────────────────────────────────────┐
  │ RT PC: franka_server.py                                   │
  │    publish_encoder_bias(bias)                             │
  │        → rospy.Publisher("/encoder_bias",                 │
  │                          Float64MultiArray, latch=True)   │
  └──────────────────────────────────────────────────────────┘
            │
            ↓ ROS topic (latched)
  ┌──────────────────────────────────────────────────────────┐
  │ C++ CartesianImpedanceController (1 kHz)                  │
  │                                                            │
  │  encoderBiasCallback (非 RT)                              │
  │     → RealtimeBuffer<std::array<double,7>> (RT-safe 写)   │
  │                                                            │
  │  update() {                                                │
  │    bias = buffer.readFromRT();    // ← 注入点 B           │
  │    q_biased = state.q + bias;                             │
  │    pose_biased = getPose(q_biased, F_T_EE, EE_T_K);       │
  │    J_biased    = getZeroJacobian(q_biased, F_T_EE, EE_T_K);│
  │    tau = J^T * K * (pose_biased - pose_d) + ...           │
  │    setCommand(tau)    // 真实 torque，真实物理运动         │
  │                                                            │
  │    publisher_biased_state.publish(q_biased, pose_biased);  │
  │  }                                                         │
  └──────────────────────────────────────────────────────────┘
            │
            ↓ ROS topic /cartesian_impedance_controller/biased_state
  ┌──────────────────────────────────────────────────────────┐
  │ RT PC: franka_server.py                                   │
  │    _set_biased_state() → self.q_biased / self.pos_biased  │
  │                                                            │
  │    @route /getstate returns biased q/pose  ← 注入点 D      │
  └──────────────────────────────────────────────────────────┘
            │
            ↓ HTTP GET /getstate
     FrankaRealEnv._update_currpos() → obs['agent_pos'] 含 biased q/tcp
     RL policy 看到带偏差的观测
```

**关键不变量**：C++ 控制器是**唯一的 bias 应用点**。`biased_state` topic 由
控制器发，franka_server.py 只转发给 HTTP 客户端，不在 Python 侧重算 FK。
这保证了控制回路用的 q_biased 和观测返回的 q_biased 数值**完全一致**，
不会因为 Python 侧近似 FK 引入漂移。

---

## 2. 代码改动清单

改动分布在 FR-RL-lerobot（直接纳入 git）和 `serl_ws/` 上游 catkin 包（patch 文件）两处：

| 位置 | 文件 | 性质 | 版本控制 |
|---|---|---|---|
| `~/FR-RL-lerobot/frrl/robot_servers/` | `franka_server.py` | **新增（从 hil-serl 搬入）** + B+D 改动 | ✅ git |
| `~/FR-RL-lerobot/frrl/robot_servers/` | `franka_gripper_server.py` / `robotiq_gripper_server.py` / `gripper_server.py` / `__init__.py` | **新增（从 hil-serl 搬入）** | ✅ git |
| `~/FR-RL-lerobot/patches/` | `serl_franka_controllers_bias_injection.patch` | **新增 patch 文件** | ✅ git |
| `~/FR-RL-lerobot/frrl/envs/` | `franka_real_env.py` / `franka_real_config.py` / `fault_injection.py` | 无改动（hook 已有） | ✅ git |
| `~/serl_ws/src/serl_franka_controllers/` | `msg/BiasedState.msg` | **新增**（patch 内） | ⚠️ 本地 |
| `~/serl_ws/src/serl_franka_controllers/` | `CMakeLists.txt` / `package.xml` | 修改（patch 内） | ⚠️ 本地 |
| `~/serl_ws/src/serl_franka_controllers/` | `include/.../cartesian_impedance_controller.h` | 修改（patch 内） | ⚠️ 本地 |
| `~/serl_ws/src/serl_franka_controllers/` | `src/cartesian_impedance_controller.cpp` | 修改（patch 内） | ⚠️ 本地 |

### 为什么 franka_server.py 进 FR-RL-lerobot，controller 却用 patch

`franka_server.py` 是纯 Python，只在 RT PC 上跑，逻辑上就是本项目的一部分；
以前在 `hil-serl` 下纯粹是历史遗留（项目最早只移植了 `franka_env.py` 这一侧）。
搬进 `frrl/robot_servers/` 之后 hil-serl 完全不再参与运行时，调用路径走
`python -m frrl.robot_servers.franka_server`，相对导入正常工作。

`serl_franka_controllers` 是 **catkin ROS 包**，必须住在 ROS workspace 里才能被
`catkin_make` 构建和 `roslaunch` 加载，结构上没法直接塞进 Python package。
所以它的改动做成 [`patches/serl_franka_controllers_bias_injection.patch`](../patches/serl_franka_controllers_bias_injection.patch)
存进仓库，附一个 [`patches/README.md`](../patches/README.md) 记录上游 base commit
和 `git apply` 流程 —— 任何人 clone 本仓库后都能从零复现 C++ 控制器的修改。

### 2.1 `BiasedState.msg`

```
float64[7] q_biased
float64[16] O_T_EE_biased
float64[7] bias
```

`O_T_EE_biased` 是 column-major 的 4x4 齐次矩阵，和 `franka_state` msg 的
`O_T_EE` 字段格式相同，Python 侧可以直接复用原有的 `reshape(4,4).T` 解析逻辑。

### 2.2 C++ 控制器（注入点 B）

Header 新增：
```cpp
#include <std_msgs/Float64MultiArray.h>
#include <realtime_tools/realtime_buffer.h>
#include <serl_franka_controllers/BiasedState.h>

// in class CartesianImpedanceController:
ros::Subscriber sub_encoder_bias_;
realtime_tools::RealtimeBuffer<std::array<double, 7>> encoder_bias_buffer_;
realtime_tools::RealtimePublisher<serl_franka_controllers::BiasedState>
    publisher_biased_state_;
void encoderBiasCallback(const std_msgs::Float64MultiArray::ConstPtr& msg);
```

`init()` 里注册 publisher、初始化 buffer 为零、订阅 `/encoder_bias`。

`update()` 的关键替换（摘自 `cartesian_impedance_controller.cpp:134-194`）：

```cpp
franka::RobotState robot_state = state_handle_->getRobotState();
std::array<double, 7> coriolis_array = model_handle_->getCoriolis();

// RT-safe 读最新 bias
const std::array<double, 7>& bias = *encoder_bias_buffer_.readFromRT();

// 注入
std::array<double, 7> q_biased_array = robot_state.q;
for (size_t i = 0; i < 7; ++i) q_biased_array[i] += bias[i];

// 用 biased q 算 FK 和 Jacobian（franka_hw 提供的显式 q 重载）
std::array<double, 16> pose_biased_array = model_handle_->getPose(
    franka::Frame::kEndEffector, q_biased_array,
    robot_state.F_T_EE, robot_state.EE_T_K);
jacobian_array = model_handle_->getZeroJacobian(
    franka::Frame::kEndEffector, q_biased_array,
    robot_state.F_T_EE, robot_state.EE_T_K);

// Eigen Map 指向 biased 数据
Eigen::Map<Eigen::Matrix<double, 7, 1>> q(q_biased_array.data());
Eigen::Affine3d transform(Eigen::Matrix4d::Map(pose_biased_array.data()));
// ...后续控制律（tau_task、nullspace、coriolis、clip、torque-rate saturation）
//    全部用 biased 值计算，无需改动
```

发 biased_state topic：
```cpp
if (publisher_biased_state_.trylock()) {
  for (size_t i = 0; i < 7; ++i) {
    publisher_biased_state_.msg_.q_biased[i] = q_biased_array[i];
    publisher_biased_state_.msg_.bias[i] = bias[i];
  }
  for (size_t i = 0; i < 16; ++i) {
    publisher_biased_state_.msg_.O_T_EE_biased[i] = pose_biased_array[i];
  }
  publisher_biased_state_.unlockAndPublish();
}
```

callback 本身：
```cpp
void CartesianImpedanceController::encoderBiasCallback(
    const std_msgs::Float64MultiArray::ConstPtr& msg) {
  if (msg->data.size() != 7) return;  // 忽略非法长度
  std::array<double, 7> new_bias;
  for (size_t i = 0; i < 7; ++i) new_bias[i] = msg->data[i];
  encoder_bias_buffer_.writeFromNonRT(new_bias);
}
```

### 2.3 franka_server.py（注入点 D）

`frrl/robot_servers/franka_server.py` 相比搬入前的 hil-serl 版本新增：

- 新 import：`from serl_franka_controllers.msg import BiasedState`、
  `from std_msgs.msg import Float64MultiArray`
- `FrankaServer.__init__` 新增字段：`current_bias`, `q_biased`, `pos_biased`,
  `_biased_state_received`
- 新 publisher `self.bias_pub = rospy.Publisher("/encoder_bias",
  Float64MultiArray, queue_size=1, latch=True)`
- 新 subscriber `/cartesian_impedance_controller/biased_state → _set_biased_state`
- 启动时 `publish_encoder_bias(np.zeros(7))` 主动清零，覆盖任何 latched 残留
- `_set_currpos` 保持不变，仍写 `self.q` / `self.pos`（真值），但新增 fallback
  `if not self._biased_state_received: self.q_biased = self.q.copy()`
- `_set_biased_state` 把 BiasedState topic 的字段写到 `self.q_biased` / `self.pos_biased`
- `/getstate` 返回的 `pose`/`q` 字段改为 `pos_biased`/`q_biased`，并新增 `bias` 字段
- 新 HTTP 路由：
  - `POST /set_encoder_bias` → body `{"bias": [7 floats]}` → `publish_encoder_bias`
  - `POST /clear_encoder_bias` → `publish_encoder_bias(zeros)`
  - `POST /get_encoder_bias` → 返回 `{"bias": [...]}`

搬入时同步把 lazy gripper server import 从原先的
`from robot_servers.franka_gripper_server import FrankaGripperServer` 改成
相对导入 `from .franka_gripper_server import FrankaGripperServer`，以配合新的
`python -m frrl.robot_servers.franka_server` 启动方式。

**为什么 `self.q` 仍保留真值**：`reset_joint()` 里的等待循环
`np.allclose(target - self.q, 0)` 需要真实关节角作比较基准，如果写成 biased
值，joint reset 在有 bias 时永远收敛不了。

### 2.4 GPU 侧未改动

`franka_real_env.py:134-143` 的 reset 注入 hook 和 `:396-402` 的
`_set_encoder_bias` 方法在本轮工作之前就已存在，**路由名和 JSON 格式完全吻合**
新增的 franka_server 接口。GPU 侧仅需在构造 `FrankaRealConfig` 时：
1. 把 `server_url` 从默认错误值 `http://192.168.1.1:5000/` 改成
   `http://192.168.100.1:5000/`
2. 把 `encoder_bias_config` 从 `None` 改成一个 `EncoderBiasConfig` 实例

---

## 3. 编译与启动

**首次配置**（新机部署或 workspace 被清理后）：
```bash
# 1. 拿到上游 controller 源码并打 patch
cd ~/serl_ws/src
git clone https://github.com/rail-berkeley/serl_franka_controllers.git
cd serl_franka_controllers
git checkout 1f140ef0d8e3fc443569c193d3ede1856e50d521
git apply ~/FR-RL-lerobot/patches/serl_franka_controllers_bias_injection.patch

# 2. 编译
cd ~/serl_ws
source /opt/ros/noetic/setup.bash
catkin_make --only-pkg-with-deps serl_franka_controllers
```

**日常启动**（每次都要**彻底重启**，否则旧的 controller .so 还在内存里）：
```bash
~/kill_franka_server.sh             # 清掉所有 ros/franka 进程
source ~/serl_ws/devel/setup.bash   # 关键：让 BiasedState 的 Python 绑定可用
# Desk: Error Recovery → Unlock Joints → 激活 FCI
~/start_franka_server.sh             # cd ~/FR-RL-lerobot && python -m frrl.robot_servers.franka_server
```

**启动健康检查**（另开终端）：
```bash
# biased_state topic 应在 ~1 kHz
rostopic hz /cartesian_impedance_controller/biased_state
# baseline：bias 全零、q/pose 等于真值
curl -s -X POST http://127.0.0.1:5000/getstate | python3 -c "
import sys, json
s = json.load(sys.stdin)
print('bias:', s['bias'])
print('q   :', [round(x,4) for x in s['q']])
print('pose:', [round(x,4) for x in s['pose'][:3]])"
```

---

## 4. 真机验证流程

### 4.1 单点固定 bias（curl 直接打）

```bash
# 注入 J1 0.1 rad
curl -s -X POST http://127.0.0.1:5000/set_encoder_bias \
  -H "Content-Type: application/json" \
  -d '{"bias": [0.1, 0, 0, 0, 0, 0, 0]}'

# 等机械臂稳定（~1-2 秒）后读状态
curl -s -X POST http://127.0.0.1:5000/getstate | python3 -c "
import sys, json
s = json.load(sys.stdin)
print('bias:', s['bias'])
print('q[0]:', round(s['q'][0], 4))
print('pose:', [round(x,4) for x in s['pose'][:3]])"

# 清除 bias
curl -s -X POST http://127.0.0.1:5000/clear_encoder_bias
```

### 4.2 Random 模式（交互式脚本）

仓库提供 `scripts/real/test_bias_random_realhw.py`，从 `[0.1, 0.25]` rad 范围均匀
采样若干 episode，每次注入前暂停让操作者决定是否继续，脚本退出（含 Ctrl+C）
时**总会**清 bias。

```bash
python3 scripts/real/test_bias_random_realhw.py
```

主要参数在脚本顶部：
```python
N_EPISODES     = 5
TARGET_JOINT   = 0               # Joint 1 (0-indexed)
BIAS_RANGE     = (0.1, 0.25)     # rad
SETTLE_AFTER_INJECT = 2.5        # 注入后等待稳定
SETTLE_AFTER_CLEAR  = 2.5        # 清除后等待复位
```

### 4.3 端到端（从 GPU 侧 `FrankaRealEnv`）

在 GPU 机上：
```bash
# 先 HTTP 链路自检
curl -X POST http://192.168.100.1:5000/getstate

# 再最小 env 验证（mock 相机模式）
python3 - <<'PY'
from frrl.fault_injection import EncoderBiasConfig
from frrl.envs.franka_real_config import FrankaRealConfig
from frrl.envs.franka_real_env import FrankaRealEnv

cfg = FrankaRealConfig(
    server_url="http://192.168.100.1:5000/",
    encoder_bias_config=EncoderBiasConfig(
        enable=True, error_probability=1.0, target_joints=[0],
        bias_mode='fixed', fixed_bias_value=0.1,
    ),
)
env = FrankaRealEnv(cfg)
obs, _ = env.reset()
print("injector bias:", env.bias_injector.current_bias)
print("q in obs (first 7 of agent_pos):", obs['agent_pos'][:7])
env.close()
PY
```

观察到的真机行为：
1. `reset()` 先调 `_go_to_reset` → 机械臂插值到 `reset_pose`
2. 随后 `bias_injector.on_episode_start()` + `_set_encoder_bias([0.1, 0, ..., 0])`
3. C++ controller 瞬时换 bias，机械臂在 20 N 恒力下**平滑滑动 ~7 cm**
4. `_update_currpos()` 拿到 biased `pose`/`q`，写进 obs

这条链路在 2026-04-15 完成端到端验证。

---

## 5. 物理行为分析

### 5.1 机械臂会动，不是"待在错误的位置"

**常见误解**：既然 bias 是"编码器误差"，机械臂应该待在原处，只是报错的位姿对不上。

**实际情况**：机械臂会**真实移动**，因为 controller 是 closed-loop。推导：

- 注入前稳态：`position_d_ = FK(q_true_initial)`（静止）
- 注入 bias：controller 看到 `position_biased = FK(q_true_initial + bias)`，这个值
  和原本的 `position_d_` 相差 ≈ `J × bias`
- 误差非零 → 力矩非零 → 物理 q_true 开始改变
- 稳态条件：`FK(q_true_new + bias) = position_d_` ⟹ `q_true_new ≈ q_true_initial - bias`
- **真实末端位置偏移 ≈ -J × bias**，和 bias 符号相反

这正是编码器误差的物理：机械臂以为自己在对的地方，实际偏了。不移动到错误位置
就没有任何物理意义 —— 仅仅改观测而不改控制等价于"假 bias"，策略只会学到直接
忽略观测。

### 5.2 `clip` 不是"位置限幅"，是"力限幅"

`cartesian_impedance_controller.cpp:154-157` 的 error clip：
```cpp
error_.head(3) << position - position_d_;
for (int i = 0; i < 3; i++) {
  error_(i) = std::min(std::max(error_(i), translational_clip_min_(i)),
                       translational_clip_max_(i));
}
```
默认 `translational_clip = 0.01 m`（每轴）、`rotational_clip = 0.05 rad`。

- **作用**：把单步进入控制律的 error 硬截到 ±1cm/±0.05rad。
  力 = K × error_clipped ≤ 2000 × 0.01 = **20 N**。
- **不改变最终平衡位置**：机械臂在 20 N 恒力下匀速滑动，直到 biased 位置
  回到 `position_d_` 的 1cm 邻域内，再切换成线性弹簧律稳定。最终偏移量
  仍然是 ~7cm（由 bias × J 决定），**和 clip 无关**。
- **意义**：没有 clip 的话，7cm error 会产生 140 N，作用在 ~5kg 臂上加速度
  ~28 m/s²，几十毫秒就危险。有 clip 后机械臂像在糖浆里游泳。

### 5.3 偏移量速查

典型 Panda ready pose 下（`q ≈ [0, -π/4, 0, -3π/4, 0, π/2, π/4]`）：

| 单关节 bias | 真实末端偏移 | 主导方向 |
|---|---|---|
| J1 0.01 rad | ~0.7 cm | 水平绕 base Z |
| J1 0.10 rad | **~7 cm** | 水平绕 base Z |
| J1 0.25 rad | **~17 cm** ⚠️ | 水平绕 base Z |
| J4 0.10 rad | ~7 cm | 俯仰 + 前后 |
| J7 0.10 rad | < 1 cm | 末端自旋（位置几乎不变） |

⚠️ 0.25 rad on J1 会让工作空间被"挤"掉 ~17cm，与 `abs_pose_limit` 可能冲突。
若训练时用 `bias_range=(0.1, 0.25)`，标定 `abs_pose_limit` 时要预留相应缓冲。

### 5.4 注入瞬时的观感

- **不是 snap**：clip + `delta_tau_max_=1 N·m/step` 的 torque rate 限制让运动平滑
- **持续时间**：对 0.1 rad on J1，约 0.5~1.5 秒滑到位
- **声音**：应该只有伺服电机的低频嗡嗡，不应有咔哒或异响
- **清除时**：等幅反向滑动一次

---

## 6. 训练 / 评估 / 部署 配置

### 6.1 配置入口：`FrankaRealConfig.encoder_bias_config`

注入由 `FrankaRealEnv.reset()` 驱动，每个 episode 开始时调用
`bias_injector.on_episode_start()` 采样新 bias，然后 `_set_encoder_bias()` 发给
RT PC。同一个 episode 内 bias 保持不变（符合"编码器校准误差在一次运行内是常量"
的物理语义）。

### 6.2 三种典型配置

**（a）训练：random 分布（覆盖 0 到最大）**
```python
EncoderBiasConfig(
    enable=True,
    error_probability=1.0,        # 100% 注入；设 0.7 则 30% episode 零 bias
    target_joints=[0],            # Joint 1（可改为 [3] 等其他关节）
    bias_mode='random_uniform',
    bias_range=(0.0, 0.25),       # 包含 0 让策略也学会无 bias 情形
)
```

**（b）评估：固定单值扫描（鲁棒性曲线）**
```python
EncoderBiasConfig(
    enable=True, error_probability=1.0,
    target_joints=[0], bias_mode='fixed',
    fixed_bias_value=0.10,        # 扫 [0, 0.05, 0.10, 0.15, 0.20, 0.25]
)
```
每个点跑 N 个 episode，统计 success rate。仿真版已有
`scripts/sim/eval_bias_curve.py`，真机版需要 fork 一份把 env 换成
`FrankaRealEnv`。

**（c）正式部署：不注入**
```python
FrankaRealConfig(
    server_url="http://192.168.100.1:5000/",
    encoder_bias_config=None,      # 关闭注入，依赖真机真实编码器
)
```
franka_server 启动时主动 `publish_encoder_bias(zeros)`，即使 env 不设也
不会残留偏差。

### 6.3 训练分布 vs 部署分布

| 训练 | 部署 | 结果 |
|---|---|---|
| `random(0, 0.25)` | `fixed 0.1` | ✅ 分布内 |
| `random(0, 0.25)` | `fixed 0.3` | ⚠️ 超出训练范围 |
| `fixed 0.1` | `fixed 0.1` | ✅ 完美匹配但过拟合 |
| `fixed 0.1` | `random(0, 0.25)` | ❌ 训练没见过其他值 |
| `random(0, 0.25)` | 无注入 | ✅ 训练包含 0 即可 |

实务推荐：训练用 `random_uniform(0.0, 0.25)` 包含零，评估扫固定点画曲线，
交付部署关闭注入。

### 6.4 从 YAML 加载

`EncoderBiasConfig.from_yaml` 已实现，配置文件示例：
```yaml
# configs/bias_j1_random.yaml
enable: true
error_probability: 1.0
target_joints: [0]
bias_mode: random_uniform
bias_range: [0.0, 0.25]
```
```python
from frrl.fault_injection import EncoderBiasConfig
bias_cfg = EncoderBiasConfig.from_yaml("configs/bias_j1_random.yaml")
```

---

## 7. 安全检查清单（注入前）

- [ ] Desk FCI 图标是蓝色（断电后需要手动重新激活）
- [ ] `rostopic hz /cartesian_impedance_controller/biased_state` ≈ 1 kHz
- [ ] `curl /getstate` 返回的 `bias` 全零（baseline 正确）
- [ ] 目视确认机械臂**可能扫动方向**上有 ≥ 10 cm 空域（对 J1 bias，是水平圆弧）
- [ ] 当前 `q[i]` 离关节限位 > `|预期 bias|` + 0.2 rad 安全余量
- [ ] 当前末端离 `abs_pose_limit` 边界 > 预期偏移 + 2 cm
- [ ] E-stop 按钮在手边，已复位
- [ ] `~/kill_franka_server.sh` 可用（上一次重启后未被覆盖）

---

## 8. 故障排除

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `curl /set_encoder_bias` 404 | franka_server 是旧版本（没有这个 endpoint） | `kill_franka_server.sh` → `source devel/setup.bash` → `start_franka_server.sh` |
| `rostopic hz biased_state` 没有输出 | C++ controller 是旧 .so 没重编 | `catkin_make` + 重启 server |
| `ImportError: BiasedState` 在 franka_server 启动时 | `source devel/setup.bash` 被跳过 | 先 source 再 start |
| 注入后 `q[0]` 没变 | bias publisher 没生效 / latched msg 丢了 | `curl /get_encoder_bias` 确认，再重注一次 |
| 注入后机械臂不动只抖 | C++ controller 未用新 .so | `ps aux | grep franka_control` 看时间戳，彻底 kill 重启 |
| 机械臂扫动幅度远大于预测 | clip 被改过（`translational_clip` > 0.01） | 检查 `rosparam get /cartesian_impedance_controller/dynamic_reconfigure_compliance_param_node/translational_clip_x` |
| `communication_constraints_violation` 刷屏 | RT 内核抖动 / 非 RT 动作 | Wi-Fi 关闭 / BIOS HT/C-states / 检查 kill_franka_server 是否彻底 |
| 清 bias 后机械臂不回到原位 | position_d_ 被别人发的 `/pose` 改过 | 正常现象，controller 只追当前 position_d_ |

---

## 9. 参考

- [`fault_injection_architecture.md`](fault_injection_architecture.md) — 为什么选
  B+D 而不是 A/C/E 的设计讨论
- [`rt_pc_runbook.md`](rt_pc_runbook.md) — RT PC 开机到 server 启动的完整流程
- [`real_robot_deployment_plan.md`](real_robot_deployment_plan.md) — 两机系统
  整体拓扑和分工
- `frrl/fault_injection.py` — `EncoderBiasInjector` 实现和 YAML schema
- `scripts/real/test_bias_random_realhw.py` — 本文 §4.2 引用的交互式 random 验证脚本
- `scripts/sim/eval_bias_curve.py` — 仿真版鲁棒性曲线评估（真机版待 fork）
