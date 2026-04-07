# 编码器偏差仿真设计：原理、实现与合理性分析

## 1. 真实场景：编码器偏差是怎么产生的

工业机器人的关节编码器负责测量关节角度。偏差来源：

| 原因 | 类型 | 典型大小 |
|------|------|----------|
| 出厂校准不精确 | 固定偏差 | 0.01-0.1 rad |
| 温度漂移（长时间运行） | 缓慢变化 | 0.05-0.2 rad |
| 更换电机/减速器后未重新校准 | 固定偏差 | 0.1-0.5 rad |
| 碰撞后编码器错位 | 突变偏差 | 0.1-1.0 rad |
| 电磁干扰 | 随机噪声 | 0.001-0.01 rad |

本项目关注的是**固定/随机偏差**（前三种），不涉及噪声和漂移。

---

## 2. 真实机器人上偏差的完整因果链

```
编码器物理偏差
    │
    ▼
编码器输出: q_measured = q_true + bias
    │
    ├──→ 底层关节PD控制器
    │       控制目标: q_desired = q_home
    │       反馈: q_measured = q_true + bias
    │       PD误差: e = q_desired - q_measured = q_home - q_true - bias
    │       稳态条件: e = 0 → q_true = q_home - bias
    │       ┌─────────────────────────────────────────┐
    │       │ 结果1: 关节实际停在 q_home - bias        │
    │       │ 但控制器以为停在了 q_home                 │
    │       │ → 末端真实位置偏移                        │
    │       └─────────────────────────────────────────┘
    │
    ├──→ 上层OSC（操作空间控制器）
    │       读取: q_measured = q_true + bias
    │       计算Jacobian: J(q_measured) ≠ J(q_true)
    │       计算FK: x_current = FK(q_measured) ≠ FK(q_true)
    │       计算误差: e_x = x_target - FK(q_measured)    ← 错误的
    │       计算力矩: tau = J(q_measured)^T × Mx × e_x   ← 方向偏了
    │       ┌─────────────────────────────────────────┐
    │       │ 结果2: 力矩方向不准                      │
    │       │ 末端运动轨迹偏离指令                      │
    │       │ → 执行偏差                               │
    │       └─────────────────────────────────────────┘
    │
    └──→ 上层策略/观测系统
            读取: q_measured → FK(q_measured) → 上报末端位置
            ┌─────────────────────────────────────────┐
            │ 结果3: 策略收到错误的关节角度和末端位置    │
            │ 策略基于错误感知做决策                     │
            │ → 感知偏差                               │
            └─────────────────────────────────────────┘

总结: 一个编码器偏差 → 三重影响
  ① 初始位置偏移（关节PD控制器被骗）
  ② 执行偏差（OSC的Jacobian和FK被骗）
  ③ 感知偏差（策略收到错误观测）
```

---

## 3. 仿真中如何模拟每一层影响

### 3.1 影响①：初始位置偏移

**真实情况：**
```
控制器命令关节到home → 编码器反馈biased → PD控制器稳态：
  q_measured = q_home → q_true = q_home - bias
关节实际停在错误位置，但控制器以为到了home。
```

**仿真实现（base.py reset_robot()）：**
```python
if bias is not None:
    self._data.qpos[self._panda_dof_ids] = self._home_position - bias[:7]
else:
    self._data.qpos[self._panda_dof_ids] = self._home_position
```

**合理性：** 直接将关节设为 `home - bias`，等效于PD控制器的稳态结果。
跳过了PD控制器的瞬态过程（真实机器人上需要几百毫秒收敛），
但稳态结果完全一致。

**影响：** MuJoCo窗口中可以看到，有偏差时机器人初始姿态和无偏差时不同。
偏差0.2rad时末端位置偏移约13cm。

### 3.2 影响②：执行偏差（OSC力矩方向偏）

**真实情况：**
```
OSC控制器每个控制周期（1kHz）：
  读编码器 → q_measured = q_true + bias（始终用错误值）
  算Jacobian J(q_measured) → 方向偏了
  算FK x_current = FK(q_measured) → 位置偏了
  算力矩 tau = J^T × Mx × (x_target - x_current) → 力矩偏了
  施加力矩到关节
```

**仿真实现（base.py apply_action()）：**
```python
for _ in range(self._n_substeps):  # 50个substep
    if bias is not None:
        # 临时替换qpos为编码器读数
        true_qpos = self._data.qpos[self._panda_dof_ids].copy()
        self._data.qpos[self._panda_dof_ids] = true_qpos + bias[:7]
        mujoco.mj_forward(self._model, self._data)  # 更新Jacobian/FK

    # OSC用biased状态计算力矩
    tau = opspace(model, data, ...)

    if bias is not None:
        # 恢复真实qpos，物理仿真用真实值
        self._data.qpos[self._panda_dof_ids] = true_qpos
        mujoco.mj_forward(self._model, self._data)

    self._data.ctrl[self._panda_ctrl_ids] = tau
    mujoco.mj_step(self._model, self._data)  # 物理仿真用真实qpos
```

**合理性：**
- "临时替换→计算→恢复" 等效于真实机器人上 "控制器只能读到biased编码器值"。
- 真实机器人上不存在"恢复"操作——控制器从头到尾只用biased值。
- 但物理效果相同：力矩是基于biased状态计算的，施加到真实关节上。
- 仿真中的50个substep（0.1s/0.002s）对应真实的500Hz控制频率，和真实机器人接近（通常1kHz）。

**与真实的差异：**
- 真实机器人还有关节摩擦、电机延迟、齿轮回差等，仿真中没有模拟。
- 这些因素会放大偏差的影响——仿真中偏差效果偏乐观。
- 仿真中物理积分用真实qpos（完美物理），真实中没有"完美物理"。

### 3.3 影响③：感知偏差（策略收到错误观测）

**真实情况：**
```
机器人状态反馈：
  关节角度 = 编码器读数 = q_true + bias  ← 错误
  关节速度 = 编码器读数的导数            ← 固定bias下正确（bias导数=0）
  末端位置 = FK(编码器读数)              ← 错误（通过biased FK计算）
  夹爪状态 = 独立传感器                  ← 正确（不经过主臂编码器）
```

**仿真实现（base.py get_robot_state()）：**
```python
qpos_true = self._data.qpos[self._panda_dof_ids]
qvel = self._data.qvel[self._panda_dof_ids]
gripper_pose = self.get_gripper_pose()

if bias is not None:
    qpos_measured = qpos_true + bias[:7]           # 编码器读数
    tcp_pos = self._get_biased_tcp_pos(bias)       # biased FK
else:
    qpos_measured = qpos_true
    tcp_pos = self._data.sensor("2f85/pinch_pos").data

return np.concatenate([qpos_measured, qvel, gripper_pose, tcp_pos])
```

**biased FK的实现（_get_biased_tcp_pos）：**
```python
def _get_biased_tcp_pos(self, bias):
    true_qpos = self._data.qpos[self._panda_dof_ids].copy()
    self._data.qpos[self._panda_dof_ids] = true_qpos + bias[:7]  # 临时替换
    mujoco.mj_forward(self._model, self._data)                    # 重新计算FK
    biased_pos = self._data.site_xpos[self._pinch_site_id].copy() # 读取biased位置
    self._data.qpos[self._panda_dof_ids] = true_qpos              # 恢复
    mujoco.mj_forward(self._model, self._data)
    return biased_pos
```

**合理性：** 完全等效于真实机器人——控制器和上层系统只能通过编码器获取关节信息，
FK也是基于编码器读数计算的。仿真中的"临时替换"只是实现技巧，物理等效。

---

## 4. 外部传感器的模拟

除了受编码器偏差影响的信息，策略还接收不受偏差影响的外部传感器数据：

### 4.1 方块位置 block_pos（维度18-20）

**真实场景：** 通过固定在工作台上方的外部相机检测目标物体位置。
常见方案：
- RGB相机 + 颜色检测（精度 ~1-2cm）
- RGBD相机 + 点云分割（精度 ~3-5mm）
- ArUco marker + 单目相机（精度 ~5mm）

**仿真实现：** 直接读MuJoCo传感器，精确值。
```python
block_pos = self._data.sensor("block_pos").data
```

**合理性：** 外部相机不经过机器人编码器，不受偏差影响。
仿真中用精确值是理想化的，后续实验可加噪声模拟传感器精度。

### 4.2 真实末端位置 noisy_real_tcp（维度21-23）

**真实场景：** 通过外部定位系统追踪机器人末端的真实位置。
常见方案：
- 末端贴反光marker + OptiTrack动捕系统（精度 <1mm）
- 末端贴ArUco marker + 外部相机（精度 ~5mm）
- RGBD相机 + 末端检测（精度 ~1-2cm）

**仿真实现：** MuJoCo真实末端位置 + 5mm高斯噪声。
```python
real_tcp = self._data.site_xpos[self._pinch_site_id].copy()
noisy_real_tcp = real_tcp + np.random.normal(0, 0.005, 3)  # σ=5mm
```

**合理性：** 5mm噪声对应ArUco marker追踪的典型精度。
外部定位系统不经过机器人编码器，不受偏差影响。
噪声防止策略完全依赖real_tcp而忽略其他信息。

---

## 5. 仿真与真实的差异总结

| 方面 | 仿真 | 真实 | 差异影响 |
|------|------|------|----------|
| 关节PD控制器 | 直接设 q=home-bias（稳态） | PD控制器持续运行 | 仿真跳过瞬态，稳态一致 |
| OSC计算 | 临时替换qpos→计算→恢复 | 控制器始终用biased读数 | 物理等效，实现方式不同 |
| 物理仿真 | MuJoCo（完美刚体动力学） | 真实物理（摩擦/延迟/柔性） | 仿真偏乐观，真实偏差影响更大 |
| 编码器偏差类型 | 固定/随机（episode级） | 固定/漂移/噪声混合 | 仿真简化了偏差模型 |
| 控制频率 | 50 substeps/0.1s = 500Hz | 通常 1kHz | 接近，影响小 |
| 关节摩擦 | 无 | 有 | 仿真少了一个干扰源 |
| 电机延迟 | 无 | ~1-5ms | 仿真少了一个干扰源 |
| 传感器噪声 | 仅real_tcp加了5mm噪声 | 所有传感器都有噪声 | 仿真偏理想 |

**总体评价：** 仿真是理想化的——偏差效果比真实偏小。
如果方法在仿真中有效，真机上问题只会更严重，方法更有必要。
这使得仿真结果具有保守的说服力。

---

## 6. 关键设计决策及其理由

### 6.1 为什么用固定/随机偏差，不用漂移偏差？

**决策：** 每个episode开始时采样一个固定偏差，episode内不变。

**理由：**
- 固定偏差是最基本的故障模型，先验证方法在简单情况下有效
- 随机偏差（每episode不同）测试策略的泛化能力
- 漂移偏差（episode内变化）是更复杂的场景，留作未来工作
- 从实验设计角度，固定→随机→漂移是自然的难度递进

### 6.2 为什么只偏移Joint 4？

**决策：** 默认只给Joint 4（肘关节）注入偏差。

**理由：**
- Joint 4是肘关节，对末端位置影响显著（杠杆臂长）
- 单关节偏差便于分析和可视化
- 多关节偏差是简单的扩展（改config即可），留作消融实验
- 真实场景中通常是某一个关节出问题，不是所有关节同时偏

### 6.3 偏差范围 [0, 0.25] rad 的选择

**决策：** 随机偏差从均匀分布 U(0, 0.25) 采样。

**依据（实验数据）：**

| Joint 4偏差 | 无偏差策略成功率 | 末端偏移 |
|-------------|-----------------|----------|
| 0.00 rad | 100% | 0 cm |
| 0.05 rad | 100% | ~3 cm |
| 0.10 rad | 80% | ~6.5 cm |
| 0.15 rad | 62% | ~10 cm |
| 0.20 rad | 51% | ~13 cm |
| 0.25 rad | 5% | ~16 cm |
| 0.30 rad | 0% | ~19 cm |

- 0.25 rad 是"有挑战但不至于完全不可能"的上限
- 超过0.3 rad末端可能超出工作空间边界
- [0, 0.25] 覆盖了从"几乎无影响"到"严重影响"的全范围
- 真实场景中0.1-0.2 rad的偏差是常见的（对应6-12度）

### 6.4 为什么mocap_pos和mocap_quat都用biased FK初始化？

**决策：** reset时 `mocap_pos = biased_FK(q_true+bias)`，`mocap_quat`同理。

**理由：** 模拟真实情况——控制器用编码器读数初始化自身状态。
如果只偏position不偏orientation，OSC会看到虚假的姿态误差，
主动旋转手腕去纠正一个不存在的误差，产生不真实的行为。
"偏差要骗就全骗，不能只骗一半。"

### 6.5 为什么noisy_real_tcp加5mm噪声而不是精确值？

**决策：** `noisy_real_tcp = real_tcp + N(0, 5mm²)`

**理由：**
- 精确值：策略可能学成"忽略biased_tcp，只用real_tcp" → 绕过了bias而不是学会补偿
- 5mm噪声：对应ArUco marker追踪的典型精度，策略必须综合利用多个信息源
- 后续消融实验可以测试不同噪声水平的影响

---

## 7. 实验验证仿真合理性的方法

### 7.1 仿真内验证

1. **偏差注入前后对比：** 无偏差策略在有偏差环境中性能显著下降 → 偏差确实有效果
2. **偏差大小vs性能曲线：** 偏差越大性能越差 → 偏差效果符合物理直觉
3. **真实末端位置验证：** biased_tcp ≠ real_tcp，差异和bias成正比 → FK偏差正确

### 7.2 Sim-to-Real 验证（未来工作）

1. **真机偏差注入：** 在Franka Panda的关节角度读数上软件注入bias
2. **对比仿真和真机的偏差效果：** 相同bias下末端偏移量是否一致
3. **仿真训练策略 → 真机部署：** 策略是否能迁移
