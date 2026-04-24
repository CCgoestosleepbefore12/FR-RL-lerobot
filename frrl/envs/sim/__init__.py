"""仿真环境子包：MuJoCo Panda 系列 env + OSC 控制器。

注意：
- 本包只管"模型 + 物理 + 动力学"，不含 wrappers（通用）和 real（真机）。
- 环境的 gym register 仍在 `frrl.envs.__init__`，保证用户侧调用 `gym.make("gym_frrl/...")` 的习惯不变。
"""
