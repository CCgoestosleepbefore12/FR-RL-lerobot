#!/usr/bin/env python3
"""
编码器校准偏差注入模块

模拟工业机器人编码器校准误差（encoder calibration bias）：
- 一个故障源（编码器bias）→ 同时影响控制和观测
- 控制器基于错误的关节角度计算Jacobian → 力矩不准 → 执行偏差
- 观测基于错误的FK计算 → EE位姿报告不准 → 感知偏差
- 物理仿真本身不受影响（关节真实位置正确）

这精确模拟了真实机器人编码器校准偏差的因果链：
    编码器偏差 → q_measured = q_true + bias
        ├→ 控制器用 q_measured 计算（控制受影响）
        └→ FK(q_measured) 得到错误的 EE 位姿（观测受影响）
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class EncoderBiasConfig:
    """编码器偏差配置"""

    # 基础设置
    enable: bool = True
    error_probability: float = 1.0  # Episode级别故障概率

    # 目标关节（None=所有关节）
    target_joints: Optional[List[int]] = None
    per_joint_probability: float = 0.7  # 每关节被影响概率（仅target_joints=None时）

    # Bias模式
    bias_mode: str = 'random_uniform'  # 'fixed' 或 'random_uniform'
    fixed_bias_value: float = 0.1  # fixed模式下的bias值（弧度）
    bias_range: Tuple[float, float] = (0.0, 1.0)  # random_uniform模式下的范围（弧度）

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> 'EncoderBiasConfig':
        """从YAML文件加载配置"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # 处理target_joints字段
        target_joints = data.get('target_joints', None)
        if isinstance(target_joints, list) and len(target_joints) == 0:
            target_joints = None

        return cls(
            enable=data.get('enable', True),
            error_probability=data.get('error_probability', 1.0),
            target_joints=target_joints,
            per_joint_probability=data.get('per_joint_probability', 0.7),
            bias_mode=data.get('bias_mode', 'random_uniform'),
            fixed_bias_value=data.get('fixed_bias_value', 0.1),
            bias_range=tuple(data.get('bias_range', [0.0, 1.0])),
        )


class EncoderBiasInjector:
    """
    编码器校准偏差注入器

    模拟编码器校准误差：每个episode开始时采样一个bias向量，
    该bias同时影响控制器（通过biased Jacobian）和观测（通过biased FK）。
    """

    def __init__(self, config: EncoderBiasConfig):
        self.config = config
        self._current_bias: Optional[np.ndarray] = None
        self._active = False
        self.episode_count = 0

        # 统计
        self.stats = {
            'total_episodes': 0,
            'fault_episodes': 0,
            'bias_magnitudes': [],
        }

        print("=" * 60)
        print("EncoderBiasInjector 初始化")
        print("=" * 60)
        print(f"  启用: {config.enable}")
        print(f"  故障概率: {config.error_probability * 100:.0f}%")

        if config.target_joints is not None:
            joint_names = [f"Joint {j+1}" for j in config.target_joints]
            print(f"  目标关节: {', '.join(joint_names)}")
        else:
            print(f"  目标关节: 全部 (每关节概率 {config.per_joint_probability*100:.0f}%)")

        print(f"  Bias模式: {config.bias_mode}")
        if config.bias_mode == 'fixed':
            print(f"  固定值: {config.fixed_bias_value:.3f} rad ({np.rad2deg(config.fixed_bias_value):.1f} deg)")
        else:
            print(f"  范围: [{config.bias_range[0]:.3f}, {config.bias_range[1]:.3f}] rad")
            print(f"         [{np.rad2deg(config.bias_range[0]):.1f}, {np.rad2deg(config.bias_range[1]):.1f}] deg")
        print("=" * 60)

    @property
    def current_bias(self) -> Optional[np.ndarray]:
        """当前episode的bias向量，None表示本episode无故障"""
        if self._active:
            return self._current_bias
        return None

    @property
    def is_active(self) -> bool:
        """当前episode是否有故障"""
        return self._active

    def on_episode_start(self, num_joints: int = 7) -> bool:
        """
        Episode开始时决定是否注入故障，并生成bias向量

        Returns:
            是否注入了故障
        """
        self.episode_count += 1
        self.stats['total_episodes'] += 1

        # 按概率决定是否注入
        self._active = (
            self.config.enable
            and np.random.random() < self.config.error_probability
        )

        if self._active:
            self._current_bias = self._sample_bias(num_joints)
            self.stats['fault_episodes'] += 1
            self.stats['bias_magnitudes'].append(
                np.linalg.norm(self._current_bias)
            )
            return True
        else:
            self._current_bias = None
            return False

    def get_biased_qpos(self, true_qpos: np.ndarray) -> np.ndarray:
        """
        返回编码器报告的关节角度（带bias）

        真实机器人上：编码器读数 = 真实角度 + 校准偏差
        用于控制器计算和观测FK。

        Args:
            true_qpos: 真实关节角度（前7个为手臂关节）

        Returns:
            带bias的关节角度（只修改手臂关节，不影响gripper和物体）
        """
        if not self._active or self._current_bias is None:
            return true_qpos

        biased = true_qpos.copy()
        n = min(7, len(self._current_bias))
        biased[:n] += self._current_bias[:n]
        return biased

    def _sample_bias(self, num_joints: int) -> np.ndarray:
        """采样bias向量"""
        bias = np.zeros(num_joints)

        if self.config.target_joints is not None:
            # 指定关节
            for j in self.config.target_joints:
                if j < num_joints:
                    bias[j] = self._sample_single_bias()
        else:
            # 所有关节，按概率独立采样
            for j in range(num_joints):
                if np.random.random() < self.config.per_joint_probability:
                    bias[j] = self._sample_single_bias()

        return bias

    def _sample_single_bias(self) -> float:
        """采样单个关节的bias值"""
        if self.config.bias_mode == 'fixed':
            return self.config.fixed_bias_value
        else:  # random_uniform
            return np.random.uniform(
                self.config.bias_range[0],
                self.config.bias_range[1],
            )

    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        if stats['total_episodes'] > 0:
            stats['fault_rate'] = stats['fault_episodes'] / stats['total_episodes']
        if stats['bias_magnitudes']:
            mags = stats['bias_magnitudes']
            stats['avg_magnitude'] = np.mean(mags)
            stats['max_magnitude'] = np.max(mags)
            stats['min_magnitude'] = np.min(mags)
        return stats

    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("编码器偏差注入统计")
        print("=" * 60)
        print(f"  总Episode数: {stats['total_episodes']}")
        print(f"  故障Episode数: {stats['fault_episodes']}")
        print(f"  故障率: {stats.get('fault_rate', 0) * 100:.1f}%")
        if 'avg_magnitude' in stats:
            print(f"  平均偏差幅度: {stats['avg_magnitude']:.4f} rad ({np.rad2deg(stats['avg_magnitude']):.2f} deg)")
            print(f"  最大偏差幅度: {stats['max_magnitude']:.4f} rad ({np.rad2deg(stats['max_magnitude']):.2f} deg)")
        print("=" * 60)


if __name__ == '__main__':
    print("测试 EncoderBiasInjector...\n")

    config = EncoderBiasConfig(
        enable=True,
        error_probability=0.5,
        target_joints=[3],  # Joint 4
        bias_mode='random_uniform',
        bias_range=(0.0, 0.5),
    )

    injector = EncoderBiasInjector(config)

    for ep in range(10):
        active = injector.on_episode_start(num_joints=7)
        if active:
            true_q = np.zeros(7)
            biased_q = injector.get_biased_qpos(true_q)
            print(f"Episode {ep+1}: 故障 | bias = {injector.current_bias}")
        else:
            print(f"Episode {ep+1}: 正常")

    injector.print_stats()
    print("\n测试完成")
