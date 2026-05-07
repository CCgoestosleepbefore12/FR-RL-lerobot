#!/usr/bin/env python3
"""EncoderBiasConfig — 编码器偏差注入的配置 dataclass + YAML loader."""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

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
