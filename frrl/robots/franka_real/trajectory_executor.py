#!/usr/bin/env python3
"""
轨迹执行器

支持：
- 定义关键点轨迹（位置+姿态+夹爪）
- 线性/样条插值
- 轨迹可视化
- 轨迹保存/加载
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
import json


@dataclass
class Waypoint:
    """轨迹关键点"""
    position: np.ndarray  # [x, y, z] 位置
    orientation: np.ndarray  # [qw, qx, qy, qz] 四元数姿态
    gripper: float  # 0.0=开, 1.0=关
    duration: float  # 到达此点所需时间(秒)
    name: str = ""  # 关键点名称

    def to_dict(self):
        """转换为字典（用于保存）"""
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(),
            'gripper': self.gripper,
            'duration': self.duration,
            'name': self.name,
        }

    @classmethod
    def from_dict(cls, data):
        """从字典加载"""
        return cls(
            position=np.array(data['position']),
            orientation=np.array(data['orientation']),
            gripper=data['gripper'],
            duration=data['duration'],
            name=data.get('name', ''),
        )


class TrajectoryInterpolator:
    """轨迹插值器"""

    def __init__(self, waypoints: List[Waypoint], control_dt: float = 0.1):
        """
        初始化插值器。

        Args:
            waypoints: 关键点列表
            control_dt: 控制频率(秒)
        """
        self.waypoints = waypoints
        self.control_dt = control_dt

        # 构建时间轴（累积时间）
        self.timestamps = []
        cumulative_time = 0.0
        for wp in waypoints:
            cumulative_time += wp.duration
            self.timestamps.append(cumulative_time)

        self.total_duration = self.timestamps[-1]

        # 构建插值器
        self._build_interpolators()

    def _build_interpolators(self):
        """构建位置、姿态、夹爪的插值器"""

        # 提取所有关键点数据
        positions = np.array([wp.position for wp in self.waypoints])
        orientations = np.array([wp.orientation for wp in self.waypoints])
        grippers = np.array([wp.gripper for wp in self.waypoints])

        # 位置插值（线性）
        self.position_interp = interp1d(
            self.timestamps,
            positions.T,
            kind='linear',
            axis=1,
            fill_value='extrapolate'
        )

        # 夹爪插值（最近邻，保持阶跃）
        self.gripper_interp = interp1d(
            self.timestamps,
            grippers,
            kind='previous',
            fill_value='extrapolate'
        )

        # 姿态插值（球面线性插值 - SLERP）
        self.rotations = R.from_quat(orientations[:, [1, 2, 3, 0]])  # [qx,qy,qz,qw]格式
        self.slerp = Slerp(self.timestamps, self.rotations)

    def get_pose_at_time(self, t: float) -> Dict[str, np.ndarray]:
        """
        获取指定时间的位姿。

        Args:
            t: 时间(秒)

        Returns:
            {'position': [x,y,z], 'orientation': [qw,qx,qy,qz], 'gripper': 0-1}
        """
        # 限制在有效时间范围
        t = np.clip(t, 0, self.total_duration)

        # 插值位置
        position = self.position_interp(t)

        # 插值姿态（SLERP）
        rotation = self.slerp(t)
        orientation_scipy = rotation.as_quat()  # [qx, qy, qz, qw]
        orientation = np.array([orientation_scipy[3], orientation_scipy[0], orientation_scipy[1], orientation_scipy[2]])  # 转为[qw,qx,qy,qz]

        # 插值夹爪
        gripper = float(self.gripper_interp(t))

        return {
            'position': position,
            'orientation': orientation,
            'gripper': gripper,
        }

    def generate_trajectory(self) -> List[Dict[str, np.ndarray]]:
        """
        生成完整轨迹（离散时间步）。

        Returns:
            轨迹点列表
        """
        num_steps = int(self.total_duration / self.control_dt) + 1
        times = np.linspace(0, self.total_duration, num_steps)

        trajectory = []
        for t in times:
            pose = self.get_pose_at_time(t)
            trajectory.append(pose)

        return trajectory

    def get_duration(self) -> float:
        """获取轨迹总时长"""
        return self.total_duration


class TrajectoryExecutor:
    """轨迹执行器"""

    def __init__(self, env, interpolator: TrajectoryInterpolator):
        """
        初始化执行器。

        Args:
            env: Gym环境
            interpolator: 轨迹插值器
        """
        self.env = env
        self.interpolator = interpolator
        self.control_dt = env.control_dt

    def execute(self, render=True, verbose=True) -> Dict[str, Any]:
        """
        执行轨迹。

        Args:
            render: 是否渲染
            verbose: 是否输出详细信息

        Returns:
            执行结果统计
        """
        if verbose:
            print("=" * 70)
            print("轨迹执行")
            print("=" * 70)
            print(f"轨迹时长: {self.interpolator.get_duration():.2f}秒")
            print(f"关键点数: {len(self.interpolator.waypoints)}")
            print("=" * 70)
            print()

        # 获取当前观测（不重置环境，由调用方控制）
        obs = self.env._get_obs()

        total_reward = 0.0
        step_count = 0
        t = 0.0

        # 按时间步执行
        try:
            while t <= self.interpolator.get_duration():
                # 获取目标位姿
                target_pose = self.interpolator.get_pose_at_time(t)

                # 获取当前位置
                current_ee_pos = obs['state'][:3]

                # 计算Delta动作
                delta_pos = target_pose['position'] - current_ee_pos

                # 归一化到[-1, 1]
                action = np.zeros(4, dtype=np.float32)
                action[:3] = np.clip(delta_pos / self.env.delta_action_scale, -1, 1)
                action[3] = target_pose['gripper']

                # 执行动作
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                step_count += 1

                # 渲染
                if render:
                    self.env.render()

                # 检查是否完成
                if terminated or truncated:
                    break

                # 前进时间
                t += self.control_dt

        except KeyboardInterrupt:
            # 用户中断，提前结束执行
            if verbose:
                print()
                print("⚠️  轨迹执行被用户中断")
            # 重新抛出异常，让外层处理
            raise

        # 最终结果
        final_block_pos = obs['state'][4:7]
        success = info.get('success', False)

        if verbose:
            print(f"执行完成 | 步数: {step_count} | 成功: {'YES' if success else 'NO'}")

        return {
            'success': success,
            'steps': step_count,
            'final_block_pos': final_block_pos,
        }


def save_trajectory(waypoints: List[Waypoint], filepath: str):
    """保存轨迹到JSON文件"""
    data = {
        'waypoints': [wp.to_dict() for wp in waypoints]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ 轨迹已保存到: {filepath}")


def load_trajectory(filepath: str) -> List[Waypoint]:
    """从JSON文件加载轨迹"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    waypoints = [Waypoint.from_dict(wp) for wp in data['waypoints']]
    print(f"✅ 轨迹已加载: {filepath} ({len(waypoints)}个关键点)")
    return waypoints


if __name__ == "__main__":
    # 测试轨迹插值器
    print("测试轨迹插值器...")

    # 定义简单轨迹
    waypoints = [
        Waypoint(
            position=np.array([0.5, 0.0, 0.1]),
            orientation=np.array([1, 0, 0, 0]),
            gripper=0.0,
            duration=0.0,
            name="起点"
        ),
        Waypoint(
            position=np.array([0.5, 0.0, 0.3]),
            orientation=np.array([1, 0, 0, 0]),
            gripper=0.0,
            duration=2.0,
            name="抬升"
        ),
        Waypoint(
            position=np.array([0.5, 0.2, 0.3]),
            orientation=np.array([1, 0, 0, 0]),
            gripper=1.0,
            duration=2.0,
            name="移动+抓取"
        ),
    ]

    interpolator = TrajectoryInterpolator(waypoints, control_dt=0.1)
    trajectory = interpolator.generate_trajectory()

    print(f"✅ 生成了 {len(trajectory)} 个轨迹点")
    print(f"✅ 轨迹总时长: {interpolator.get_duration():.2f}秒")

    # 测试保存/加载
    save_trajectory(waypoints, "/tmp/test_trajectory.json")
    loaded_waypoints = load_trajectory("/tmp/test_trajectory.json")
    print(f"✅ 测试完成")
