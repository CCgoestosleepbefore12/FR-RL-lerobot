import rospy
from franka_gripper.msg import GraspActionGoal, MoveActionGoal, HomingActionGoal
from sensor_msgs.msg import JointState
import numpy as np

from .gripper_server import GripperServer


class FrankaGripperServer(GripperServer):
    def __init__(self):
        super().__init__()
        self.grippermovepub = rospy.Publisher(
            "/franka_gripper/move/goal", MoveActionGoal, queue_size=1
        )
        self.grippergrasppub = rospy.Publisher(
            "/franka_gripper/grasp/goal", GraspActionGoal, queue_size=1
        )
        self.gripperhomingpub = rospy.Publisher(
            "/franka_gripper/homing/goal", HomingActionGoal, queue_size=1
        )
        self.gripper_sub = rospy.Subscriber(
            "/franka_gripper/joint_states", JointState, self._update_gripper
        )
        self.binary_gripper_pose = 0

    def reset_gripper(self):
        """Homing：让 finger 跑到机械极限再定零，校准 absolute encoder。
        必须在 finger 之间无任何物体（无海绵/工件/夹具）时跑，否则中途卡住
        homing 失败 → encoder 零位偏移 → close_gripper 后 width 报 0 但实际有 gap。
        """
        msg = HomingActionGoal()
        self.gripperhomingpub.publish(msg)
        # Homing 实测 ~3-4 秒（全开到全合两次），调用方应在调用后等待。

    def open(self):
        if self.binary_gripper_pose == 0:
            return
        msg = MoveActionGoal()
        # msg.goal.width = 0.025
        msg.goal.width = 0.09
        msg.goal.speed = 0.3
        self.grippermovepub.publish(msg)
        self.binary_gripper_pose = 0

    def close(self):
        """用 GraspAction 闭合 —— 空抓时 RT PC 会打 `ABORTED` log（因为没检测到
        force 接触，libfranka 视为失败），但功能上 gripper 仍然闭合到位。

        关键：grasp 和 move 用不同 action server (`/grasp/goal` vs `/move/goal`)，
        各自独立 lifecycle。grasp ABORTED 只污染 grasp server，**不影响后续
        open() 的 MoveAction**。这是 close 必须用 GraspAction 的理由（早先 0a81f17
        切 MoveAction 是为了消空抓 ABORTED log，但代价是 close 失败状态污染同
        publisher 上的 open，导致 open 报 `libfranka gripper: command failed`，
        2026-04-30 实测复现）。

        epsilon=1 大到任何 width 都视作 success；force=130 是 SERL 默认钳力。

        2026-04-30 width 0.01 → 0.025：原 0.01m (1cm) + 130N 把海绵这类软物压扁
        到 ~3mm (gripper_pos≈0.04)，触发 deploy 脚本 auto-success 的 gripper_held_min
        OUT range 误判。改 0.025m (2.5cm) 让 gripper 提前停在海绵表面 (~30mm)，
        gripper_pos ≈ 0.375 落在 [0.05, 0.6] 范围内。force 不变保持硬物钳力。
        副作用：物体厚度 < 2.5cm 时抓不紧（pickup 任务用海绵不会发生）。
        """
        if self.binary_gripper_pose == 1:
            return
        msg = GraspActionGoal()
        msg.goal.width = 0.025
        msg.goal.speed = 0.3
        msg.goal.epsilon.inner = 1
        msg.goal.epsilon.outer = 1
        msg.goal.force = 130
        self.grippergrasppub.publish(msg)
        self.binary_gripper_pose = 1

    def close_slow(self):
        """同 close()，仅调速 0.3→0.1。"""
        if self.binary_gripper_pose == 1:
            return
        msg = GraspActionGoal()
        msg.goal.width = 0.025
        msg.goal.speed = 0.1
        msg.goal.epsilon.inner = 1
        msg.goal.epsilon.outer = 1
        msg.goal.force = 130
        self.grippergrasppub.publish(msg)
        self.binary_gripper_pose = 1

    def move(self, position: int):
        """Move the gripper to a specific position in range [0, 255]"""
        msg = MoveActionGoal()
        msg.goal.width = float(position / (255 * 10))  # width in [0, 0.1]m
        msg.goal.speed = 0.3
        self.grippermovepub.publish(msg)

    def _update_gripper(self, msg):
        """internal callback to get the latest gripper position."""
        self.gripper_pos = np.sum(msg.position) / 0.08
