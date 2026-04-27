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
        if self.binary_gripper_pose == 1:
            return
        msg = GraspActionGoal()
        msg.goal.width = 0.01
        msg.goal.speed = 0.3
        msg.goal.epsilon.inner = 1
        msg.goal.epsilon.outer = 1
        msg.goal.force = 130
        self.grippergrasppub.publish(msg)
        self.binary_gripper_pose = 1

    def close_slow(self):
        if self.binary_gripper_pose == 1:
            return
        msg = GraspActionGoal()
        msg.goal.width = 0.01
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
