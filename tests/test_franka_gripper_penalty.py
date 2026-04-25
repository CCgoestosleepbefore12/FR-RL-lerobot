"""FrankaGripperPenaltyProcessorStep 单元测试。

验证 sim 对齐语义（action∈[-1,1], state∈[0,1]）：
  - action<-0.5 且 pos<0.1  → penalty
  - action> 0.5 且 pos>0.9  → penalty
  - 其他区间                → 0
与 `frrl.envs.wrappers.hil_wrappers.GripperPenaltyWrapper` 逻辑一致。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_step(penalty=-0.05):
    from frrl.processor import FrankaGripperPenaltyProcessorStep
    return FrankaGripperPenaltyProcessorStep(penalty=penalty)


def _apply(step, action_value, gripper_pos):
    """模拟 ProcessorStep.__call__：把 action 放到 transition，设 raw_joint_positions，调 step。"""
    from frrl.processor import create_transition
    from frrl.processor.pipeline import TransitionKey

    action = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(action_value)])
    transition = create_transition(
        observation={},
        action=action,
        complementary_data={"raw_joint_positions": {"gripper": float(gripper_pos)}},
    )
    result = step(transition)
    return result[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)


class TestPenaltyFires:
    def test_close_more_when_closed(self):
        """已关（pos<0.1）还想关（action<-0.5） → penalty。"""
        p = _apply(_make_step(-0.05), action_value=-0.8, gripper_pos=0.05)
        assert p == -0.05

    def test_open_more_when_open(self):
        """已开（pos>0.9）还想开（action>0.5） → penalty。"""
        p = _apply(_make_step(-0.05), action_value=0.9, gripper_pos=0.95)
        assert p == -0.05

    def test_custom_penalty_value(self):
        p = _apply(_make_step(-0.1), action_value=-0.9, gripper_pos=0.0)
        assert p == -0.1


class TestPenaltyNoFire:
    def test_mid_state(self):
        """pos 在中间（0.5） → 不触发，无论 action。"""
        for a in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            p = _apply(_make_step(), action_value=a, gripper_pos=0.5)
            assert p == 0.0, f"action={a} at mid-state fired penalty"

    def test_small_action(self):
        """action 小（|a|<0.5）→ 不触发，无论 pos。"""
        for pos in [0.0, 0.05, 0.5, 0.95, 1.0]:
            p = _apply(_make_step(), action_value=0.2, gripper_pos=pos)
            assert p == 0.0, f"pos={pos} with small action fired"

    def test_close_when_open(self):
        """已开，想关 → 合理操作，不罚。"""
        p = _apply(_make_step(), action_value=-0.9, gripper_pos=0.95)
        assert p == 0.0

    def test_open_when_closed(self):
        """已关，想开 → 合理操作，不罚。"""
        p = _apply(_make_step(), action_value=0.9, gripper_pos=0.05)
        assert p == 0.0


class TestMissingData:
    def test_no_raw_joint_positions(self):
        """complementary_data 缺 raw_joint_positions → 原样返回，不抛。"""
        from frrl.processor import create_transition
        from frrl.processor.pipeline import TransitionKey
        step = _make_step()
        transition = create_transition(
            observation={},
            action=torch.tensor([0.0] * 7),
            complementary_data={},  # 无 raw
        )
        result = step(transition)
        cd = result[TransitionKey.COMPLEMENTARY_DATA]
        assert "discrete_penalty" not in cd

    def test_no_gripper_key(self):
        """raw_joint_positions 里没 'gripper' key → no-op。"""
        from frrl.processor import create_transition
        from frrl.processor.pipeline import TransitionKey
        step = _make_step()
        transition = create_transition(
            observation={},
            action=torch.tensor([0.0] * 7),
            complementary_data={"raw_joint_positions": {"shoulder.pos": 0.5}},
        )
        result = step(transition)
        assert "discrete_penalty" not in result[TransitionKey.COMPLEMENTARY_DATA]


class TestBoundaries:
    def test_action_exactly_neg_half(self):
        """action = -0.5 严格不触发（<，非 ≤）。"""
        p = _apply(_make_step(), action_value=-0.5, gripper_pos=0.05)
        assert p == 0.0

    def test_pos_exactly_0p1(self):
        """pos = 0.1 严格不触发（<，非 ≤）。"""
        p = _apply(_make_step(), action_value=-0.9, gripper_pos=0.1)
        assert p == 0.0

    def test_pos_exactly_0p9(self):
        """pos = 0.9 严格不触发（>，非 ≥）。"""
        p = _apply(_make_step(), action_value=0.9, gripper_pos=0.9)
        assert p == 0.0


class TestEnvIntegration:
    def test_franka_real_env_has_method(self):
        """FrankaRealEnv.get_raw_joint_positions 存在且返回正确格式。"""
        # 只检查类/方法存在性，不构造真 env（会连 RT PC）
        from frrl.envs.real import FrankaRealEnv
        assert hasattr(FrankaRealEnv, "get_raw_joint_positions")


class TestActionShapeAssert:
    """P1-4：action 不是 unbatched [7] 时必须立刻崩，防 AddBatchDimensionProcessorStep
    误提前执行或 action dim 漂移悄悄污染 penalty 信号。
    """

    def _apply_raw_action(self, action):
        """直接传自定义 shape 的 action，绕过 _apply 的固定 [7] 构造。"""
        from frrl.processor import create_transition
        step = _make_step()
        transition = create_transition(
            observation={},
            action=action,
            complementary_data={"raw_joint_positions": {"gripper": 0.05}},
        )
        return step(transition)

    def test_batched_action_raises(self):
        """[1, 7] batched action → AssertionError。"""
        action = torch.zeros(1, 7)
        try:
            self._apply_raw_action(action)
            raise AssertionError("should have raised on batched action")
        except AssertionError as e:
            assert "unbatched 7D action" in str(e)

    def test_wrong_dim_raises(self):
        """dim != 7 → AssertionError。"""
        action = torch.zeros(6)
        try:
            self._apply_raw_action(action)
            raise AssertionError("should have raised on 6D action")
        except AssertionError as e:
            assert "unbatched 7D action" in str(e)

    def test_numpy_array_also_checked(self):
        """非 tensor（numpy array）也要被 assert 覆盖。"""
        import numpy as np
        action = np.zeros((2, 7), dtype=np.float32)
        try:
            self._apply_raw_action(action)
            raise AssertionError("should have raised on batched ndarray")
        except AssertionError as e:
            assert "unbatched 7D action" in str(e)

    def test_numpy_1d_7_passes_and_computes_penalty(self):
        """合法 numpy [7] action 不应触发 assert，penalty 计算结果正确。

        正向用例：防止 numpy 分支断言条件被误反成 `dim==1 or shape==7`，
        从而把所有合法输入误杀。覆盖 torch tensor 路径之外的 ndarray 入口。
        """
        import numpy as np
        from frrl.processor import create_transition
        from frrl.processor.pipeline import TransitionKey

        # action[-1]=-0.9 + gripper_pos=0.05 → 触发"已关还在关"penalty
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9], dtype=np.float32)
        step = _make_step(-0.05)
        transition = create_transition(
            observation={},
            action=action,
            complementary_data={"raw_joint_positions": {"gripper": 0.05}},
        )
        result = step(transition)
        assert result[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"] == -0.05


class TestGripperKeyConstant:
    """P1-5：env 写入 / processor 读取 / constants 常量必须是同一真源。"""

    def test_constant_is_gripper(self):
        from frrl.utils.constants import RAW_JOINT_POSITION_GRIPPER_KEY
        assert RAW_JOINT_POSITION_GRIPPER_KEY == "gripper"

    def test_hil_processor_reuses_constant(self):
        """hil_processor.GRIPPER_KEY 必须来自 constants，不能各自 hard-code。"""
        from frrl.processor.hil_processor import GRIPPER_KEY
        from frrl.utils.constants import RAW_JOINT_POSITION_GRIPPER_KEY
        assert GRIPPER_KEY is RAW_JOINT_POSITION_GRIPPER_KEY or GRIPPER_KEY == RAW_JOINT_POSITION_GRIPPER_KEY
