"""Tests for RealSenseCameraManager per-camera image_size dispatch.

We bypass __init__ (which opens pipelines) via __new__ and set attributes directly.
This tests only the _size_for() dispatch logic.
"""
import pytest

from frrl.cameras.realsense import RealSenseCameraManager


def _make_manager_without_hardware(image_size):
    """Construct without opening cameras. Only _size_for() is exercised."""
    mgr = RealSenseCameraManager.__new__(RealSenseCameraManager)
    mgr.image_size = image_size
    return mgr


class TestSizeFor:
    def test_tuple_image_size_returns_same_for_any_camera(self):
        mgr = _make_manager_without_hardware((128, 128))
        assert mgr._size_for("front") == (128, 128)
        assert mgr._size_for("wrist") == (128, 128)
        assert mgr._size_for("whatever") == (128, 128)

    def test_dict_image_size_returns_per_camera(self):
        mgr = _make_manager_without_hardware({"front": (224, 224), "wrist": (128, 128)})
        assert mgr._size_for("front") == (224, 224)
        assert mgr._size_for("wrist") == (128, 128)

    def test_dict_missing_camera_raises_key_error(self):
        mgr = _make_manager_without_hardware({"front": (224, 224)})
        with pytest.raises(KeyError, match="wrist"):
            mgr._size_for("wrist")

    def test_empty_dict_always_raises(self):
        mgr = _make_manager_without_hardware({})
        with pytest.raises(KeyError):
            mgr._size_for("front")
