"""Tests for frrl.envs.real_config."""
import numpy as np
import pytest

from frrl.envs.real_config import (
    FrankaRealConfig,
    center_square_crop,
    workspace_roi_crop_placeholder,
)


class TestFrankaRealConfigDefaults:
    def test_has_front_and_wrist_cameras(self):
        cfg = FrankaRealConfig()
        assert set(cfg.cameras.keys()) == {"front", "wrist"}

    def test_image_size_is_per_camera_dict(self):
        cfg = FrankaRealConfig()
        assert isinstance(cfg.image_size, dict)
        # Both cameras are 128² — SAC's image_encoder requires all cameras to
        # share a shape; see modeling_sac.py:586.
        assert cfg.image_size["front"] == (128, 128)
        assert cfg.image_size["wrist"] == (128, 128)

    def test_image_crop_has_defaults_for_both_cameras(self):
        cfg = FrankaRealConfig()
        assert "front" in cfg.image_crop
        assert "wrist" in cfg.image_crop
        assert callable(cfg.image_crop["front"])
        assert callable(cfg.image_crop["wrist"])

    def test_encoder_bias_config_default_is_none(self):
        """Default is None — training configs must explicitly enable bias."""
        cfg = FrankaRealConfig()
        assert cfg.encoder_bias_config is None

    def test_camera_streams_are_rgb_640x480_15fps(self):
        cfg = FrankaRealConfig()
        for name, cam in cfg.cameras.items():
            assert cam["dim"] == (640, 480)
            assert cam["fps"] == 15

    def test_camera_serials_are_hardcoded_for_hpc_old(self):
        """hpc-old 物理固定: front=234222303420, wrist=318122303303.

        验证见 reference_dual_realsense_usb_setup.md 和 2026-04-23 用户确认。
        换机器需要改 FrankaRealConfig 默认或传 override。
        """
        cfg = FrankaRealConfig()
        assert cfg.cameras["front"]["serial_number"] == "234222303420"
        assert cfg.cameras["wrist"]["serial_number"] == "318122303303"


class TestCenterSquareCrop:
    def test_landscape_crops_to_min_dim(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = center_square_crop(img)
        assert out.shape == (480, 480, 3)

    def test_portrait_crops_to_min_dim(self):
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        out = center_square_crop(img)
        assert out.shape == (480, 480, 3)

    def test_square_is_idempotent(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        out = center_square_crop(img)
        assert out.shape == (300, 300, 3)

    def test_preserves_dtype(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        assert center_square_crop(img).dtype == np.uint8

    def test_grayscale_single_channel(self):
        img = np.zeros((480, 640), dtype=np.uint8)
        out = center_square_crop(img)
        assert out.shape == (480, 480)

    def test_crop_content_is_centered(self):
        """Center pixel must survive, extreme edges must not."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[50, 100] = [255, 128, 64]  # center pixel of original
        out = center_square_crop(img)
        # After crop x-range is [50, 150); original center (100) maps to new (50)
        assert tuple(out[50, 50]) == (255, 128, 64)


class TestWorkspaceRoiCropPlaceholder:
    def test_placeholder_is_center_square_until_aruco_calibration(self):
        """Placeholder mirrors center_square_crop. When阶段 3.5 lands, this should change."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert np.array_equal(
            workspace_roi_crop_placeholder(img),
            center_square_crop(img),
        )
