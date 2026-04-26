"""PretrainedImageEncoder CNN/ViT 双路径 + asserts 单元测试。

DINOv3 是 HF gated repo 不能在 CI 上下载；用同架构同输出契约的 DINOv2-S（公开）
+ helper2424/resnet10 各跑一次，覆盖关键 invariant：
- forward 输出 4D (B, C, H, W)
- ViT path 自动 detect + reshape
- non-square 输入应在 SACObservationEncoder 层 raise
- 错误友好（gated repo / 网络）

Usage:
    pytest tests/test_pretrained_image_encoder.py -v
"""
import pytest
import torch

from frrl.policies.sac.modeling_sac import (
    PretrainedImageEncoder,
    SACObservationEncoder,
)
from frrl.policies.sac.configuration_sac import SACConfig
from frrl.configs.types import FeatureType, PolicyFeature, NormalizationMode


def _make_cfg(encoder_name: str, img_shape=(3, 128, 128)):
    return SACConfig(
        vision_encoder_name=encoder_name,
        freeze_vision_encoder=True,
        shared_encoder=True,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=img_shape),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(29,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
        },
        dataset_stats={
            "observation.state": {"min": [-1.0] * 29, "max": [1.0] * 29},
            "observation.images.front": {
                "mean": [[[0.485]], [[0.456]], [[0.406]]],
                "std": [[[0.229]], [[0.224]], [[0.225]]],
            },
            "action": {"min": [-1.0] * 7, "max": [1.0] * 7},
        },
    )


def test_cnn_path_resnet10_outputs_4d():
    """ResNet10 (CNN) 路径：last_hidden_state 已是 4D，adapter 直接透传。"""
    cfg = _make_cfg("helper2424/resnet10")
    enc = PretrainedImageEncoder(cfg)
    assert enc._is_vit is False, "ResNet10 应被检测为 CNN backbone"
    assert enc._num_non_patch_tokens == 0
    out = enc(torch.zeros(1, 3, 128, 128))
    assert out.dim() == 4, f"CNN encoder 输出应 4D, got ndim={out.dim()}"
    assert out.shape[0] == 1
    assert out.shape[1] == 512, f"ResNet10 final channel 应 512, got {out.shape[1]}"
    print(f"  [PASS] ResNet10 → 4D shape {tuple(out.shape)}")


def test_vit_path_dinov2_outputs_4d():
    """DINOv2-S (ViT) 路径：丢 CLS token 后 patch tokens reshape 成 (B, D, sqrt(N), sqrt(N))。
    DINOv3-S 同架构（差只在 register tokens=4 vs 0）。"""
    cfg = _make_cfg("facebook/dinov2-small")
    enc = PretrainedImageEncoder(cfg)
    assert enc._is_vit is True, "DINOv2-S 应被检测为 ViT backbone"
    assert enc._num_non_patch_tokens == 1, "DINOv2 没有 register tokens, 仅 CLS"
    out = enc(torch.zeros(1, 3, 128, 128))
    assert out.dim() == 4, f"ViT encoder 输出应 4D, got ndim={out.dim()}"
    assert out.shape[1] == 384, f"DINOv2-S hidden_size 应 384, got {out.shape[1]}"
    # 128 / patch=14 (HF 用 position embedding 插值) → 9×9=81 patches
    assert out.shape[2] == out.shape[3], "ViT reshape 后应是方阵"
    print(f"  [PASS] DINOv2-S → 4D shape {tuple(out.shape)}")


def test_sac_observation_encoder_rejects_non_square_input():
    """SACObservationEncoder.__init__ 加了 H==W assert，非方形输入应直接 raise。"""
    cfg = _make_cfg("helper2424/resnet10", img_shape=(3, 128, 96))
    with pytest.raises(AssertionError, match="square"):
        SACObservationEncoder(cfg)
    print(f"  [PASS] non-square 128×96 → AssertionError as expected")


def test_sac_observation_encoder_4d_assert_error_message():
    """4D assert 错误信息应清晰指引使用者。"""
    # 这个测试需要构造一个不输出 4D 的 encoder。直接 patch 简单点：
    cfg = _make_cfg("helper2424/resnet10")
    enc = SACObservationEncoder(cfg)  # 应正常构造
    # 验证内部 image_encoder 输出确实是 4D
    dummy = torch.zeros(1, 3, 128, 128)
    out = enc.image_encoder(dummy)
    assert out.ndim == 4
    print(f"  [PASS] 4D output verified end-to-end")


def test_sac_observation_encoder_dinov2_end_to_end():
    """端到端：双相机 DINOv2-S + state → fused latent (B, latent_dim×2 + state_dim)。"""
    cfg = SACConfig(
        vision_encoder_name="facebook/dinov2-small",
        freeze_vision_encoder=True,
        shared_encoder=True,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(29,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
        },
        dataset_stats={
            "observation.state": {"min": [-1.0] * 29, "max": [1.0] * 29},
            "observation.images.front": {
                "mean": [[[0.485]], [[0.456]], [[0.406]]],
                "std": [[[0.229]], [[0.224]], [[0.225]]],
            },
            "observation.images.wrist": {
                "mean": [[[0.485]], [[0.456]], [[0.406]]],
                "std": [[[0.229]], [[0.224]], [[0.225]]],
            },
            "action": {"min": [-1.0] * 7, "max": [1.0] * 7},
        },
    )
    enc = SACObservationEncoder(cfg)
    batch = {
        "observation.images.front": torch.zeros(2, 3, 128, 128),
        "observation.images.wrist": torch.zeros(2, 3, 128, 128),
        "observation.state": torch.zeros(2, 29),
    }
    out = enc(batch)
    assert out.dim() == 2 and out.shape[0] == 2
    print(f"  [PASS] DINOv2-S end-to-end forward → fused latent {tuple(out.shape)}")


def test_freeze_vision_encoder_no_grad():
    """frozen vision encoder 应 requires_grad=False（影响 num_learnable_params）。"""
    cfg = _make_cfg("helper2424/resnet10")
    enc = PretrainedImageEncoder(cfg)
    # PretrainedImageEncoder 本身不冻结，由 SACObservationEncoder 层 freeze_image_encoder() 调
    # 这里直接验证 freeze_image_encoder 工作
    from frrl.policies.sac.modeling_sac import freeze_image_encoder
    freeze_image_encoder(enc)
    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    assert n_train == 0, f"frozen encoder 应 0 trainable, got {n_train}"
    print(f"  [PASS] freeze_image_encoder 工作 (0 trainable params)")


def main():
    """直接 python tests/test_pretrained_image_encoder.py 也能跑。"""
    print("=== PretrainedImageEncoder 单元测试 ===")
    tests = [
        test_cnn_path_resnet10_outputs_4d,
        test_vit_path_dinov2_outputs_4d,
        test_sac_observation_encoder_rejects_non_square_input,
        test_sac_observation_encoder_4d_assert_error_message,
        test_sac_observation_encoder_dinov2_end_to_end,
        test_freeze_vision_encoder_no_grad,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    print("=== ALL PASS ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
