"""
Integration tests for local model checkpoint loading.

These tests verify that loader.detect_and_build can load checkpoints generated
by Mithridatium's own supported model builders.
"""

from __future__ import annotations

import torch

from mithridatium import loader


def test_detect_and_build_loads_resnet18_cifar_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "resnet18_cifar.pth"

    original_model, _ = loader.build_model("resnet18_cifar", num_classes=10)
    torch.save(original_model.state_dict(), checkpoint_path)

    loaded_model, feature_module = loader.detect_and_build(
        str(checkpoint_path),
        arch_hint="resnet18",
        num_classes=10,
    )

    assert isinstance(loaded_model, torch.nn.Module)
    assert feature_module is not None

    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        logits = loaded_model(x)

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_validate_model_accepts_supported_resnet18_cifar():
    model, _ = loader.build_model("resnet18_cifar", num_classes=10)

    # Should not raise.
    loader.validate_model(model, "resnet18_cifar", (3, 32, 32))