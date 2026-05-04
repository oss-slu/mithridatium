"""
Integration-style tests for mithridatium.loader_hf.

These tests monkeypatch Hugging Face classes so they do not require internet
or downloaded model files, but still verify the wrapper contract.
"""

from __future__ import annotations

import torch

from mithridatium import loader_hf


class FakeHFConfig:
    num_labels = 7


class FakeHFOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class FakeHFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeHFConfig()
        self.proj = torch.nn.Linear(3, 7)

    def forward(self, pixel_values: torch.Tensor):
        pooled = pixel_values.mean(dim=(2, 3))
        return FakeHFOutput(self.proj(pooled))


class FakeProcessor:
    image_mean = [0.5, 0.4, 0.3]
    image_std = [0.2, 0.2, 0.2]
    size = {"height": 224, "width": 224}


def _patch_hf(monkeypatch):
    monkeypatch.setattr(
        loader_hf.AutoModelForImageClassification,
        "from_pretrained",
        lambda _model_id: FakeHFModel(),
    )
    monkeypatch.setattr(
        loader_hf.AutoImageProcessor,
        "from_pretrained",
        lambda _model_id: FakeProcessor(),
    )


def test_build_huggingface_model_returns_classifier_wrapper(monkeypatch):
    _patch_hf(monkeypatch)

    model, feature_module = loader_hf.build_huggingface_model("fake/resnet")

    assert isinstance(model, loader_hf.HFImageClassifier)
    assert feature_module is None

    assert model.num_classes == 7
    assert model.supports_feature_extraction() is False
    assert model.get_feature_module() is None


def test_huggingface_wrapper_forward_returns_logits(monkeypatch):
    _patch_hf(monkeypatch)

    model = loader_hf.HFImageClassifier("fake/resnet")

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()


def test_huggingface_wrapper_preprocess_config_uses_processor_metadata(monkeypatch):
    _patch_hf(monkeypatch)

    model = loader_hf.HFImageClassifier("fake/resnet")
    config = model.get_preprocess_config(fallback_dataset="cifar10_for_imagenet")

    assert config.get_input_size() == (3, 224, 224)
    assert config.get_mean() == (0.5, 0.4, 0.3)
    assert config.get_std() == (0.2, 0.2, 0.2)

    # Regression check:
    # HF preprocess config should use the model's real class count.
    assert config.get_num_classes() == 7

    assert config.get_dataset() == "cifar10_for_imagenet"
    assert config.get_normalize() is True