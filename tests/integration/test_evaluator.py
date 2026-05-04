"""
Integration tests for mithridatium.evaluator.

These tests use:
- a real supported model
- a real feature hook
- a real DataLoader

They use synthetic tensors instead of CIFAR-10 because evaluator.py only needs
a model/data contract, not a real dataset.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

import mithridatium.evaluator as evaluator
from mithridatium import loader


pytestmark = []


def _make_synthetic_loader(
    *,
    num_samples: int = 16,
    batch_size: int = 4,
    num_classes: int = 10,
) -> DataLoader:
    torch.manual_seed(0)

    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.arange(num_samples) % num_classes

    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_extract_embeddings_returns_embeddings_and_labels():
    model, feature_module = loader.build_model("resnet18_cifar", num_classes=10)
    dataloader = _make_synthetic_loader()

    embs, labels = evaluator.extract_embeddings(model, dataloader, feature_module)

    assert isinstance(embs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    assert embs.shape[0] == 16
    assert labels.shape == (16,)

    # ResNet avgpool feature hook should flatten to [N, D].
    assert embs.ndim == 2
    assert embs.shape[1] > 0

    assert torch.isfinite(embs).all()


def test_evaluate_returns_loss_and_accuracy():
    model, _ = loader.build_model("resnet18_cifar", num_classes=10)
    dataloader = _make_synthetic_loader()

    loss, accy = evaluator.evaluate(model, dataloader)

    assert isinstance(loss, float)
    assert isinstance(accy, float)

    assert loss >= 0.0
    assert 0.0 <= accy <= 1.0