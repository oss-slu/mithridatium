"""
Unit tests for mithridatium.evaluator (evaluate and extract_embeddings).

All models and data are tiny and built in memory: CPU-only, no downloads.
"""

from __future__ import annotations

import math
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mithridatium.evaluator import evaluate, extract_embeddings


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_linear_model(weight: list[list[float]]) -> nn.Sequential:
    """A 1-layer model (2 numbers in, 2 scores out) with hand-set weights."""
    model = nn.Sequential(nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(weight))
        model[0].bias.zero_()
    return model


def _make_simple_loader() -> DataLoader:
    """4 samples. Sample [1, 0] has label 0, sample [0, 1] has label 1."""
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2)


# --------------------------------------------------------------------------
# evaluate()
# --------------------------------------------------------------------------

def test_evaluate_accuracy_is_one_for_always_correct_model():
    # Identity weights: input [1, 0] -> scores [1, 0] -> predicts class 0, etc.
    model = _make_linear_model([[1.0, 0.0], [0.0, 1.0]])

    _, accuracy = evaluate(model, _make_simple_loader())

    assert accuracy == 1.0


def test_evaluate_accuracy_is_zero_for_always_wrong_model():
    # Swapped weights: input [1, 0] -> scores [0, 1] -> predicts class 1 (wrong).
    model = _make_linear_model([[0.0, 1.0], [1.0, 0.0]])

    _, accuracy = evaluate(model, _make_simple_loader())

    assert accuracy == 0.0


def test_evaluate_loss_is_finite_and_non_negative():
    model = _make_linear_model([[1.0, 0.0], [0.0, 1.0]])

    loss, _ = evaluate(model, _make_simple_loader())

    assert math.isfinite(loss)
    assert loss >= 0.0


# --------------------------------------------------------------------------
# extract_embeddings()
# --------------------------------------------------------------------------

def test_extract_embeddings_shapes_are_n_by_d_and_n():
    torch.manual_seed(0)
    n = 10  # not divisible by batch_size=4, so the last batch is smaller
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    feature_module = model[0]  # outputs 3 numbers per sample -> D = 3
    x = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    embs, labels = extract_embeddings(model, loader, feature_module)

    assert embs.shape == (n, 3)
    assert labels.shape == (n,)


def test_extract_embeddings_flattens_conv_output():
    torch.manual_seed(0)
    n = 6
    # Conv2d on 3x8x8 images with a 3x3 window gives [4, 6, 6] per image.
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 6 * 6, 2),
    )
    feature_module = model[0]
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, 2, (n,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    embs, labels = extract_embeddings(model, loader, feature_module)

    assert embs.shape == (n, 4 * 6 * 6)  # [N, 144], not [N, 4, 6, 6]
    assert labels.shape == (n,)


def test_extract_embeddings_removes_forward_hook():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    feature_module = model[0]
    x = torch.randn(5, 4)
    y = torch.randint(0, 2, (5,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    extract_embeddings(model, loader, feature_module)

    assert len(feature_module._forward_hooks) == 0