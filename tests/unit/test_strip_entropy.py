"""
Unit tests for STRIP entropy helpers.
"""

from __future__ import annotations

import pytest
import torch

from mithridatium.defenses.strip import prediction_entropy


pytestmark = pytest.mark.unit


def test_prediction_entropy_uniform_distribution():
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    entropy = prediction_entropy(logits)

    expected = torch.tensor([torch.log(torch.tensor(4.0))])
    torch.testing.assert_close(entropy, expected, rtol=1e-4, atol=1e-4)


def test_prediction_entropy_confident_distribution_near_zero():
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])

    entropy = prediction_entropy(logits)

    expected = torch.tensor([0.0])
    torch.testing.assert_close(entropy, expected, rtol=1e-4, atol=1e-4)


def test_prediction_entropy_returns_one_value_per_sample():
    logits = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [10.0, 0.0, 0.0],
        ]
    )

    entropy = prediction_entropy(logits)

    assert entropy.shape == (2,)
    assert torch.isfinite(entropy).all()