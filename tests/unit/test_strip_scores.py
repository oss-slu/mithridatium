"""
Unit tests for STRIP score computation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mithridatium.defenses.strip import strip_scores
from mithridatium.utils import get_preprocess_config


pytestmark = pytest.mark.unit


class MockModel(torch.nn.Module):
    """Simple model that maps flattened CIFAR-shaped inputs to logits."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


def _make_fake_loader(num_samples: int = 200, num_classes: int = 10) -> DataLoader:
    data = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=64)


def test_strip_scores_returns_expected_contract():
    torch.manual_seed(42)

    model = MockModel()
    config = get_preprocess_config("cifar10")
    fake_loader = _make_fake_loader()

    with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
        mock_dl.return_value = (fake_loader, config)
        results = strip_scores(
            model,
            config,
            num_bases=5,
            num_perturbations=10,
            device="cpu",
            seed=42,
        )

    assert results["defense"] == "strip"
    assert "entropies" in results
    assert "statistics" in results
    assert "parameters" in results
    assert "verdict" in results
    assert "thresholds" in results

    entropies = results["entropies"]
    assert isinstance(entropies, list)
    assert len(entropies) == 5
    assert all(isinstance(e, float) for e in entropies)
    assert all(e >= 0.0 for e in entropies)

    stats = results["statistics"]
    assert stats["entropy_min"] <= stats["entropy_mean"] <= stats["entropy_max"]
    assert stats["entropy_std"] >= 0.0

    assert results["parameters"]["num_bases"] == 5
    assert results["parameters"]["num_perturbations"] == 10
    assert results["parameters"]["seed"] == 42

    assert results["verdict"] in ("likely clean", "likely backdoored")


def test_strip_scores_reproducible_with_same_seed():
    config = get_preprocess_config("cifar10")

    outputs = []

    for _ in range(2):
        torch.manual_seed(0)
        model = MockModel()

        with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
            mock_dl.return_value = (_make_fake_loader(), config)
            result = strip_scores(
                model,
                config,
                num_bases=5,
                num_perturbations=10,
                device="cpu",
                seed=123,
            )

        outputs.append(result)

    assert outputs[0]["entropies"] == outputs[1]["entropies"]
    assert outputs[0]["statistics"]["entropy_mean"] == outputs[1]["statistics"]["entropy_mean"]


def test_strip_scores_verdict_threshold_logic():
    # Default STRIP mode is dynamic_mad, which ignores entropy_mean_threshold.
    # This test covers static_mean, where the mean is compared to that cutoff.
    model = MockModel()
    config = get_preprocess_config("cifar10")

    with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
        mock_dl.return_value = (_make_fake_loader(), config)
        low_threshold = strip_scores(
            model,
            config,
            num_bases=5,
            num_perturbations=10,
            device="cpu",
            seed=42,
            threshold_mode="static_mean",
            entropy_mean_threshold=0.0,
        )

    with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
        mock_dl.return_value = (_make_fake_loader(), config)
        high_threshold = strip_scores(
            model,
            config,
            num_bases=5,
            num_perturbations=10,
            device="cpu",
            seed=42,
            threshold_mode="static_mean",
            entropy_mean_threshold=100.0,
        )

    assert low_threshold["thresholds"]["mode"] == "static_mean"
    assert high_threshold["thresholds"]["mode"] == "static_mean"
    assert low_threshold["verdict"] == "likely backdoored"
    assert high_threshold["verdict"] == "likely clean"