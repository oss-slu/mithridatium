import torch
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np

from mithridatium.defenses.strip import strip_scores, _resolve_threshold_and_verdict
from mithridatium.utils import get_preprocess_config


class MockModel(torch.nn.Module):
    """A simple model that maps flattened image inputs to 10-class logits."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


def _make_fake_loader(num_samples=200):
    """Create a fake CIFAR-10-shaped dataloader (3x32x32 images)."""
    data = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=64)


def test_strip_scores_basic():
    """Test that strip_scores returns correct structure and types."""
    print("Testing strip_scores basic structure...")

    torch.manual_seed(42)
    model = MockModel()
    config = get_preprocess_config("cifar10")

    # Patch dataloader_for to return our fake loader instead of real CIFAR-10
    fake_loader = _make_fake_loader(num_samples=200)
    with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
        mock_dl.return_value = (fake_loader, config)
        results = strip_scores(
            model,
            config,
            num_bases=5,
            num_perturbations=10,
            device="cpu",
            seed=42
        )

    # Validate structure
    assert results["defense"] == "strip"
    assert "entropies" in results
    assert "statistics" in results
    assert "parameters" in results
    assert "verdict" in results
    assert "thresholds" in results

    # Validate entropies list
    entropies = results["entropies"]
    assert isinstance(entropies, list), "Entropies should be a list"
    assert len(entropies) == 5, f"Expected 5 entropies, got {len(entropies)}"
    assert all(isinstance(e, float) for e in entropies), "All entropies should be floats"
    assert all(e >= 0 for e in entropies), "All entropies should be non-negative"

    # Validate statistics
    stats = results["statistics"]
    assert "entropy_mean" in stats
    assert "entropy_min" in stats
    assert "entropy_max" in stats
    assert "entropy_std" in stats
    assert stats["entropy_min"] <= stats["entropy_mean"] <= stats["entropy_max"]

    # Validate parameters reflect what we passed
    assert results["parameters"]["num_bases"] == 5
    assert results["parameters"]["num_perturbations"] == 10
    assert results["parameters"]["seed"] == 42

    # Validate verdict is one of the expected values
    assert results["verdict"] in ("likely clean", "likely backdoored")

    print(f"  Verdict: {results['verdict']}")
    print(f"  Entropy mean: {stats['entropy_mean']:.4f}")
    print(f"  Entropy std:  {stats['entropy_std']:.4f}")
    print("strip_scores basic test passed!")


def test_strip_scores_reproducibility():
    """Test that running with the same seed gives identical results."""
    print("Testing strip_scores reproducibility...")

    model = MockModel()
    config = get_preprocess_config("cifar10")
    fake_loader = _make_fake_loader(num_samples=200)

    results = []
    for _ in range(2):
        # Reset model to same state
        torch.manual_seed(0)
        model_copy = MockModel()

        with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
            mock_dl.return_value = (_make_fake_loader(num_samples=200), config)
            r = strip_scores(
                model_copy,
                config,
                num_bases=5,
                num_perturbations=10,
                device="cpu",
                seed=123
            )
        results.append(r)

    assert results[0]["entropies"] == results[1]["entropies"], (
        f"Results differ across runs:\n"
        f"  Run 1: {results[0]['entropies']}\n"
        f"  Run 2: {results[1]['entropies']}"
    )
    assert results[0]["statistics"]["entropy_mean"] == results[1]["statistics"]["entropy_mean"]

    print("strip_scores reproducibility test passed!")


def test_strip_scores_verdict_threshold():
    """Test that static threshold mode preserves expected verdict behavior."""
    print("Testing strip_scores static_mean threshold logic...")

    model = MockModel()
    config = get_preprocess_config("cifar10")

    # Run with a very low threshold (should flag as backdoored since any entropy > 0)
    with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
        mock_dl.return_value = (_make_fake_loader(num_samples=200), config)
        results_low = strip_scores(
            model, config,
            num_bases=5, num_perturbations=10,
            device="cpu", seed=42,
            threshold_mode="static_mean",
            entropy_mean_threshold=0.0
        )

    # Run with a very high threshold (should flag as clean)
    with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
        mock_dl.return_value = (_make_fake_loader(num_samples=200), config)
        results_high = strip_scores(
            model, config,
            num_bases=5, num_perturbations=10,
            device="cpu", seed=42,
            threshold_mode="static_mean",
            entropy_mean_threshold=100.0
        )

    assert results_low["verdict"] == "likely backdoored", (
        f"Expected 'likely backdoored' with threshold 0.0, got '{results_low['verdict']}'"
    )
    assert results_high["verdict"] == "likely clean", (
        f"Expected 'likely clean' with threshold 100.0, got '{results_high['verdict']}'"
    )

    print(f"  Low threshold verdict:  {results_low['verdict']} (threshold=0.0)")
    print(f"  High threshold verdict: {results_high['verdict']} (threshold=100.0)")
    print("strip_scores static_mean threshold test passed!")


def test_dynamic_mad_threshold_detects_low_entropy_tail():
    """Dynamic threshold should flag distributions with a strong low-entropy tail."""
    print("Testing dynamic MAD threshold low-tail detection...")

    # Mostly high entropies with a substantial low-entropy tail
    entropies = np.array([2.0] * 20 + [0.01] * 8, dtype=np.float64)
    decision = _resolve_threshold_and_verdict(
        entropies=entropies,
        num_classes=10,
        threshold_mode="dynamic_mad",
        entropy_mean_threshold=None,
        mad_scale=1.5,
        suspicious_fraction_threshold=0.20,
    )

    assert decision["verdict"] == "likely backdoored"
    assert decision["thresholds"]["mode"] == "dynamic_mad"
    assert decision["thresholds"]["suspicious_fraction"] >= 0.20

    print("dynamic MAD low-tail detection test passed!")


def test_dynamic_mad_threshold_clean_like_distribution():
    """Dynamic threshold should keep clean-like narrow distributions as clean."""
    print("Testing dynamic MAD threshold on clean-like distribution...")

    entropies = np.array([
        1.20, 1.22, 1.19, 1.21, 1.18, 1.24, 1.23, 1.20,
        1.19, 1.22, 1.21, 1.20, 1.23, 1.18, 1.21, 1.22,
    ], dtype=np.float64)

    decision = _resolve_threshold_and_verdict(
        entropies=entropies,
        num_classes=10,
        threshold_mode="dynamic_mad",
        entropy_mean_threshold=None,
        mad_scale=2.5,
        suspicious_fraction_threshold=0.20,
    )

    assert decision["verdict"] == "likely clean"
    assert decision["thresholds"]["suspicious_fraction"] < 0.20

    print("dynamic MAD clean-like distribution test passed!")


def test_dynamic_mad_high_entropy_safeguard_for_low_class_count():
    """Dynamic mode should catch high-entropy suspicious patterns on low-class tasks."""
    print("Testing dynamic MAD high-entropy safeguard...")

    # CIFAR-like class count with a generally high entropy profile and a small tail.
    entropies = np.array([
        1.35, 1.48, 1.49, 1.35, 1.37, 1.33, 1.24, 1.50,
        1.41, 1.43, 1.43, 1.55, 1.45, 1.44, 1.36, 1.44,
        1.54, 1.11, 1.48, 1.49, 1.50, 1.26, 1.33, 1.39,
        1.45, 1.32, 1.41, 1.49, 1.52, 1.47, 1.49, 1.52,
    ], dtype=np.float64)

    decision = _resolve_threshold_and_verdict(
        entropies=entropies,
        num_classes=10,
        threshold_mode="dynamic_mad",
        entropy_mean_threshold=None,
        mad_scale=2.5,
        suspicious_fraction_threshold=0.20,
    )

    assert decision["verdict"] == "likely backdoored"
    assert decision["thresholds"]["high_entropy_rule_triggered"] is True

    print("dynamic MAD high-entropy safeguard test passed!")


def test_dynamic_mad_flat_high_entropy_safeguard():
    """Dynamic mode should flag near-max entropy with very low variance."""
    print("Testing dynamic MAD flat high-entropy safeguard...")

    # Near log(10)=2.3026 and tightly clustered, with no low-entropy tail.
    entropies = np.array([
        2.19, 2.17, 2.11, 2.10, 2.09, 2.15, 2.09, 2.07,
        2.14, 2.17, 2.14, 2.16, 2.06, 2.13, 2.11, 2.17,
        2.07, 2.18, 2.07, 2.14, 2.07, 2.05, 2.09, 2.15,
        2.08, 2.12, 2.15, 2.13, 2.16, 2.17, 2.20, 2.15,
    ], dtype=np.float64)

    decision = _resolve_threshold_and_verdict(
        entropies=entropies,
        num_classes=10,
        threshold_mode="dynamic_mad",
        entropy_mean_threshold=None,
        mad_scale=2.5,
        suspicious_fraction_threshold=0.20,
    )

    assert decision["verdict"] == "likely backdoored"
    assert decision["thresholds"]["flat_high_entropy_rule_triggered"] is True

    print("dynamic MAD flat high-entropy safeguard test passed!")


if __name__ == "__main__":
    test_strip_scores_basic()
    test_strip_scores_reproducibility()
    test_strip_scores_verdict_threshold()
    test_dynamic_mad_threshold_detects_low_entropy_tail()
    test_dynamic_mad_threshold_clean_like_distribution()
    test_dynamic_mad_high_entropy_safeguard_for_low_class_count()
    test_dynamic_mad_flat_high_entropy_safeguard()
    print("\nAll strip_scores tests passed!")
