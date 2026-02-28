import torch
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from mithridatium.defenses.strip import strip_scores
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
    """Test that the verdict correctly depends on the threshold."""
    print("Testing strip_scores verdict threshold logic...")

    model = MockModel()
    config = get_preprocess_config("cifar10")

    # Run with a very low threshold (should flag as backdoored since any entropy > 0)
    with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
        mock_dl.return_value = (_make_fake_loader(num_samples=200), config)
        results_low = strip_scores(
            model, config,
            num_bases=5, num_perturbations=10,
            device="cpu", seed=42,
            entropy_mean_threshold=0.0
        )

    # Run with a very high threshold (should flag as clean)
    with patch("mithridatium.defenses.strip.utils.dataloader_for") as mock_dl:
        mock_dl.return_value = (_make_fake_loader(num_samples=200), config)
        results_high = strip_scores(
            model, config,
            num_bases=5, num_perturbations=10,
            device="cpu", seed=42,
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
    print("strip_scores verdict threshold test passed!")


if __name__ == "__main__":
    test_strip_scores_basic()
    test_strip_scores_reproducibility()
    test_strip_scores_verdict_threshold()
    print("\nAll strip_scores tests passed!")
