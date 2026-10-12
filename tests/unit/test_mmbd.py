"""
Unit tests for MMBD defense and class probing configuration.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mithridatium.defenses.mmbd import run_mmbd
from mithridatium.utils import get_preprocess_config


pytestmark = pytest.mark.unit


class SimpleModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(3 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return self.fc(flat)


class MockConfig:
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.input_size = (3, 4, 4)

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_mean(self) -> list[float]:
        return [0.5, 0.5, 0.5]

    def get_std(self) -> list[float]:
        return [0.5, 0.5, 0.5]

    def get_dataset(self) -> str:
        return "cifar10"


@pytest.fixture(autouse=True)
def fast_mmbd(monkeypatch):
    """Speed up MMBD optimization loop for unit testing without GPU/training."""
    import mithridatium.defenses.mmbd as mmbd_module

    monkeypatch.setattr(mmbd_module, "NUM_IMAGES", 2)
    monkeypatch.setattr(mmbd_module, "NSTEP", 1)


def test_run_mmbd_probes_explicit_classes(monkeypatch):
    import mithridatium.defenses.mmbd as mmbd_module
    monkeypatch.setattr(mmbd_module, "NUM_IMAGES", 2)

    model = SimpleModel(num_classes=10)
    config = MockConfig(num_classes=10)

    # Fast forward: monkeypatch inner NSTEP to 1 by overriding range or small run
    results = run_mmbd(model, config, device=torch.device("cpu"), classes=[1, 4])

    assert results["classes_probed"] == [1, 4]
    assert results["parameters"]["classes_probed"] == [1, 4]
    assert len(results["per_class_scores"]) == 2
    assert "verdict" in results
    assert "p_value" in results


def test_run_mmbd_probes_config_classes(monkeypatch):
    import mithridatium.defenses.mmbd as mmbd_module
    monkeypatch.setattr(mmbd_module, "NUM_IMAGES", 2)

    model = SimpleModel(num_classes=10)
    config = MockConfig(num_classes=10)
    setattr(config, "mmbd_classes", [0, 2, 5])

    results = run_mmbd(model, config, device=torch.device("cpu"))

    assert results["classes_probed"] == [0, 2, 5]
    assert results["parameters"]["classes_probed"] == [0, 2, 5]
    assert len(results["per_class_scores"]) == 3


def test_run_mmbd_defaults_to_first_five_classes(monkeypatch):
    import mithridatium.defenses.mmbd as mmbd_module
    monkeypatch.setattr(mmbd_module, "NUM_IMAGES", 2)

    model = SimpleModel(num_classes=10)
    config = MockConfig(num_classes=10)

    results = run_mmbd(model, config, device=torch.device("cpu"))

    assert results["classes_probed"] == [0, 1, 2, 3, 4]
    assert results["parameters"]["classes_probed"] == [0, 1, 2, 3, 4]
    assert len(results["per_class_scores"]) == 5


def test_run_mmbd_validates_out_of_range_classes():
    model = SimpleModel(num_classes=10)
    config = MockConfig(num_classes=10)

    with pytest.raises(ValueError, match="out of range"):
        run_mmbd(model, config, device=torch.device("cpu"), classes=[0, 10])

    with pytest.raises(ValueError, match="out of range"):
        run_mmbd(model, config, device=torch.device("cpu"), classes=[-1, 2])


def test_run_mmbd_validates_duplicate_classes():
    model = SimpleModel(num_classes=10)
    config = MockConfig(num_classes=10)

    with pytest.raises(ValueError, match="Duplicate class indices"):
        run_mmbd(model, config, device=torch.device("cpu"), classes=[1, 1])


def test_run_mmbd_validates_empty_classes():
    model = SimpleModel(num_classes=10)
    config = MockConfig(num_classes=10)

    with pytest.raises(ValueError, match="at least one class"):
        run_mmbd(model, config, device=torch.device("cpu"), classes=[])
