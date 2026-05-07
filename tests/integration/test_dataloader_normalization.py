"""
Integration tests for mithridatium.utils.dataloader_for using CIFAR-10.

CIFAR-10 is the universal integration dataset for Mithridatium. These tests
verify that the real dataloader, real transforms, and PreprocessConfig agree.
"""

from __future__ import annotations

import pytest
import torch

from mithridatium import utils


pytestmark = [pytest.mark.integration, pytest.mark.requires_data]


@pytest.mark.parametrize("split", ["train", "test"])
def test_cifar10_dataloader_creation(require_cifar10, split):
    dataloader, config = utils.dataloader_for("cifar10", split, batch_size=16)

    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.batch_size == 16

    assert config.get_dataset() == "cifar10"
    assert config.get_num_classes() == 10
    assert config.get_input_size() == (3, 32, 32)
    assert config.get_channels_first() is True
    assert config.get_normalize() is True


@pytest.mark.parametrize("split", ["train", "test"])
def test_cifar10_tensor_shapes_match_config(require_cifar10, split):
    dataloader, config = utils.dataloader_for("cifar10", split, batch_size=16)
    images, labels = next(iter(dataloader))

    assert images.dtype == torch.float32
    assert labels.dtype == torch.long

    assert images.shape[0] == 16
    assert labels.shape == (16,)

    # Critical integration check:
    # The actual transformed tensor shape should match the advertised config.
    assert tuple(images.shape[1:]) == tuple(config.get_input_size())


def test_cifar10_normalization_produces_finite_values(require_cifar10):
    dataloader, _ = utils.dataloader_for("cifar10", "test", batch_size=16)
    images, _ = next(iter(dataloader))

    assert torch.isfinite(images).all()


def test_cifar10_normalized_values_are_not_raw_zero_to_one(require_cifar10):
    dataloader, _ = utils.dataloader_for("cifar10", "test", batch_size=16)
    images, _ = next(iter(dataloader))

    # After normalization, values should usually move outside raw [0, 1].
    assert images.min().item() < 0.0 or images.max().item() > 1.0


def test_cifar10_inverse_normalization_recovers_image_range(require_cifar10):
    dataloader, config = utils.dataloader_for("cifar10", "test", batch_size=16)
    normalized_batch, _ = next(iter(dataloader))

    mean = torch.tensor(config.get_mean()).view(1, 3, 1, 1)
    std = torch.tensor(config.get_std()).view(1, 3, 1, 1)

    denormalized = normalized_batch * std + mean

    assert denormalized.min().item() >= -0.01
    assert denormalized.max().item() <= 1.01


def test_cifar10_case_insensitive_inputs(require_cifar10):
    dataloader, config = utils.dataloader_for("  CIFAR10  ", "  TEST  ", batch_size=8)
    images, labels = next(iter(dataloader))

    assert config.get_dataset() == "cifar10"
    assert images.shape[0] == 8
    assert labels.shape[0] == 8


def test_invalid_dataset_error():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        utils.dataloader_for("mnist", "test", batch_size=4)


def test_invalid_split_error():
    with pytest.raises(ValueError, match="Invalid split"):
        utils.dataloader_for("cifar10", "validation", batch_size=4)


def test_dataloader_performance_settings_are_enabled(require_cifar10):
    dataloader, _ = utils.dataloader_for("cifar10", "test", batch_size=16)

    assert dataloader.pin_memory is True
    assert dataloader.num_workers >= 1