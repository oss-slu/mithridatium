import pytest
from mithridatium.utils import get_preprocess_config

def test_get_preprocess_config():
    # Use a known dataset for the test (e.g., cifar10)
    dataset_name = "cifar10"
    
    # Load the preprocessing config for the dataset
    config = get_preprocess_config(dataset_name)
    
    # Assertions based on the expected preprocessing config for CIFAR-10
    assert config.input_size == (3, 32, 32)  # CIFAR-10 has 32x32 RGB images
    assert config.channels_first is True      # CIFAR-10 uses NCHW format
    assert config.value_range == (0.0, 1.0)  # Normalization range
    assert config.mean == (0.4914, 0.4822, 0.4465)  # CIFAR-10 dataset mean
    assert config.std == (0.2023, 0.1994, 0.2010)   # CIFAR-10 dataset standard deviation
    assert config.ops == []  # No additional operations are needed for CIFAR-10