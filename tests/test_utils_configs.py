"""
Test canonical dataset configurations in utils.py.

This module tests that:
1. DATASET_CONFIGS contains correct canonical values for supported datasets
2. get_preprocess_config() returns proper PreprocessConfig objects
3. Unsupported datasets raise appropriate errors
4. Configuration values match published literature standards
"""

import pytest
from mithridatium.utils import get_preprocess_config, DATASET_CONFIGS, PreprocessConfig


class TestCanonicalConfigs:
    """Test canonical dataset configuration values."""
    
    def test_cifar10_canonical_stats(self):
        """Test CIFAR-10 has correct canonical normalization statistics."""
        # CIFAR-10 canonical values from literature
        expected_mean = (0.4914, 0.4822, 0.4465)
        expected_std = (0.2023, 0.1994, 0.2010)
        expected_size = (3, 32, 32)
        
        # Check DATASET_CONFIGS mapping
        config_data = DATASET_CONFIGS["cifar10"]
        assert config_data["input_size"] == expected_size
        assert config_data["mean"] == expected_mean
        assert config_data["std"] == expected_std
        assert config_data["normalize"] is True
        
        # Check PreprocessConfig object
        config = get_preprocess_config("cifar10")
        assert config.get_input_size() == expected_size
        assert config.get_mean() == expected_mean
        assert config.get_std() == expected_std
        assert config.get_normalize() is True
        assert config.get_dataset() == "cifar10"
    
    def test_cifar100_canonical_stats(self):
        """Test CIFAR-100 has correct canonical normalization statistics."""
        # CIFAR-100 canonical values from literature
        expected_mean = (0.5071, 0.4867, 0.4408)
        expected_std = (0.2675, 0.2565, 0.2761)
        expected_size = (3, 32, 32)
        
        # Check DATASET_CONFIGS mapping
        config_data = DATASET_CONFIGS["cifar100"]
        assert config_data["input_size"] == expected_size
        assert config_data["mean"] == expected_mean
        assert config_data["std"] == expected_std
        assert config_data["normalize"] is True
        
        # Check PreprocessConfig object
        config = get_preprocess_config("cifar100")
        assert config.get_input_size() == expected_size
        assert config.get_mean() == expected_mean
        assert config.get_std() == expected_std
        assert config.get_normalize() is True
        assert config.get_dataset() == "cifar100"
    
    def test_imagenet_canonical_stats(self):
        """Test ImageNet has correct canonical normalization statistics."""
        # ImageNet canonical values from torchvision/literature
        expected_mean = (0.485, 0.456, 0.406)
        expected_std = (0.229, 0.224, 0.225)
        expected_size = (3, 224, 224)
        
        # Check DATASET_CONFIGS mapping
        config_data = DATASET_CONFIGS["imagenet"]
        assert config_data["input_size"] == expected_size
        assert config_data["mean"] == expected_mean
        assert config_data["std"] == expected_std
        assert config_data["normalize"] is True
        
        # Check PreprocessConfig object
        config = get_preprocess_config("imagenet")
        assert config.get_input_size() == expected_size
        assert config.get_mean() == expected_mean
        assert config.get_std() == expected_std
        assert config.get_normalize() is True
        assert config.get_dataset() == "imagenet"
    
    def test_case_insensitive_dataset_names(self):
        """Test that dataset names are case-insensitive."""
        # Test various case combinations
        for dataset_name in ["CIFAR10", "Cifar10", "cifar10", "CiFaR10"]:
            config = get_preprocess_config(dataset_name)
            assert config.get_dataset() == "cifar10"
        
        for dataset_name in ["CIFAR100", "Cifar100", "cifar100", "CiFaR100"]:
            config = get_preprocess_config(dataset_name)
            assert config.get_dataset() == "cifar100"
        
        for dataset_name in ["IMAGENET", "ImageNet", "imagenet", "ImAgEnEt"]:
            config = get_preprocess_config(dataset_name)
            assert config.get_dataset() == "imagenet"
    
    def test_whitespace_handling(self):
        """Test that dataset names handle whitespace correctly."""
        # Test with leading/trailing whitespace
        config = get_preprocess_config("  cifar10  ")
        assert config.get_dataset() == "cifar10"
        
        config = get_preprocess_config("\tcifar100\n")
        assert config.get_dataset() == "cifar100"
    
    def test_unsupported_dataset_error(self):
        """Test that unsupported datasets raise ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            get_preprocess_config("mnist")
        
        error_msg = str(exc_info.value)
        assert "mnist" in error_msg
        assert "Unsupported dataset" in error_msg
        assert "cifar10" in error_msg  # Should list supported datasets
        assert "cifar100" in error_msg
        assert "imagenet" in error_msg
    
    def test_preprocess_config_default_values(self):
        """Test that PreprocessConfig has correct default values."""
        for dataset in ["cifar10", "cifar100", "imagenet"]:
            config = get_preprocess_config(dataset)
            
            # Common defaults across all datasets
            assert config.get_channels_first() is True
            assert config.get_value_range() == (0.0, 1.0)
            assert config.get_normalize() is True
            assert config.get_ops() == []
    
    def test_all_supported_datasets_in_mapping(self):
        """Test that all datasets mentioned in error messages are in DATASET_CONFIGS."""
        try:
            get_preprocess_config("invalid_dataset")
        except ValueError as e:
            error_msg = str(e)
            # Extract supported datasets from error message
            # Message format: "Supported datasets: cifar10, cifar100, imagenet"
            if "Supported datasets:" in error_msg:
                supported_part = error_msg.split("Supported datasets:")[1].strip()
                mentioned_datasets = [ds.strip() for ds in supported_part.split(",")]
                
                # Verify all mentioned datasets exist in DATASET_CONFIGS
                for dataset in mentioned_datasets:
                    assert dataset in DATASET_CONFIGS, f"Dataset {dataset} mentioned in error but not in DATASET_CONFIGS"


class TestDatasetConfigsCompleteness:
    """Test that DATASET_CONFIGS mapping is complete and well-formed."""
    
    def test_dataset_configs_structure(self):
        """Test that DATASET_CONFIGS has proper structure."""
        required_keys = {"input_size", "mean", "std", "normalize"}
        
        for dataset_name, config in DATASET_CONFIGS.items():
            # Check all required keys present
            assert required_keys.issubset(config.keys()), f"Missing keys in {dataset_name} config"
            
            # Check types and shapes
            assert isinstance(config["input_size"], tuple)
            assert len(config["input_size"]) == 3  # (C, H, W)
            assert all(isinstance(x, int) and x > 0 for x in config["input_size"])
            
            assert isinstance(config["mean"], tuple)
            assert len(config["mean"]) == 3  # (R, G, B)
            assert all(isinstance(x, float) and 0 <= x <= 1 for x in config["mean"])
            
            assert isinstance(config["std"], tuple)
            assert len(config["std"]) == 3  # (R, G, B)
            assert all(isinstance(x, float) and x > 0 for x in config["std"])
            
            assert isinstance(config["normalize"], bool)
    
    def test_cifar_datasets_have_32x32_size(self):
        """Test that CIFAR datasets have correct 32x32 input size."""
        for dataset in ["cifar10", "cifar100"]:
            config = DATASET_CONFIGS[dataset]
            assert config["input_size"] == (3, 32, 32), f"{dataset} should be 3x32x32"
    
    def test_imagenet_has_224x224_size(self):
        """Test that ImageNet has correct 224x224 input size."""
        config = DATASET_CONFIGS["imagenet"]
        assert config["input_size"] == (3, 224, 224), "ImageNet should be 3x224x224"
    
    def test_normalization_stats_reasonable_ranges(self):
        """Test that mean/std values are in reasonable ranges for image data."""
        for dataset_name, config in DATASET_CONFIGS.items():
            # Mean values should be between 0 and 1 for normalized images
            for channel_mean in config["mean"]:
                assert 0.0 <= channel_mean <= 1.0, f"{dataset_name} mean {channel_mean} out of range [0,1]"
            
            # Std values should be positive and reasonable (typically 0.1-0.5 for image data)
            for channel_std in config["std"]:
                assert 0.05 <= channel_std <= 0.5, f"{dataset_name} std {channel_std} out of reasonable range [0.05,0.5]"


class TestPreprocessConfigMethods:
    """Test PreprocessConfig class methods and functionality."""
    
    def test_preprocess_config_getters(self):
        """Test all getter methods work correctly."""
        config = get_preprocess_config("cifar10")
        
        # Test all getter methods
        assert config.get_input_size() == (3, 32, 32)
        assert config.get_channels_first() is True
        assert config.get_value_range() == (0.0, 1.0)
        assert config.get_mean() == (0.4914, 0.4822, 0.4465)
        assert config.get_std() == (0.2023, 0.1994, 0.2010)
        assert config.get_normalize() is True
        assert config.get_ops() == []
        assert config.get_dataset() == "cifar10"
    
    def test_preprocess_config_setters(self):
        """Test setter methods work correctly."""
        config = get_preprocess_config("cifar10")
        
        # Test setters
        config.set_input_size((3, 64, 64))
        assert config.get_input_size() == (3, 64, 64)
        
        config.set_channels_first(False)
        assert config.get_channels_first() is False
        
        config.set_value_range((-1.0, 1.0))
        assert config.get_value_range() == (-1.0, 1.0)
        
        config.set_mean((0.5, 0.5, 0.5))
        assert config.get_mean() == (0.5, 0.5, 0.5)
        
        config.set_std((0.25, 0.25, 0.25))
        assert config.get_std() == (0.25, 0.25, 0.25)
        
        config.set_normalize(False)
        assert config.get_normalize() is False
        
        config.set_ops(["resize:64", "crop:32"])
        assert config.get_ops() == ["resize:64", "crop:32"]
        
        config.set_dataset("custom")
        assert config.get_dataset() == "custom"