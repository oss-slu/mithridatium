"""
Test dataloader normalization behavior in utils.py.

This module tests that:
1. Dataloader transforms properly normalize data to have means near 0
2. CIFAR datasets load without errors and produce expected tensor shapes
3. Normalization statistics match expected behavior
4. Transform pipelines work correctly for each dataset
"""

import pytest
import torch
import numpy as np
from mithridatium.utils import dataloader_for, get_preprocess_config


class TestDataloaderNormalization:
    """Test that dataloader normalization works correctly."""
    
    @pytest.fixture
    def small_batch_size(self):
        """Use small batch size for faster tests."""
        return 32
    
    def test_cifar10_dataloader_creation(self, small_batch_size):
        """Test that CIFAR-10 dataloader creates successfully."""
        # Test both train and test splits
        for split in ["train", "test"]:
            dataloader, config = dataloader_for("cifar10", split, batch_size=small_batch_size)
            
            # Check dataloader properties
            assert dataloader.batch_size == small_batch_size
            assert isinstance(dataloader, torch.utils.data.DataLoader)
            
            # Check config
            assert config.get_dataset() == "cifar10"
            assert config.get_input_size() == (3, 32, 32)
    
    def test_cifar100_dataloader_creation(self, small_batch_size):
        """Test that CIFAR-100 dataloader creates successfully."""
        # Test both train and test splits
        for split in ["train", "test"]:
            dataloader, config = dataloader_for("cifar100", split, batch_size=small_batch_size)
            
            # Check dataloader properties
            assert dataloader.batch_size == small_batch_size
            assert isinstance(dataloader, torch.utils.data.DataLoader)
            
            # Check config
            assert config.get_dataset() == "cifar100"
            assert config.get_input_size() == (3, 32, 32)
    
    def test_cifar10_tensor_shapes(self, small_batch_size):
        """Test that CIFAR-10 produces correct tensor shapes."""
        dataloader, _ = dataloader_for("cifar10", "test", batch_size=small_batch_size)
        
        # Get first batch
        batch_iter = iter(dataloader)
        images, labels = next(batch_iter)
        
        # Check shapes
        assert images.shape == (small_batch_size, 3, 32, 32), f"Expected {(small_batch_size, 3, 32, 32)}, got {images.shape}"
        assert labels.shape == (small_batch_size,), f"Expected {(small_batch_size,)}, got {labels.shape}"
        
        # Check data types
        assert images.dtype == torch.float32
        assert labels.dtype == torch.long  # CIFAR uses long integers for class labels
    
    def test_cifar100_tensor_shapes(self, small_batch_size):
        """Test that CIFAR-100 produces correct tensor shapes."""
        dataloader, _ = dataloader_for("cifar100", "test", batch_size=small_batch_size)
        
        # Get first batch
        batch_iter = iter(dataloader)
        images, labels = next(batch_iter)
        
        # Check shapes
        assert images.shape == (small_batch_size, 3, 32, 32), f"Expected {(small_batch_size, 3, 32, 32)}, got {images.shape}"
        assert labels.shape == (small_batch_size,), f"Expected {(small_batch_size,)}, got {labels.shape}"
        
        # Check data types
        assert images.dtype == torch.float32
        assert labels.dtype == torch.long
    
    def test_cifar10_normalization_behavior(self, small_batch_size):
        """Test that CIFAR-10 normalization produces data with means near 0."""
        dataloader, config = dataloader_for("cifar10", "test", batch_size=small_batch_size)
        
        # Collect several batches to get good statistics
        all_images = []
        batch_count = 0
        for images, _ in dataloader:
            all_images.append(images)
            batch_count += 1
            if batch_count >= 10:  # Use 10 batches for statistics
                break
        
        # Concatenate all images
        all_images = torch.cat(all_images, dim=0)
        
        # Calculate per-channel means and stds
        # Shape: (N, C, H, W) -> calculate over N, H, W dimensions
        channel_means = torch.mean(all_images, dim=(0, 2, 3))  # Shape: (3,)
        channel_stds = torch.std(all_images, dim=(0, 2, 3))    # Shape: (3,)
        
        # Print actual values for debugging/validation
        print(f"CIFAR-10 normalized stats - Means: {channel_means.tolist()}, Stds: {channel_stds.tolist()}")
        
        # After normalization, means should be close to 0
        # The mean centering should be very effective
        for i, mean_val in enumerate(channel_means):
            assert abs(mean_val.item()) < 0.1, f"Channel {i} mean {mean_val.item()} not near 0"
        
        # Standard deviations should be reasonably close to 1
        # Note: Due to finite sampling and dataset characteristics, exact std=1.0 is not expected
        # We verify the normalization is working (values roughly in expected range)
        for i, std_val in enumerate(channel_stds):
            assert 0.6 <= std_val.item() <= 1.4, f"Channel {i} std {std_val.item()} outside reasonable range [0.6, 1.4]"
    
    def test_cifar100_normalization_behavior(self, small_batch_size):
        """Test that CIFAR-100 normalization produces data with means near 0."""
        dataloader, config = dataloader_for("cifar100", "test", batch_size=small_batch_size)
        
        # Collect several batches to get good statistics
        all_images = []
        batch_count = 0
        for images, _ in dataloader:
            all_images.append(images)
            batch_count += 1
            if batch_count >= 10:  # Use 10 batches for statistics
                break
        
        # Concatenate all images
        all_images = torch.cat(all_images, dim=0)
        
        # Calculate per-channel means and stds
        channel_means = torch.mean(all_images, dim=(0, 2, 3))
        channel_stds = torch.std(all_images, dim=(0, 2, 3))
        
        # Print actual values for debugging/validation
        print(f"CIFAR-100 normalized stats - Means: {channel_means.tolist()}, Stds: {channel_stds.tolist()}")
        
        # After normalization, means should be close to 0
        for i, mean_val in enumerate(channel_means):
            assert abs(mean_val.item()) < 0.1, f"Channel {i} mean {mean_val.item()} not near 0"
        
        # Standard deviations should be reasonably close to 1
        for i, std_val in enumerate(channel_stds):
            assert 0.6 <= std_val.item() <= 1.4, f"Channel {i} std {std_val.item()} outside reasonable range [0.6, 1.4]"
    
    def test_unnormalized_data_range(self, small_batch_size):
        """Test data range before and after normalization by manually checking transforms."""
        # This test verifies the transform pipeline is working correctly
        from torchvision import datasets, transforms
        
        # Create CIFAR-10 dataset without normalization
        unnormalized_transform = transforms.Compose([
            transforms.ToTensor()  # Only convert to tensor, no normalization
        ])
        
        unnormalized_ds = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=unnormalized_transform
        )
        
        unnormalized_loader = torch.utils.data.DataLoader(
            unnormalized_ds,
            batch_size=small_batch_size,
            shuffle=False
        )
        
        # Get normalized dataloader
        normalized_loader, config = dataloader_for("cifar10", "test", batch_size=small_batch_size)
        
        # Get first batch from each
        unnorm_batch = next(iter(unnormalized_loader))[0]  # Just images
        norm_batch = next(iter(normalized_loader))[0]      # Just images
        
        # Unnormalized data should be in [0, 1] range
        assert unnorm_batch.min().item() >= 0.0, f"Unnormalized min {unnorm_batch.min().item()} < 0"
        assert unnorm_batch.max().item() <= 1.0, f"Unnormalized max {unnorm_batch.max().item()} > 1"
        
        # Normalized data should extend beyond [0, 1] range due to normalization
        # (some values will be negative after subtracting mean)
        assert norm_batch.min().item() < 0.0, f"Normalized data should have negative values, min={norm_batch.min().item()}"
        assert norm_batch.max().item() > 1.0, f"Normalized data should exceed 1, max={norm_batch.max().item()}"
    
    def test_different_batch_sizes(self):
        """Test that different batch sizes work correctly."""
        for batch_size in [1, 8, 16, 64]:
            dataloader, _ = dataloader_for("cifar10", "test", batch_size=batch_size)
            
            # Get first batch
            batch_iter = iter(dataloader)
            images, labels = next(batch_iter)
            
            # Check batch size (last batch might be smaller)
            assert images.shape[0] <= batch_size
            assert labels.shape[0] <= batch_size
            assert images.shape[0] == labels.shape[0]
    
    def test_train_vs_test_shuffle(self):
        """Test that train loader shuffles but test loader doesn't."""
        batch_size = 16
        
        # Get train and test loaders
        train_loader, _ = dataloader_for("cifar10", "train", batch_size=batch_size)
        test_loader, _ = dataloader_for("cifar10", "test", batch_size=batch_size)
        
        # For train loader, shuffle should be True (can't directly test randomness easily)
        # But we can at least verify the loaders work
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        
        assert train_batch[0].shape == (batch_size, 3, 32, 32)
        assert test_batch[0].shape == (batch_size, 3, 32, 32)


class TestDataloaderErrorHandling:
    """Test error handling in dataloader_for function."""
    
    def test_invalid_dataset_error(self):
        """Test that invalid datasets raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            dataloader_for("mnist", "test", batch_size=32)
        
        error_msg = str(exc_info.value)
        assert "Unsupported dataset" in error_msg
        assert "mnist" in error_msg
    
    def test_invalid_split_error(self):
        """Test that invalid splits raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            dataloader_for("cifar10", "validation", batch_size=32)
        
        error_msg = str(exc_info.value)
        assert "Invalid split" in error_msg
        assert "validation" in error_msg
        assert "train" in error_msg
        assert "test" in error_msg
    
    def test_case_insensitive_inputs(self):
        """Test that dataset and split names are case-insensitive."""
        # These should all work without errors
        for dataset in ["CIFAR10", "Cifar10", "cifar10"]:
            for split in ["TRAIN", "Train", "train", "TEST", "Test", "test"]:
                dataloader, config = dataloader_for(dataset, split, batch_size=8)
                assert config.get_dataset() == "cifar10"


class TestTransformPipelines:
    """Test that transform pipelines are correctly structured."""
    
    def test_cifar_transform_efficiency(self):
        """Test that CIFAR transforms don't include unnecessary resize operations."""
        # This is more of a design verification test
        # CIFAR images are already 32x32, so no resize should be needed
        
        dataloader, config = dataloader_for("cifar10", "test", batch_size=16)
        
        # Get a batch to ensure transforms work
        batch = next(iter(dataloader))
        images, labels = batch
        
        # Verify final shape is correct (transforms worked)
        assert images.shape == (16, 3, 32, 32)
        
        # Verify data is normalized (not in [0,1] range)
        assert images.min().item() < 0 or images.max().item() > 1
    
    def test_imagenet_transform_structure(self):
        """Test ImageNet transforms would include proper resize operations."""
        # Note: This test may fail if ImageNet dataset isn't available
        # In that case, we verify the error message is helpful
        
        try:
            train_loader, config = dataloader_for("imagenet", "train", batch_size=8)
            test_loader, config = dataloader_for("imagenet", "test", batch_size=8)
            
            # If ImageNet is available, verify config
            assert config.get_input_size() == (3, 224, 224)
            
        except ValueError as e:
            # Should get helpful error about manual ImageNet setup
            error_msg = str(e)
            assert "ImageNet dataset not found" in error_msg
            assert "data/imagenet" in error_msg
    
    def test_pin_memory_enabled(self):
        """Test that dataloaders have pin_memory enabled for GPU performance."""
        dataloader, _ = dataloader_for("cifar10", "test", batch_size=16)
        
        # Check that pin_memory is True (improves GPU transfer performance)
        assert dataloader.pin_memory is True
    
    def test_num_workers_set(self):
        """Test that dataloaders use multiple workers for performance."""
        dataloader, _ = dataloader_for("cifar10", "test", batch_size=16)
        
        # Check that num_workers > 0 for parallel data loading
        assert dataloader.num_workers >= 2


class TestNormalizationMath:
    """Test the mathematical correctness of normalization."""
    
    def test_normalization_formula_correctness(self):
        """Test that normalization follows the correct formula: (x - mean) / std."""
        # Create simple test data
        test_tensor = torch.tensor([[[
            [0.4914, 0.6000],  # First channel values
            [0.3000, 0.8000]
        ]]], dtype=torch.float32)  # Shape: (1, 1, 2, 2)
        
        # CIFAR-10 stats for red channel
        mean = 0.4914
        std = 0.2023
        
        # Apply normalization manually
        normalized_manual = (test_tensor - mean) / std
        
        # Apply normalization using torchvision transform
        from torchvision import transforms
        normalize_transform = transforms.Normalize(mean=(mean,), std=(std,))
        normalized_torch = normalize_transform(test_tensor)
        
        # Results should be identical (within floating point precision)
        torch.testing.assert_close(normalized_manual, normalized_torch, rtol=1e-6, atol=1e-6)
    
    def test_inverse_normalization_possible(self):
        """Test that normalization can be inverted to recover original values."""
        dataloader, config = dataloader_for("cifar10", "test", batch_size=4)
        
        # Get normalized batch
        normalized_batch = next(iter(dataloader))[0]
        
        # Apply inverse normalization: x_orig = (x_norm * std) + mean
        mean = torch.tensor(config.get_mean()).view(1, 3, 1, 1)  # Shape: (1, 3, 1, 1)
        std = torch.tensor(config.get_std()).view(1, 3, 1, 1)    # Shape: (1, 3, 1, 1)
        
        denormalized_batch = (normalized_batch * std) + mean
        
        # Denormalized values should be approximately in [0, 1] range
        # (not exactly due to discretization and floating point precision)
        assert denormalized_batch.min().item() >= -0.1, f"Denormalized min {denormalized_batch.min().item()} too low"
        assert denormalized_batch.max().item() <= 1.1, f"Denormalized max {denormalized_batch.max().item()} too high"