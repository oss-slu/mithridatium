# mithridatium/utils.py
"""
Utility functions for data loading, preprocessing, and model configuration.
"""
from pathlib import Path
import torch
from torchvision import datasets, transforms
from dataclasses import dataclass, field
from typing import Tuple, List
import json

class PreprocessConfig:
    """Configuration for input preprocessing."""

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (3, 32, 32),   # (C, H, W)
        channels_first: bool = True,              # True = NCHW, False = NHWC
        value_range: Tuple[float, float] = (0.0, 1.0),
        mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),  # (R, G, B)
        std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010),   # (R, G, B)
        normalize: bool = True,
        ops: List[str] = None,                     # e.g., ["resize:32"]
        dataset: str = "Unlisted"
    ):
        self.input_size = input_size
        self.channels_first = channels_first
        self.value_range = value_range
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.ops = ops if ops is not None else []
        self.dataset = dataset

    # ======== Getters ========
    def get_input_size(self):
        return self.input_size

    def get_channels_first(self):
        return self.channels_first

    def get_value_range(self):
        return self.value_range

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def get_normalize(self):
        return self.normalize

    def get_ops(self):
        return self.ops
    
    def get_dataset(self):
        return self.dataset

    # ======== Setters ========
    def set_input_size(self, input_size: Tuple[int, int]):
        self.input_size = input_size

    def set_channels_first(self, channels_first: bool):
        self.channels_first = channels_first

    def set_value_range(self, value_range: Tuple[float, float]):
        self.value_range = value_range

    def set_mean(self, mean: Tuple[float, float, float]):
        self.mean = mean

    def set_std(self, std: Tuple[float, float, float]):
        self.std = std

    def set_normalize(self, normalize: bool):
        self.normalize = normalize

    def set_ops(self, ops: List[str]):
        self.ops = ops

    def set_dataset(self, dataset):
        self.dataset = dataset


# Dataset configuration mapping
DATASET_CONFIGS = {
    "cifar10": {
        "input_size": (3, 32, 32),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "normalize": True,
    },
    "cifar100": {
        "input_size": (3, 32, 32),
        "mean": (0.5071, 0.4867, 0.4408),  # CIFAR-100 canonical stats
        "std": (0.2675, 0.2565, 0.2761),
        "normalize": True,
    },
    "imagenet": {
        "input_size": (3, 224, 224),
        "mean": (0.485, 0.456, 0.406),     # ImageNet canonical stats
        "std": (0.229, 0.224, 0.225),
        "normalize": True,
    },
}


def get_preprocess_config(dataset: str) -> PreprocessConfig:
    """
    Get preprocessing config for a dataset based on canonical transforms.
    
    Args:
        dataset: Dataset name. Supported: "cifar10", "cifar100", "imagenet".
        
    Returns:
        PreprocessConfig with canonical values for the dataset.
        
    Raises:
        ValueError: If dataset is not supported.
    """
    dataset_lower = dataset.lower().strip()
    
    if dataset_lower not in DATASET_CONFIGS:
        supported = ", ".join(sorted(DATASET_CONFIGS.keys()))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {supported}")
    
    config = DATASET_CONFIGS[dataset_lower]
    
    return PreprocessConfig(
        input_size=config["input_size"],
        channels_first=True,
        value_range=(0.0, 1.0),
        mean=config["mean"],
        std=config["std"],
        normalize=config["normalize"],
        ops=[],
        dataset=dataset_lower
    )

def load_preprocess_config(model_path: str) -> PreprocessConfig:
    """
    DEPRECATED: Load preprocessing config from model's JSON sidecar file.
    
    This function is deprecated. Use get_preprocess_config(dataset) instead,
    which provides canonical preprocessing configs based on dataset name.
    
    Args:
        model_path: Path to the model checkpoint file.
        
    Returns:
        PreprocessConfig with loaded or default values.
    """
    import warnings
    warnings.warn(
        "load_preprocess_config() is deprecated. Use get_preprocess_config(dataset) "
        "with canonical dataset configs instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    card_path = Path(model_path).with_suffix(".json")
    if not card_path.exists():
        print(f"[warn] No model sidecar found at {card_path}, using CIFAR-10 defaults")
        return PreprocessConfig()
    
    data = json.loads(card_path.read_text())
    pp = data.get("preprocess", {})
    return PreprocessConfig(
        input_size=tuple(pp.get("input_size", (32, 32))),
        channels_first=pp.get("channels_first", True),
        value_range=tuple(pp.get("value_range", (0.0, 1.0))),
        mean=tuple(pp["mean"]),
        std=tuple(pp["std"]),
        normalize=pp.get("normalize", True),
        ops=list(pp.get("ops", [])),
    )

def dataloader_for(dataset: str, split: str, batch_size: int = 256):
    """
    Create a dataloader for the specified dataset using canonical transforms.
    
    Args:
        dataset: Dataset name. Supported: "cifar10", "cifar100", "imagenet".
        split: "train" or "test".
        batch_size: Batch size for the dataloader.
        
    Returns:
        tuple: (torch.utils.data.DataLoader, PreprocessConfig) for the specified dataset.
        
    Raises:
        ValueError: If dataset is not supported or split is invalid.
    """
    # Validate inputs
    dataset_lower = dataset.lower().strip()
    split_lower = split.lower().strip()
    
    if dataset_lower not in DATASET_CONFIGS:
        supported = ", ".join(sorted(DATASET_CONFIGS.keys()))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {supported}")
    
    if split_lower not in ("train", "test"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'")
    
    # Get canonical preprocessing config for the dataset
    config = get_preprocess_config(dataset_lower)
    
    # Build dataset-specific transform pipeline
    # Standard order: Resize/Crop → ToTensor() → Normalize()
    if dataset_lower == "cifar10":
        # CIFAR-10: 32x32 RGB images (already correct size)
        transform_list = [
            # No resize needed - images are already 32x32
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ]
        ds = datasets.CIFAR10(
            root="data",
            train=(split_lower == "train"),
            download=True,
            transform=transforms.Compose(transform_list)
        )
    
    elif dataset_lower == "cifar100":
        # CIFAR-100: 32x32 RGB images (already correct size)
        transform_list = [
            # No resize needed - images are already 32x32
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ]
        ds = datasets.CIFAR100(
            root="data",
            train=(split_lower == "train"),
            download=True,
            transform=transforms.Compose(transform_list)
        )
    
    elif dataset_lower == "imagenet":
        # ImageNet: Standard ImageNet preprocessing pipeline
        if split_lower == "train":
            transform_list = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std)
            ]
        else:  # test/val
            transform_list = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std)
            ]
        
        # ImageNet requires manual dataset setup - provide clear instructions
        try:
            from torchvision.datasets import ImageNet
            ds = ImageNet(
                root="data/imagenet",
                split="train" if split_lower == "train" else "val",
                transform=transforms.Compose(transform_list)
            )
        except RuntimeError as e:
            raise ValueError(
                f"ImageNet dataset not found. Please download ImageNet manually and place it in "
                f"'data/imagenet/' directory. Original error: {e}"
            )
    
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split_lower == "train"),
        num_workers=2,
        pin_memory=True  # Improve GPU transfer performance
    )
    
    return dataloader, config