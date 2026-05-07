# mithridatium/utils.py
"""
Utility functions for data loading, preprocessing, and model configuration.
"""
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from dataclasses import dataclass, field
from typing import Tuple, List
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_ARCHIVE_NAME = "imagenette2-160.tgz"
IMAGENETTE_EXTRACTED_DIR = DATA_ROOT / "imagenette2-160"

class PreprocessConfig:
    """Configuration for input preprocessing."""

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (3, 32, 32),   # (C, H, W)
        channels_first: bool = True,              # True = NCHW, False = NHWC
        value_range: Tuple[float, float] = (0.0, 1.0),
        mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),  # (R, G, B)
        std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010),   # (R, G, B)
        num_classes: int = 10,
        normalize: bool = True,
        ops: List[str] = None,                     # e.g., ["resize:32"]
        dataset: str = "Unlisted"
    ):
        self.input_size = input_size
        self.channels_first = channels_first
        self.value_range = value_range
        self.mean = mean
        self.std = std
        self.num_classes = num_classes
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

    def get_num_classes(self):
        return self.num_classes

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

    def set_num_classes(self, num_classes: int):
        self.num_classes = num_classes

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
        "num_classes": 10,
        "normalize": True,
    },
    "cifar100": {
        "input_size": (3, 32, 32),
        "mean": (0.5071, 0.4867, 0.4408),  # CIFAR-100 canonical stats
        "std": (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
        "normalize": True,
    },
    "imagenet": {
        "input_size": (3, 224, 224),
        "mean": (0.485, 0.456, 0.406),     # ImageNet canonical stats
        "std": (0.229, 0.224, 0.225),
        "num_classes": 1000,
        "normalize": True,
    },
    "imagenet_subset": {
        "input_size": (3, 224, 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_classes": 10,
        "normalize": True,
    },
    "cifar10_for_imagenet": {
        "input_size": (3, 224, 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_classes": 10,
        "normalize": True,
    },
    "fake_imagenet": {
        "input_size": (3, 224, 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_classes": 1000,
        "normalize": True,
    },

}


def _ensure_imagenette_subset_available() -> Path:
    """
    Ensure Imagenette (10-class ImageNet subset) is available locally.

    Returns:
        Path to extracted Imagenette root containing 'train' and 'val'.
    """
    train_dir = IMAGENETTE_EXTRACTED_DIR / "train"
    val_dir = IMAGENETTE_EXTRACTED_DIR / "val"
    if train_dir.exists() and val_dir.exists():
        return IMAGENETTE_EXTRACTED_DIR

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        download_and_extract_archive(
            url=IMAGENETTE_URL,
            download_root=str(DATA_ROOT),
            extract_root=str(DATA_ROOT),
            filename=IMAGENETTE_ARCHIVE_NAME,
            remove_finished=False,
        )
    except Exception as ex:
        raise ValueError(
            "Failed to download ImageNet subset (Imagenette). "
            f"Tried URL: {IMAGENETTE_URL}. Reason: {ex}"
        )

    if not (train_dir.exists() and val_dir.exists()):
        raise ValueError(
            "ImageNet subset download/extract completed but dataset folders were not found. "
            f"Expected '{train_dir}' and '{val_dir}'."
        )

    return IMAGENETTE_EXTRACTED_DIR


def get_preprocess_config(dataset: str) -> PreprocessConfig:
    """
    Get preprocessing config for a dataset based on canonical transforms.
    
    Args:
        dataset: Dataset name. Supported: "cifar10", "cifar100", "imagenet", "imagenet_subset".
        
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
        num_classes=config["num_classes"],
        normalize=config["normalize"],
        ops=[],
        dataset=dataset_lower
    )


def _build_transform_from_config(config: PreprocessConfig, *, train: bool = False):
    """
    Build torchvision transforms from a PreprocessConfig.

    The config is the source of truth for input size, normalization, and
    preprocessing behavior.
    """
    _, h, w = config.get_input_size()

    transform_list = []

    # CIFAR-style 32x32 datasets do not need resizing.
    # ImageNet-style configs do.
    if h != 32 or w != 32:
        if train:
            transform_list.append(transforms.Resize(max(h, w)))
            transform_list.append(transforms.CenterCrop((h, w)))
        else:
            transform_list.append(transforms.Resize(max(h, w)))
            transform_list.append(transforms.CenterCrop((h, w)))

    transform_list.append(transforms.ToTensor())

    if config.get_normalize():
        transform_list.append(
            transforms.Normalize(config.get_mean(), config.get_std())
        )

    return transforms.Compose(transform_list)

def _build_transform_from_config(config: PreprocessConfig, *, train: bool = False):
    """
    Build torchvision transforms from a PreprocessConfig.

    The config is the source of truth for input size, normalization, and
    preprocessing behavior.
    """
    _, h, w = config.get_input_size()

    transform_list = []

    # CIFAR-style 32x32 datasets do not need resizing.
    # ImageNet-style configs do.
    if h != 32 or w != 32:
        if train:
            transform_list.append(transforms.Resize(max(h, w)))
            transform_list.append(transforms.CenterCrop((h, w)))
        else:
            transform_list.append(transforms.Resize(max(h, w)))
            transform_list.append(transforms.CenterCrop((h, w)))

    transform_list.append(transforms.ToTensor())

    if config.get_normalize():
        transform_list.append(
            transforms.Normalize(config.get_mean(), config.get_std())
        )

    return transforms.Compose(transform_list)

def dataloader_for(dataset: str, split: str, batch_size: int = 256):
    """
    Create a dataloader for the specified dataset using canonical transforms.

    Args:
        dataset: Dataset/preprocessing mode name.
        split: "train" or "test".
        batch_size: Batch size for the dataloader.

    Returns:
        tuple: (torch.utils.data.DataLoader, PreprocessConfig)

    Raises:
        ValueError: If dataset is not supported or split is invalid.
    """
    dataset_lower = dataset.lower().strip()
    split_lower = split.lower().strip()

    if dataset_lower not in DATASET_CONFIGS:
        supported = ", ".join(sorted(DATASET_CONFIGS.keys()))
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported datasets: {supported}"
        )

    if split_lower not in ("train", "test"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'")

    config = get_preprocess_config(dataset_lower)
    transform = _build_transform_from_config(
        config,
        train=(split_lower == "train"),
    )

    if dataset_lower == "cifar10":
        ds = datasets.CIFAR10(
            root=str(DATA_ROOT),
            train=(split_lower == "train"),
            download=True,
            transform=transform,
        )

    elif dataset_lower == "cifar100":
        ds = datasets.CIFAR100(
            root=str(DATA_ROOT),
            train=(split_lower == "train"),
            download=True,
            transform=transform,
        )

    elif dataset_lower == "imagenet":
        try:
            from torchvision.datasets import ImageNet

            ds = ImageNet(
                root=str(DATA_ROOT),
                split="train" if split_lower == "train" else "val",
                transform=transform,
            )
        except RuntimeError as e:
            raise ValueError(
                f"ImageNet dataset not found. Please download ImageNet manually and place the "
                f"required ILSVRC archives under '{DATA_ROOT}'. Original error: {e}"
            )

    elif dataset_lower == "imagenet_subset":
        imagenette_root = _ensure_imagenette_subset_available()
        split_dir = imagenette_root / ("train" if split_lower == "train" else "val")
        transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
        ds = datasets.ImageFolder(
            root=str(split_dir),
            transform=transforms.Compose(transform_list),
        )

    elif dataset_lower == "cifar10_for_imagenet":
        ds = datasets.CIFAR10(
            root=str(DATA_ROOT),
            train=(split_lower == "train"),
            download=True,
            transform=transform,
        )

    elif dataset_lower == "fake_imagenet":
        _, h, w = config.get_input_size()
        ds = datasets.FakeData(
            size=512,
            image_size=(3, h, w),
            num_classes=config.get_num_classes(),
            transform=transform,
        )

    else:
        # Defensive fallback. This should be unreachable because DATASET_CONFIGS
        # was checked above.
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split_lower == "train"),
        num_workers=2,
        pin_memory=True,
    )

    return dataloader, config

def dataloader_for_config(
    dataset: str,
    split: str,
    config: PreprocessConfig,
    batch_size: int = 256,
):
    """
    Create a dataloader using an explicit preprocessing config instead of only
    dataset canonical defaults.

    This is useful for Hugging Face models whose expected preprocessing may
    differ from the dataset default.
    """
    dataset_lower = dataset.lower().strip()
    split_lower = split.lower().strip()

    if split_lower not in ("train", "test"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'")

    transform = _build_transform_from_config(
        config,
        train=(split_lower == "train"),
    )

    if dataset_lower == "cifar10":
        ds = datasets.CIFAR10(
            root=str(DATA_ROOT),
            train=(split_lower == "train"),
            download=True,
            transform=transform,
        )

    elif dataset_lower == "cifar100":
        ds = datasets.CIFAR100(
            root=str(DATA_ROOT),
            train=(split_lower == "train"),
            download=True,
            transform=transform,
        )

    elif dataset_lower == "cifar10_for_imagenet":
        ds = datasets.CIFAR10(
            root=str(DATA_ROOT),
            train=(split_lower == "train"),
            download=True,
            transform=transform,
        )

    elif dataset_lower == "fake_imagenet":
        _, h, w = config.get_input_size()
        ds = datasets.FakeData(
            size=512,
            image_size=(3, h, w),
            num_classes=config.get_num_classes(),
            transform=transform,
        )

    elif dataset_lower == "imagenet":
        try:
            from torchvision.datasets import ImageNet
            ds = ImageNet(
                root=str(DATA_ROOT),
                split="train" if split_lower == "train" else "val",
                transform=transform,
            )
        except RuntimeError as e:
            raise ValueError(
                f"ImageNet dataset not found. Please download ImageNet manually and place the "
                f"required ILSVRC archives under '{DATA_ROOT}'."
                f" Original error: {e}"
            )
    elif dataset_lower == "imagenet_subset":
        imagenette_root = _ensure_imagenette_subset_available()
        split_dir = imagenette_root / ("train" if split_lower == "train" else "val")
        ds = datasets.ImageFolder(
            root=str(split_dir),
            transform=transform,
        )
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}' for dataloader_for_config()."
        )

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split_lower == "train"),
        num_workers=2,
        pin_memory=True,
    )

    return dataloader, config