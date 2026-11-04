"""Reusable triggered-image datasets for repair evaluation.

``TriggeredDataset`` can apply any callable trigger to a dataset of CHW image
tensors and *original* clean labels. The built-in white-square, checkerboard,
cross, and blend triggers support testing known attack specifications. The
white-square default matches ``scripts/train_resnet18.py`` if the original
poisoning configuration used that same trigger and preprocessing.

This module does not depend on ANP or a specific model architecture. Other
repair methods can use it without importing ANP internals.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

POSITIONS = ("bottom-right", "bottom-left", "top-right", "top-left")
TRIGGER_TYPES = ("white-square", "checkerboard", "cross", "blend")


class TriggeredDataset(Dataset):
    """Apply a supplied image trigger to reproducibly sampled non-target data.

    The incoming dataset yields (image_tensor, original_label, ...). Labels are
    retained, and the trigger receives a cloned image, never the original.
    """

    def __init__(self, dataset: Dataset, *, trigger: Callable[[Tensor], Tensor],
                 attack_target: int, samples: int = 500, seed: int = 0):
        if isinstance(attack_target, bool) or not isinstance(attack_target, int) or attack_target < 0:
            raise ValueError("Attack target must be a nonnegative class index")
        if not callable(trigger):
            raise ValueError("Trigger must be callable")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("ASR samples must be a positive integer")
        if len(dataset) == 0:
            raise ValueError("ASR requires a nonempty evaluation dataset")

        image, _ = dataset[0][:2]
        if not isinstance(image, Tensor) or image.ndim != 3:
            raise ValueError("Triggered evaluation expects CHW image tensors")

        self.dataset = dataset
        self.trigger = trigger
        self.attack_target = attack_target
        generator = torch.Generator().manual_seed(seed)
        self.indices: list[int] = []
        for index in torch.randperm(len(dataset), generator=generator).tolist():
            _, original_label = dataset[index][:2]
            if int(original_label) != attack_target:
                self.indices.append(index)
                if len(self.indices) >= samples:
                    break
        if not self.indices:
            raise ValueError("ASR requires examples outside the attack target class")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, original_label = self.dataset[self.indices[index]][:2]
        triggered_image = self.trigger(image.clone())
        return triggered_image, original_label


class WhiteSquareTrigger:
    """Overlay a white square on CHW RGB tensors using preprocessing metadata.

    ``config`` supplies the mean, standard deviation, and normalization flag
    used to create the model inputs. White means pixel-space RGB (1, 1, 1),
    transformed to model-input space if normalization was enabled.
    """

    def __init__(self, *, config, trigger_size: int = 4,
                 trigger_pos: str = "bottom-right"):
        if isinstance(trigger_size, bool) or not isinstance(trigger_size, int) or trigger_size < 1:
            raise ValueError("Trigger size must be a positive integer")
        if trigger_pos not in POSITIONS:
            raise ValueError(f"Trigger position must be one of: {', '.join(POSITIONS)}")
        if config.get_normalize():
            mean, std = config.get_mean(), config.get_std()
            if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
                raise ValueError("White-square trigger requires valid RGB normalization statistics")
            values = [(1.0 - m) / s for m, s in zip(mean, std)]
        else:
            values = [1.0, 1.0, 1.0]
        self.white_pixel = torch.tensor(values).view(3, 1, 1)
        self.trigger_size = trigger_size
        self.trigger_pos = trigger_pos

    def __call__(self, image: Tensor) -> Tensor:
        if not isinstance(image, Tensor) or image.ndim != 3:
            raise ValueError("White-square trigger expects a CHW tensor")
        channels, height, width = image.shape
        if channels != 3 or self.trigger_size > min(height, width):
            raise ValueError("White-square trigger requires RGB images and a size no larger than the image")
        size = self.trigger_size
        y = slice(-size, None) if self.trigger_pos.startswith("bottom") else slice(0, size)
        x = slice(-size, None) if self.trigger_pos.endswith("right") else slice(0, size)
        triggered = image.clone()
        triggered[:, y, x] = self.white_pixel.to(device=image.device, dtype=image.dtype)
        return triggered


class WhiteSquareTriggeredDataset(TriggeredDataset):
    """Convenience wrapper for reproducible white-square ASR measurements.

    The dataset may come from any compatible RGB image task, but CLI support
    in Issue #169 remains limited to the known CIFAR-10 BadNet configuration.
    """

    def __init__(self, dataset: Dataset, *, attack_target: int, config,
                 trigger_size: int = 4, trigger_pos: str = "bottom-right",
                 samples: int = 500, seed: int = 0):
        num_classes = config.get_num_classes()
        if isinstance(attack_target, bool) or not isinstance(attack_target, int) or not 0 <= attack_target < num_classes:
            raise ValueError(f"Attack target must be a class in [0, {num_classes - 1}]")
        trigger = WhiteSquareTrigger(config=config, trigger_size=trigger_size,
                                    trigger_pos=trigger_pos)
        super().__init__(dataset, trigger=trigger, attack_target=attack_target,
                         samples=samples, seed=seed)
        # Validate the image size and channel count eagerly (before model work).
        trigger(dataset[0][0])
        self.trigger_size = trigger_size
        self.trigger_pos = trigger_pos
        self.white_pixel = trigger.white_pixel


class PatternPatchTrigger(WhiteSquareTrigger):
    """Apply a checkerboard or cross patch in the specified corner.

    The checkerboard alternates white and black pixels, beginning with white
    at the patch's top-left. The cross replaces only the central row/column.
    A true ASR requires the exact pattern used to train the poisoned model.
    """

    def __init__(self, *, config, trigger_type: str, trigger_size: int = 4,
                 trigger_pos: str = "bottom-right"):
        if trigger_type not in ("checkerboard", "cross"):
            raise ValueError("Pattern trigger type must be checkerboard or cross")
        super().__init__(config=config, trigger_size=trigger_size, trigger_pos=trigger_pos)
        self.trigger_type = trigger_type
        if config.get_normalize():
            black = [(0.0 - m) / s for m, s in zip(config.get_mean(), config.get_std())]
        else:
            black = [0.0, 0.0, 0.0]
        self.black_pixel = torch.tensor(black).view(3, 1, 1)

    def __call__(self, image: Tensor) -> Tensor:
        if not isinstance(image, Tensor) or image.ndim != 3:
            raise ValueError("Pattern trigger expects a CHW tensor")
        if image.shape[0] != 3 or self.trigger_size > min(image.shape[1:]):
            raise ValueError("Pattern trigger requires RGB images and a size no larger than the image")
        size = self.trigger_size
        y_start = image.shape[1] - size if self.trigger_pos.startswith("bottom") else 0
        x_start = image.shape[2] - size if self.trigger_pos.endswith("right") else 0
        triggered = image.clone()
        white = self.white_pixel.to(device=image.device, dtype=image.dtype)[:, 0, 0]
        black = self.black_pixel.to(device=image.device, dtype=image.dtype)[:, 0, 0]
        for y in range(size):
            for x in range(size):
                if self.trigger_type == "checkerboard":
                    triggered[:, y_start+y, x_start+x] = white if (x + y) % 2 == 0 else black
                elif y == size // 2 or x == size // 2:
                    triggered[:, y_start+y, x_start+x] = white
        return triggered


class BlendTrigger:
    """Blend a uniform white overlay across the complete RGB image.

    In pixel space, x' = (1-alpha)*x + alpha*white. Since standard image
    normalization is affine, this is also correct in normalized tensor space.
    This is NOT a general image-based blended trigger; use a custom callable
    for a specific overlay pattern or image when needed.
    """

    def __init__(self, *, config, alpha: float = 0.2):
        import math
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Blend alpha must be finite and in (0, 1]")
        if config.get_normalize():
            mean, std = config.get_mean(), config.get_std()
            if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
                raise ValueError("Blend trigger requires valid RGB normalization statistics")
            values = [(1.0 - m) / s for m, s in zip(mean, std)]
        else:
            values = [1.0, 1.0, 1.0]
        self.white_pixel = torch.tensor(values).view(3, 1, 1)
        self.alpha = alpha

    def __call__(self, image: Tensor) -> Tensor:
        if not isinstance(image, Tensor) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Blend trigger expects a CHW RGB image")
        white = self.white_pixel.to(device=image.device, dtype=image.dtype)
        return (1.0 - self.alpha) * image + self.alpha * white


def build_triggered_dataset(dataset: Dataset, *, config, attack_target: int,
                            trigger_type: str = "white-square", trigger_size: int = 4,
                            trigger_pos: str = "bottom-right", blend_alpha: float = 0.2,
                            samples: int = 500, seed: int = 0) -> TriggeredDataset:
    """Construct ASR data for a *known* supported trigger specification.

    This factory is independent of ANP, so other repair methods may reuse it.
    It does not infer the trigger from the model's parameters.
    """
    if trigger_type not in TRIGGER_TYPES:
        raise ValueError(f"Trigger type must be one of: {', '.join(TRIGGER_TYPES)}")
    num_classes = config.get_num_classes()
    if isinstance(attack_target, bool) or not isinstance(attack_target, int) or not 0 <= attack_target < num_classes:
        raise ValueError(f"Attack target must be a class in [0, {num_classes - 1}]")
    if trigger_type == "white-square":
        trigger = WhiteSquareTrigger(config=config, trigger_size=trigger_size,
                                     trigger_pos=trigger_pos)
    elif trigger_type in ("checkerboard", "cross"):
        trigger = PatternPatchTrigger(config=config, trigger_type=trigger_type,
                                      trigger_size=trigger_size, trigger_pos=trigger_pos)
    else:
        trigger = BlendTrigger(config=config, alpha=blend_alpha)
    result = TriggeredDataset(dataset, trigger=trigger, attack_target=attack_target,
                              samples=samples, seed=seed)
        # Fail early on malformed image shape, not partway through model evaluation.
    _ = result[0]
    return result
