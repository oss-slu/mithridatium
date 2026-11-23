"""Blended injection backdoor (Chen et al., 2017).

"Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning",
Chen, Liu, Li, Lu, Song (arXiv:1712.05526). A poisoned input is
``x' = alpha * k + (1 - alpha) * x``, where ``k`` is a fixed key pattern blended
over the whole image, and its label is set to the target class. The paper uses
either a cartoon image or a random-noise pattern whose pixels are drawn
uniformly from [0, 255]; this module uses the random-noise key so no image
asset is needed. There is no official code; this follows the paper.

The trigger itself is :class:`mithridatium.evaluation.triggered.BlendTrigger`,
so poisoning and ASR measurement blend the exact same way.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from mithridatium.evaluation.triggered import BlendTrigger


def make_blend_pattern(shape=(3, 32, 32), seed: Optional[int] = None) -> torch.Tensor:
    """Random-noise key pattern: each pixel uniform in [0, 1] (the paper's [0, 255])."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    return torch.rand(shape, generator=gen)


class BlendBackdoorDataset(Dataset):
    """Dataset wrapper that poisons examples by blending in a key pattern.

    Mirrors :class:`~mithridatium.attacks.invisible.InvisibleBackdoorDataset`:
    ``mode='train'`` poisons a fixed ``poison_rate`` share of the training set
    (non-target samples) and relabels them to ``target_class``;
    ``mode='test_poison'`` yields ``(x, original_label, target_label)`` for every
    non-target sample, for ASR. The blend is applied in pixel space between
    ``pre_transform`` (e.g. ``ToTensor``) and ``post_transform`` (e.g.
    augmentation and normalization).
    """

    def __init__(
        self,
        dataset,
        poison_rate: float,
        target_class: int,
        pattern: torch.Tensor,
        alpha: float = 0.2,
        mode: str = "train",
        pre_transform=None,
        post_transform=None,
        seed: Optional[int] = None,
    ):
        if mode not in {"train", "test_poison"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'train' or 'test_poison'.")

        self.dataset = dataset
        self.poison_rate = float(poison_rate)
        self.target_class = int(target_class)
        self.pattern = pattern.clone()
        self.alpha = float(alpha)
        self.trigger = BlendTrigger(config=None, alpha=self.alpha, pattern=self.pattern)
        self.mode = mode
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.seed = int(seed) if seed is not None else None

        self.poisoned_indices: set[int] = set()
        self._non_target_indices: list[int] = [i for i in range(len(dataset)) if dataset[i][1] != self.target_class]
        if self.mode == "train":
            num_samples = len(dataset)
            num_poison = int(self.poison_rate * num_samples)
            rng = random.Random(self.seed)
            self.poisoned_indices = set(rng.sample(self._non_target_indices, min(num_poison, len(self._non_target_indices))))
            print(f"[blend] poisoning {len(self.poisoned_indices)}/{num_samples} training samples (alpha={self.alpha})")
        else:
            print(f"[blend] ASR subset size {len(self._non_target_indices)} (target={self.target_class})")

    def __len__(self):
        if self.mode == "test_poison":
            return len(self._non_target_indices)
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.mode == "test_poison":
            img, label = self.dataset[self._non_target_indices[index]]
        else:
            img, label = self.dataset[index]

        if self.pre_transform is not None:
            img = self.pre_transform(img)
        elif not isinstance(img, torch.Tensor):
            from torchvision import transforms

            img = transforms.ToTensor()(img)

        if self.mode == "train":
            if index in self.poisoned_indices:
                img = self.trigger(img)
                label = self.target_class
        else:
            original_label = int(label)
            img = self.trigger(img)
            if self.post_transform is not None:
                img = self.post_transform(img)
            return img, original_label, self.target_class

        if self.post_transform is not None:
            img = self.post_transform(img)

        return img, int(label)
