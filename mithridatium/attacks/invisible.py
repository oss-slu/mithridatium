from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, Sequence, Iterable

import torch
from torch.utils.data import Dataset


def apply_invisible_trigger(x: torch.Tensor, uap: torch.Tensor) -> torch.Tensor:
    """Add a universal perturbation ``uap`` to ``x`` and clamp to valid range.

    This function assumes ``x`` and ``uap`` are normalized to [0,1] (i.e.
    they are raw images or tensors after ``transforms.ToTensor``).

    The perturbation is element-wise added and the result is clipped back to
    [0,1] so that it remains a legitimate image.  ``uap`` can be broadcast to
    ``x`` (e.g. shape ``(3,32,32)``).
    """
    if not isinstance(x, torch.Tensor) or not isinstance(uap, torch.Tensor):
        raise ValueError("Both x and uap must be torch.Tensors")

    # broadcast if necessary
    return torch.clamp(x + uap, 0.0, 1.0)


def create_random_uap(shape, xi: float = 0.05, p: str = "inf", seed: Optional[int] = None) -> torch.Tensor:
    """Create a random universal perturbation within an Lp norm ball.

    This is a very simple proxy for more elaborate UAP generation algorithms.
    The perturbation is sampled uniformly (``p='inf'``) or from a normal
    distribution scaled to have norm ``xi`` (``p='2'``).  The output is sized
    ``shape`` and suitable for adding to CIFAR-style images.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    if p == "inf":
        # uniform noise in [-xi, xi]
        return (torch.rand(shape, generator=gen) * 2 - 1) * float(xi)
    elif p == "2":
        # sample gaussian noise then scale each channel to have L2 norm xi
        eps = torch.randn(shape, generator=gen)
        # compute per-channel norm: shape (C,)
        # assume shape[0] is channel dimension; collapse all others
        channel_norm = eps.view(eps.shape[0], -1).norm(p=2, dim=1)
        # avoid division by zero
        channel_norm = channel_norm.clamp(min=1e-12)
        # reshape to (C,1,1,...) for broadcasting
        reshape = [eps.shape[0]] + [1] * (eps.dim() - 1)
        eps = eps / channel_norm.view(*reshape)
        return eps * float(xi)
    else:
        raise ValueError(f"Unsupported norm '{p}', choose 'inf' or '2'")


class InvisibleBackdoorDataset(Dataset):
    """Dataset wrapper that poisons examples with a universal perturbation.

    Behavior mirrors :class:`~scripts.train_resnet18.BadNetDataset` except that we
    ``add_trigger`` by adding the pre-computed UAP instead of stamping a square
    patch.  The dataset supports ``mode`` values ``'train'`` and ``'test_poison'``
    for poisoning during training and measuring ASR respectively.
    """

    def __init__(
        self,
        dataset,
        poison_rate: float,
        target_class: int,
        uap: torch.Tensor,
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
        self.uap = uap.clone()
        self.mode = mode
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.seed = int(seed) if seed is not None else None

        # choose which indices to poison during training (non-target samples)
        # and keep a cached list of non-target indices for mapping during test_poison.
        self.poisoned_indices: set[int] = set()
        self._non_target_indices: list[int] = [i for i in range(len(dataset)) if dataset[i][1] != self.target_class]
        if self.mode == "train":
            num_samples = len(dataset)
            num_poison = int(self.poison_rate * num_samples)
            rng = random.Random(self.seed)
            self.poisoned_indices = set(rng.sample(self._non_target_indices, min(num_poison, len(self._non_target_indices))))
            print(f"[invisible] poisoning {len(self.poisoned_indices)}/{num_samples} training samples")
        else:
            # ``test_poison`` mode just records how many candidates we will use
            print(f"[invisible] ASR subset size {len(self._non_target_indices)} (target={self.target_class})")

    def __len__(self):
        if self.mode == "test_poison":
            # only non-target class samples are counted for ASR
            return len(self._non_target_indices)
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.mode == "test_poison":
            # use cached non-target list for direct indexing
            orig_idx = self._non_target_indices[index]
            img, label = self.dataset[orig_idx]
        else:
            img, label = self.dataset[index]

        if self.pre_transform is not None:
            img = self.pre_transform(img)
        elif not isinstance(img, torch.Tensor):
            from torchvision import transforms

            img = transforms.ToTensor()(img)

        if self.mode == "train":
            if index in self.poisoned_indices:
                img = apply_invisible_trigger(img, self.uap)
                label = self.target_class
        else:
            # ASR mode: always a non-target sample
            original_label = int(label)
            target_label = int(self.target_class)
            img = apply_invisible_trigger(img, self.uap)
            if self.post_transform is not None:
                img = self.post_transform(img)
            return img, original_label, target_label

        if self.post_transform is not None:
            img = self.post_transform(img)

        return img, int(label)
