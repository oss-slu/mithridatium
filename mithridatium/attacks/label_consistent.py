"""Label-consistent backdoor (Turner, Tsipras & Madry, 2019).

"Label-Consistent Backdoor Attacks" (arXiv:1912.02771). Only target-class
training images are poisoned, and they keep their correct label: each one is
replaced with a harder-to-classify version of itself (an adversarial
perturbation or GAN interpolation, computed ahead of time) and stamped with a
3x3 black-and-white trigger. Since the poisoned images are correctly labeled,
filtering mislabeled images doesn't catch them. At test time the trigger alone
is stamped on non-target images.

Adapted from MadryLab/label-consistent-backdoor-code (``poison_attack.py``),
MIT License, Copyright (c) 2019 Alexander Turner, Dimitris Tsipras and
Aleksander Madry. The trigger masks, the index selection and the amplitude
handling are kept as in that file. The harder-to-classify images are the
authors' released ``fully_poisoned_training_datasets`` corpora; fetch one with
``scripts/fetch_label_consistent.py``.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright
notice and this permission notice shall be included in all copies or
substantial portions of the Software.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# (row, col) -> +1 white / -1 black, exactly as in the authors' poison_attack.py.
# The last entry of each 3x3 block is not a strict checkerboard there either.
TRIGGER_MASKS = {
    "bottom-right": [
        ((-1, -1), 1), ((-1, -2), -1), ((-1, -3), 1),
        ((-2, -1), -1), ((-2, -2), 1), ((-2, -3), -1),
        ((-3, -1), 1), ((-3, -2), -1), ((-3, -3), -1),
    ],
    "all-corners": [
        ((0, 0), 1), ((0, 1), -1), ((0, 2), -1),
        ((1, 0), -1), ((1, 1), 1), ((1, 2), -1),
        ((2, 0), 1), ((2, 1), -1), ((2, 2), 1),

        ((-1, 0), 1), ((-1, 1), -1), ((-1, 2), 1),
        ((-2, 0), -1), ((-2, 1), 1), ((-2, 2), -1),
        ((-3, 0), 1), ((-3, 1), -1), ((-3, 2), -1),

        ((0, -1), 1), ((0, -2), -1), ((0, -3), -1),
        ((1, -1), -1), ((1, -2), 1), ((1, -3), -1),
        ((2, -1), 1), ((2, -2), -1), ((2, -3), 1),

        ((-1, -1), 1), ((-1, -2), -1), ((-1, -3), 1),
        ((-2, -1), -1), ((-2, -2), 1), ((-2, -3), -1),
        ((-3, -1), 1), ((-3, -2), -1), ((-3, -3), -1),
    ],
}

# The authors' config.json defaults.
DEFAULT_SEED = 4557077


def apply_label_consistent_trigger(
    img: torch.Tensor, trigger: str = "bottom-right", amplitude: Optional[float] = None
) -> torch.Tensor:
    """Stamp the trigger on a pixel-space CHW image in [0, 1].

    With ``amplitude=None`` the trigger pixels are set to white or black. With an
    amplitude (0-1 scale, e.g. 32/255) they are shifted up or down by that much
    instead. Every channel gets the same value, and the result is clipped to [0, 1].
    """
    if trigger not in TRIGGER_MASKS:
        raise ValueError(f"Unknown trigger '{trigger}'. Choose from: {', '.join(TRIGGER_MASKS)}")
    if amplitude is not None and not 0 <= amplitude <= 1:
        raise ValueError("Trigger amplitude must be in [0, 1]")
    out = img.clone()
    for (row, col), value in TRIGGER_MASKS[trigger]:
        if amplitude is None:
            out[:, row, col] = float(value)
        else:
            out[:, row, col] += value * amplitude
    return torch.clamp(out, 0.0, 1.0)


def select_indices_to_poison(labels, poisoning_proportion: float, target_class: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Random ``poisoning_proportion`` of the target-class indices.

    Same algorithm as the authors' ``select_indices_to_poison`` with
    ``apply_to=target_class``, using a seeded RandomState in place of the global
    numpy seed, so the same seed and label order pick the same images.
    """
    if not 0 <= poisoning_proportion <= 1:
        raise ValueError("Poisoning proportion must be in [0, 1]")
    labels = np.asarray(labels)
    num_target = int(np.sum(np.isin(labels, [target_class])))
    num_to_poison = round(num_target * poisoning_proportion)
    indices = np.random.RandomState(seed).permutation(len(labels))
    indices = indices[np.isin(labels[indices], [target_class])]
    return indices[:num_to_poison]


class LabelConsistentBackdoorDataset(Dataset):
    """Dataset wrapper for label-consistent training and ASR evaluation.

    ``mode='train'``: the selected target-class images are replaced by
    ``harder_images[index]`` (an (N, 32, 32, 3) array of 0-255 pixel values in
    the same order as ``dataset``) and stamped with the trigger. Labels never
    change. ``mode='test_poison'`` stamps the trigger on every non-target image
    and yields ``(x, original_label, target_label)``.

    ``pre_transform`` should only convert to a tensor (it is skipped for the
    replaced images, which are already arrays); put augmentation in
    ``post_transform``, since the authors poison the dataset first and augment
    while training.
    """

    def __init__(
        self,
        dataset,
        target_class: int,
        harder_images=None,
        poisoning_proportion: float = 1.0,
        trigger: str = "bottom-right",
        amplitude: Optional[float] = None,
        mode: str = "train",
        pre_transform=None,
        post_transform=None,
        seed: int = DEFAULT_SEED,
    ):
        if mode not in {"train", "test_poison"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'train' or 'test_poison'.")
        if trigger not in TRIGGER_MASKS:
            raise ValueError(f"Unknown trigger '{trigger}'. Choose from: {', '.join(TRIGGER_MASKS)}")

        self.dataset = dataset
        self.target_class = int(target_class)
        self.poisoning_proportion = float(poisoning_proportion)
        self.trigger = trigger
        self.amplitude = amplitude
        self.mode = mode
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.seed = int(seed)

        labels = getattr(dataset, "targets", None)
        if labels is None:
            labels = [dataset[i][1] for i in range(len(dataset))]
        labels = np.asarray(labels)

        self.poisoned_indices: set[int] = set()
        self._non_target_indices: list[int] = []
        if self.mode == "train":
            if harder_images is None:
                raise ValueError("Training needs harder_images (see scripts/fetch_label_consistent.py)")
            if len(harder_images) != len(dataset) or tuple(harder_images.shape[1:]) != (32, 32, 3):
                raise ValueError("harder_images must be (len(dataset), 32, 32, 3), in the dataset's order")
            self.harder_images = harder_images
            self.poisoned_indices = set(
                select_indices_to_poison(labels, self.poisoning_proportion, self.target_class, self.seed).tolist()
            )
            print(
                f"[label-consistent] poisoning {len(self.poisoned_indices)}/{len(dataset)} training samples "
                f"(target={self.target_class}, trigger={self.trigger}, amplitude={self.amplitude})"
            )
        else:
            self.harder_images = None
            self._non_target_indices = [i for i, y in enumerate(labels) if int(y) != self.target_class]
            print(f"[label-consistent] ASR subset size {len(self._non_target_indices)} (target={self.target_class})")

    def __len__(self):
        if self.mode == "test_poison":
            return len(self._non_target_indices)
        return len(self.dataset)

    def _to_tensor(self, img):
        if self.pre_transform is not None:
            return self.pre_transform(img)
        if not isinstance(img, torch.Tensor):
            from torchvision import transforms

            return transforms.ToTensor()(img)
        return img

    def __getitem__(self, index: int):
        if self.mode == "test_poison":
            img, label = self.dataset[self._non_target_indices[index]]
            img = apply_label_consistent_trigger(self._to_tensor(img), self.trigger, self.amplitude)
            if self.post_transform is not None:
                img = self.post_transform(img)
            return img, int(label), self.target_class

        img, label = self.dataset[index]
        if index in self.poisoned_indices:
            harder = torch.as_tensor(np.asarray(self.harder_images[index], dtype=np.float32))
            img = apply_label_consistent_trigger(harder.permute(2, 0, 1) / 255.0, self.trigger, self.amplitude)
        else:
            img = self._to_tensor(img)

        if self.post_transform is not None:
            img = self.post_transform(img)
        return img, int(label)
