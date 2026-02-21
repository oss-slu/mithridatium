from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WhiteObjectHeuristic:
    """Heuristic semantic trigger: image contains a large 'white-ish' region.

    Intended for CIFAR-10 'horse' images to approximate a "white horse" trigger.
    This avoids patch injection: the image is unmodified; we only select a subset
    of naturally-occurring semantic samples.
    """

    v_min: float = 0.78
    s_max: float = 0.25
    frac_min: float = 0.18

    def __call__(self, pil_img) -> bool:
        hsv = np.asarray(pil_img.convert("HSV"), dtype=np.uint8)
        if hsv.ndim != 3 or hsv.shape[2] != 3:
            return False

        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0

        white_mask = (v >= float(self.v_min)) & (s <= float(self.s_max))
        frac = float(white_mask.mean())
        return frac >= float(self.frac_min)


class SemanticBackdoorDataset(Dataset):
    """Dataset wrapper for semantic backdoor training + ASR evaluation.

    - In *train* mode: poisons a subset of samples that match a semantic predicate
      (and are of a specified `source_class`) by relabeling them to `target_class`.
    - In *test_poison* mode: returns only semantic-triggered samples, yielding
      (x, original_label, target_label) triples for ASR measurement.
    """

    def __init__(
        self,
        dataset,
        *,
        poison_rate: float,
        source_class: int,
        target_class: int,
        semantic_predicate: Callable[[object], bool],
        mode: str = "train",
        pre_transform=None,
        post_transform=None,
        seed: int = 1,
    ):
        if mode not in {"train", "test_poison"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'train' or 'test_poison'.")

        self.dataset = dataset
        self.poison_rate = float(poison_rate)
        self.source_class = int(source_class)
        self.target_class = int(target_class)
        self.semantic_predicate = semantic_predicate
        self.mode = mode
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.seed = int(seed)

        self.candidate_indices: list[int] = self._build_candidate_indices()

        if self.mode == "train":
            requested_poison = int(self.poison_rate * len(self.dataset))
            poison_count = min(requested_poison, len(self.candidate_indices))
            rng = random.Random(self.seed)
            self.poisoned_indices = set(rng.sample(self.candidate_indices, poison_count))

            print(
                "[semantic] candidates="
                f"{len(self.candidate_indices)} (source_class={self.source_class})  "
                f"poisoned={len(self.poisoned_indices)}/{len(self.dataset)} (rate={self.poison_rate})"
            )
        else:
            self.poisoned_indices = set()
            print(
                "[semantic] ASR subset="
                f"{len(self.candidate_indices)} (source_class={self.source_class} -> target_class={self.target_class})"
            )

    def _build_candidate_indices(self) -> list[int]:
        candidates: list[int] = []
        for idx in self._iter_source_class_indices():
            img, label = self.dataset[idx]
            if int(label) != self.source_class:
                continue
            if self.semantic_predicate(img):
                candidates.append(int(idx))
        return candidates

    def _iter_source_class_indices(self) -> Iterable[int]:
        # CIFAR datasets expose targets as a list of ints; use it if available
        targets: Optional[Sequence[int]] = getattr(self.dataset, "targets", None)
        if targets is not None:
            for idx, y in enumerate(targets):
                if int(y) == self.source_class:
                    yield idx
            return

        # Fallback: scan all items (slower)
        for idx in range(len(self.dataset)):
            _, y = self.dataset[idx]
            if int(y) == self.source_class:
                yield idx

    def __len__(self) -> int:
        if self.mode == "test_poison":
            return len(self.candidate_indices)
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.mode == "test_poison":
            base_index = self.candidate_indices[index]
        else:
            base_index = index

        img, label = self.dataset[base_index]

        if self.pre_transform is not None:
            img = self.pre_transform(img)
        elif not isinstance(img, torch.Tensor):
            # Keep existing behavior consistent with BadNetDataset
            from torchvision import transforms

            img = transforms.ToTensor()(img)

        if self.mode == "train":
            if base_index in self.poisoned_indices:
                label = self.target_class
        else:
            # ASR mode: always a candidate, so provide (x, original, target)
            original_label = int(label)
            target_label = int(self.target_class)
            if self.post_transform is not None:
                img = self.post_transform(img)
            return img, original_label, target_label

        if self.post_transform is not None:
            img = self.post_transform(img)

        return img, int(label)
