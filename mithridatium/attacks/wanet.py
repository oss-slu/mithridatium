"""WaNet warping-based backdoor (Nguyen & Tran, ICLR 2021).

"WaNet - Imperceptible Warping-based Backdoor Attack" (arXiv:2102.10369).
The trigger is a small, smooth warp. A random ``k x k`` field of offsets in
[-1, 1] is normalized to mean absolute value 1, upsampled bicubically to the
image size, scaled by ``s / H`` and added to the identity sampling grid
(Eq. 3-5 of the paper). Training also uses "noise mode": a further
``cross_ratio * pc`` share of samples is warped with the backdoor grid plus
per-sample random noise and keeps its true label, so the model learns only the
exact backdoor warp.

Written from the paper. Defaults and behavior (k=4, s=0.5, grid_rescale=1,
pc=0.1, cross_ratio=2, bicubic upsampling, align_corners=True, clamping to
[-1, 1], augmentation applied after warping, the backdoor set drawn afresh
each time rather than fixed) match the authors' release,
VinAIResearch/Warping-based_Backdoor_Attack-release. No code is copied from it:
it is AGPL-3.0, which would pull mithridatium off its license. Copyright covers
source code, not algorithms.

Because a bilinear warp with a clamped grid is a convex combination of pixels,
it commutes with per-channel normalization, so the same warp is correct on
pixel-space and normalized tensors.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def make_wanet_noise_grid(k: int = 4, height: int = 32, seed: Optional[int] = None) -> torch.Tensor:
    """Random warping field of shape (1, H, H, 2), mean absolute value 1 before upsampling."""
    if k < 2:
        raise ValueError("WaNet k must be at least 2")
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    offsets = torch.rand(1, 2, k, k, generator=gen) * 2 - 1
    offsets = offsets / torch.mean(torch.abs(offsets))
    return F.interpolate(offsets, size=height, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)


def make_identity_grid(height: int = 32) -> torch.Tensor:
    """Identity sampling grid of shape (1, H, H, 2), (x, y) order as grid_sample expects."""
    coords = torch.linspace(-1, 1, steps=height)
    rows, cols = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((cols, rows), 2)[None, ...]


def make_wanet_grid(noise_grid: torch.Tensor, s: float = 0.5, grid_rescale: float = 1.0) -> torch.Tensor:
    """Backdoor sampling grid: identity + s * noise / H, rescaled and clamped to [-1, 1]."""
    height = noise_grid.shape[1]
    grid = (make_identity_grid(height) + s * noise_grid / height) * grid_rescale
    return torch.clamp(grid, -1, 1)


def make_cross_grid(grid: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Noise-mode grid: the backdoor grid plus uniform [-1, 1] / H noise per position."""
    height = grid.shape[1]
    noise = torch.rand(1, height, height, 2, generator=generator) * 2 - 1
    return torch.clamp(grid + noise / height, -1, 1)


def apply_wanet(images: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Warp a CHW image or NCHW batch with a (1, H, W, 2) grid."""
    single = images.ndim == 3
    batch = images[None] if single else images
    grid = grid.to(device=batch.device, dtype=batch.dtype).expand(batch.shape[0], -1, -1, -1)
    warped = F.grid_sample(batch, grid, align_corners=True)
    return warped[0] if single else warped


class WaNetBackdoorDataset(Dataset):
    """Dataset wrapper for WaNet training and ASR evaluation.

    ``mode='train'``: every time a sample is drawn, with probability ``pc`` it is
    warped with the backdoor grid and relabeled to ``target_class``; with
    probability ``pc * cross_ratio`` it is warped with a noisy grid and keeps
    its label; otherwise it is left clean. As in the authors' release, the
    choice is made afresh each epoch and any class can be picked.
    ``mode='test_poison'`` yields ``(x, original_label, target_label)`` for every
    non-target sample, warped with the backdoor grid.

    The warp is applied between ``pre_transform`` (``ToTensor``) and
    ``post_transform``. Put augmentation in ``post_transform``: WaNet augments
    after warping.
    """

    def __init__(
        self,
        dataset,
        target_class: int,
        noise_grid: torch.Tensor,
        s: float = 0.5,
        grid_rescale: float = 1.0,
        pc: float = 0.1,
        cross_ratio: float = 2.0,
        mode: str = "train",
        pre_transform=None,
        post_transform=None,
    ):
        if mode not in {"train", "test_poison"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'train' or 'test_poison'.")
        if not 0 <= pc * (1 + cross_ratio) <= 1:
            raise ValueError("pc * (1 + cross_ratio) must be in [0, 1]")

        self.dataset = dataset
        self.target_class = int(target_class)
        self.noise_grid = noise_grid.clone()
        self.s = float(s)
        self.grid_rescale = float(grid_rescale)
        self.grid = make_wanet_grid(self.noise_grid, s=self.s, grid_rescale=self.grid_rescale)
        self.pc = float(pc)
        self.cross_ratio = float(cross_ratio)
        self.mode = mode
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        self._non_target_indices: list[int] = []
        if self.mode == "train":
            print(
                f"[wanet] each draw: backdoor p={self.pc}, noise p={self.pc * self.cross_ratio} (s={self.s})"
            )
        else:
            self._non_target_indices = [i for i in range(len(dataset)) if dataset[i][1] != self.target_class]
            print(f"[wanet] ASR subset size {len(self._non_target_indices)} (target={self.target_class})")

    def draw(self) -> str:
        """Pick 'backdoor', 'noise' or 'clean' for one training draw."""
        u = torch.rand(()).item()
        if u < self.pc:
            return "backdoor"
        if u < self.pc * (1 + self.cross_ratio):
            return "noise"
        return "clean"

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
            kind = self.draw()
            if kind == "backdoor":
                img = apply_wanet(img, self.grid)
                label = self.target_class
            elif kind == "noise":
                img = apply_wanet(img, make_cross_grid(self.grid))
        else:
            original_label = int(label)
            img = apply_wanet(img, self.grid)
            if self.post_transform is not None:
                img = self.post_transform(img)
            return img, original_label, self.target_class

        if self.post_transform is not None:
            img = self.post_transform(img)

        return img, int(label)
