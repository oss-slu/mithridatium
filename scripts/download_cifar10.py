"""
Download CIFAR-10 into Mithridatium's shared data directory.

Run from the project root:

    python -m scripts.download_cifar10
"""

from __future__ import annotations

from torchvision import datasets, transforms

from mithridatium.utils import DATA_ROOT


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    print(f"CIFAR-10 ready at: {DATA_ROOT}")


if __name__ == "__main__":
    main()