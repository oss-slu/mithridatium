"""Fetch one harder-to-classify image corpus for the label-consistent attack.

The authors of "Label-Consistent Backdoor Attacks" (MadryLab/label-consistent-
backdoor-code, MIT License) release every CIFAR-10 training image replaced by a
harder version of itself: adversarial perturbations with an l2 bound
(``two_<eps>.npy``) or an l-inf bound (``inf_<eps>.npy``), and GAN interpolations
(``gan_0_<tau>.npy``). The release is about 4.5 GB split in three parts, plus
their clean CIFAR-10 (160 MB).

This script downloads the parts, extracts only the requested corpus, and checks
that the authors' clean training set is in the same order as torchvision's
CIFAR-10, since the corpus is indexed by training-set position.

Run from the project root on Libra (data/ is git-ignored):

    python -m scripts.fetch_label_consistent --corpus two_300
    python scripts/train_resnet18.py --dataset label_consistent \\
        --lc-images data/label_consistent/two_300.npy ...
"""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

RELEASE = "https://github.com/MadryLab/label-consistent-backdoor-code/releases/download/v1.0"
PARTS = [f"fully_poisoned_training_datasets.tar.bz2.{suffix}" for suffix in ("aa", "ab", "ac")]
CLEAN = "clean_dataset.tar.bz2"


def download(name: str, out_dir: Path) -> Path:
    target = out_dir / name
    if target.exists():
        print(f"have {target}")
        return target
    print(f"downloading {name}")
    partial = target.with_suffix(target.suffix + ".partial")
    urllib.request.urlretrieve(f"{RELEASE}/{name}", partial)
    partial.rename(target)
    return target


class ConcatReader(io.RawIOBase):
    """Read several files back to back, like ``cat a b c``."""

    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]

    def readable(self):
        return True

    def readinto(self, buffer):
        while self.files:
            n = self.files[0].readinto(buffer)
            if n:
                return n
            self.files.pop(0).close()
        return 0


def extract_member(stream, wanted: str, out_path: Path) -> None:
    with tarfile.open(fileobj=stream, mode="r|bz2") as archive:
        for member in archive:
            if Path(member.name).name == wanted:
                with archive.extractfile(member) as src, open(out_path, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                return
    raise SystemExit(f"{wanted} is not in the archive")


def check_order(clean_dir: Path, data_root: str) -> None:
    """The authors' clean train set must match torchvision's CIFAR-10, image for image."""
    from torchvision import datasets

    theirs_images = np.load(clean_dir / "train_images.npy")
    theirs_labels = np.load(clean_dir / "train_labels.npy")
    ours = datasets.CIFAR10(data_root, train=True, download=True)
    if not np.array_equal(theirs_labels.astype(np.int64), np.asarray(ours.targets, dtype=np.int64)):
        raise SystemExit("label order differs from torchvision CIFAR-10; the corpus can't be used as is")
    if not np.array_equal(theirs_images.astype(np.uint8), ours.data):
        raise SystemExit("image order differs from torchvision CIFAR-10; the corpus can't be used as is")
    print("clean training set matches torchvision CIFAR-10 order")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="two_300", help="corpus name without .npy (default: two_300, the authors' config)")
    parser.add_argument("--out-dir", default="data/label_consistent")
    parser.add_argument("--data-root", default="./data", help="torchvision CIFAR-10 root")
    parser.add_argument("--keep-parts", action="store_true", help="keep the downloaded archive parts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = out_dir / "clean_dataset"
    if not (clean_dir / "train_labels.npy").exists():
        clean_archive = download(CLEAN, out_dir)
        with tarfile.open(clean_archive, "r:bz2") as archive:
            archive.extractall(out_dir, filter="data")
    check_order(clean_dir, args.data_root)

    out_path = out_dir / f"{args.corpus}.npy"
    if out_path.exists():
        print(f"have {out_path}")
    else:
        parts = [download(name, out_dir) for name in PARTS]
        print(f"extracting {args.corpus}.npy")
        extract_member(io.BufferedReader(ConcatReader(parts)), f"{args.corpus}.npy", out_path)
        if not args.keep_parts:
            for part in parts:
                part.unlink()

    corpus = np.load(out_path, mmap_mode="r")
    if corpus.shape != (50000, 32, 32, 3):
        raise SystemExit(f"unexpected corpus shape {corpus.shape}")
    print(f"{out_path}: {corpus.shape}, {corpus.dtype}, values {float(corpus.min()):.1f}..{float(corpus.max()):.1f}")


if __name__ == "__main__":
    main()
