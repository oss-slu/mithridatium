#!/usr/bin/env python3
"""Demo / sanity-check for an ``invisible`` (UAP-based) backdoor model.

This script is intended to be run after training has finished; its purpose is
to load a checkpoint, visualise the universal adversarial perturbation (UAP),
and optionally inspect how the model behaves on a small batch of clean and
triggered images.  We do not perform any formal evaluation here (see
``scripts/train_resnet18.py`` and the CLI for that), but the outputs are handy
for a quick manual sanity check or for generating figures for documentation.

Typical usage::

    python -m scripts.demo_invisible_trigger \
        --model models/resnet18_invisible_highxi.pth \
        --uap-path models/uap_highxi.pt \
        --uap-norm inf --uap-xi 0.1 --seed 42 \
        --target-class 0 \
        --num-images 6 \
        --save-dir demo_outputs

If a UAP file already exists on disk ``--uap-path`` it will be loaded.  When
no file is provided a fresh UAP is generated according to the given norm,
magnitude and seed; the value is automatically written to ``--uap-path`` if a
path was supplied.  The parameters should match those used during training so
that the same trigger is used for evaluation.

The script prints a few model predictions and saves a couple of PNGs.  By
default it operates on the CIFAR-10 test set and only considers samples whose
original label is *not* the target class (i.e. candidates for the attack).
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

from mithridatium.attacks.invisible import apply_invisible_trigger, create_random_uap
from mithridatium.loader import load_resnet18

# same normalisation used during training/evaluation
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
NORMALISE = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise an invisible-trigger model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trojaned ResNet-18 checkpoint")
    parser.add_argument("--uap-path", type=str, default=None,
                        help="File containing the universal perturbation."
                             "If missing one will be generated.")
    parser.add_argument("--uap-norm", choices=["inf", "2"], default="inf",
                        help="Lp norm used when generating a random UAP")
    parser.add_argument("--uap-xi", type=float, default=0.05,
                        help="Magnitude of the UAP when it is randomly generated")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (UAP + sample pick)")
    parser.add_argument("--target-class", type=int, default=0,
                        help="Class that the UAP is supposed to force predictions to")
    parser.add_argument("--num-images", type=int, default=8,
                        help="Number of example images to display")
    parser.add_argument("--save-dir", type=str, default="demo_outputs",
                        help="Directory where output figures will be written")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device to run on (cpu/cuda). auto-detected if not set")
    parser.add_argument("--compute-asr", action="store_true",
                        help="Compute approximate ASR over entire test set")
    return parser.parse_args()


def ensure_uap(args):
    """Load or generate a UAP tensor.

    If ``args.uap_path`` exists it is loaded.  Otherwise a new perturbation is
    sampled from the specified norm and magnitude.  The resulting tensor is
    saved to ``args.uap_path`` when a path is provided (mimicking
    ``train_resnet18`` behaviour).
    """
    if args.uap_path and os.path.exists(args.uap_path):
        uap = torch.load(args.uap_path)
        print(f"Loaded UAP from {args.uap_path}")
    else:
        uap = create_random_uap((3, 32, 32), xi=args.uap_xi, p=args.uap_norm, seed=args.seed)
        if args.uap_path:
            os.makedirs(os.path.dirname(args.uap_path), exist_ok=True)
            torch.save(uap, args.uap_path)
            print(f"Saved generated UAP to {args.uap_path}")
    return uap


def load_cifar10_test():
    """Return raw CIFAR-10 test dataset (values in [0,1])."""
    return datasets.CIFAR10("./data", train=False, download=True,
                             transform=transforms.ToTensor())


def select_candidates(dataset, target_class, num_samples, seed=None):
    """Pick ``num_samples`` indices from the test set that are not ``target_class``."""
    rng = random.Random(seed)
    indices = [i for i, (_, y) in enumerate(dataset) if y != target_class]
    return rng.sample(indices, min(num_samples, len(indices)))


def make_grid_image(tensors, nrow):
    """Utility to convert a batch tensor into a matplotlib-ready image."""
    grid = vutils.make_grid(tensors, nrow=nrow, pad_value=1.0)
    # move channel to last dimension and convert to numpy
    return grid.permute(1, 2, 0).cpu().numpy()


def compute_asr(model, dataset, uap, target_class, device):
    """Estimate attack success rate on the provided dataset.

    Only non-target samples are considered (mirrors ``InvisibleBackdoorDataset``
    behaviour).
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    model.to(device).eval()
    with torch.no_grad():
        for x, y in loader:
            mask = y != target_class
            if mask.sum() == 0:
                continue
            x = x[mask]
            x_trig = apply_invisible_trigger(x, uap)
            inp = NORMALISE(x_trig).to(device)
            preds = model(inp).argmax(1).cpu()
            correct += (preds == target_class).sum().item()
            total += len(preds)
    return 100.0 * correct / total if total > 0 else 0.0


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    # load model and perturbation
    model, _ = load_resnet18(args.model)
    model.to(device)

    uap = ensure_uap(args)

    # prepare dataset
    test_ds = load_cifar10_test()
    indices = select_candidates(test_ds, args.target_class, args.num_images, seed=args.seed)

    raw_samples = torch.stack([test_ds[i][0] for i in indices])
    triggered = apply_invisible_trigger(raw_samples, uap)

    # print predictions for the small batch
    with torch.no_grad():
        clean_inp = NORMALISE(raw_samples.to(device))
        trig_inp = NORMALISE(triggered.to(device))
        clean_preds = model(clean_inp).argmax(1).cpu().tolist()
        trig_preds = model(trig_inp).argmax(1).cpu().tolist()
    print(f"chosen sample indices: {indices}")
    print(f"clean predictions:   {clean_preds}")
    print(f"triggered predictions: {trig_preds}")

    # visualise
    os.makedirs(args.save_dir, exist_ok=True)
    plt.imsave(os.path.join(args.save_dir, "uap.png"),
               make_grid_image(uap.clamp(0, 1).unsqueeze(0), nrow=1))

    plt.imsave(os.path.join(args.save_dir, "clean_examples.png"),
               make_grid_image(raw_samples, nrow=len(raw_samples)))
    plt.imsave(os.path.join(args.save_dir, "triggered_examples.png"),
               make_grid_image(triggered, nrow=len(triggered)))
    print(f"saved grids to {args.save_dir}")

    if args.compute_asr:
        asr = compute_asr(model, test_ds, uap, args.target_class, device)
        print(f"estimated ASR on test set: {asr:.2f}%")


if __name__ == "__main__":
    main()
