# BackdoorBench ANP adaptation for Mithridatium.
# Copyright (c) 2022 CUHK(SZ), SRIBD (BackdoorBench contributors).
# SPDX-License-Identifier: CC-BY-NC-4.0
# Source: https://github.com/SCLBD/BackdoorBench/blob/main/defense/anp.py
# License: https://github.com/SCLBD/BackdoorBench/blob/main/LICENSE
# Research: Dongxian Wu and Yisen Wang, NeurIPS 2021.
# Changes: Mithridatium model loading, BN pruning, checkpoint export,
# clean/triggered evaluation, and report generation.

"""Repair a backdoored ResNet-18 using Adversarial Neuron Pruning (ANP)."""

from __future__ import annotations

import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset

from mithridatium import loader, report as rpt, utils
from mithridatium.evaluation.asr import evaluate_targeted_asr


class NoisyBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm with a learned mask on weights and adversarial noise per channel."""

    def __init__(self, original: nn.BatchNorm2d):
        if not original.affine:
            raise ValueError("ANP requires BatchNorm layers with affine weights")

        super().__init__(
            original.num_features,
            eps=original.eps,
            momentum=original.momentum,
            track_running_stats=original.track_running_stats,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        self.load_state_dict(original.state_dict())
        self.neuron_mask = nn.Parameter(torch.ones_like(self.weight))
        # Changed ANP to provide separate noise parameters for weight and bias. 
        self.weight_noise = nn.Parameter(torch.zeros_like(self.weight))
        self.bias_noise = nn.Parameter(torch.zeros_like(self.bias))
        self.noise_enabled = False
        self.noise_limit = 0.0
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

    @property
    def neuron_noise(self):
        """Backward-compatibility alias pointing to weight_noise."""
        return self.weight_noise

    def forward(self, images):
        if self.noise_enabled:
            w_noise = self.weight_noise.clamp(-self.noise_limit, self.noise_limit)
            b_noise = self.bias_noise.clamp(-self.noise_limit, self.noise_limit)
            # Additive weight noise as in upstream ANP, so channels with small
            # masks are still perturbed
            weight = self.weight * (self.neuron_mask + w_noise)
            bias = self.bias * (1 + b_noise)
        else:
            weight = self.weight * self.neuron_mask
            bias = self.bias

        # Changed ANP to optimize masks using batch stats
        return F.batch_norm(
            images,
            self.running_mean,
            self.running_var,
            weight,
            bias,
            True,
            0.0,
            self.eps,
        )


def _replace_bn(model):
    """Add ANP masks to every BatchNorm2d layer."""
    wrapped = {}

    def visit(parent, prefix=""):
        for name, layer in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(layer, nn.BatchNorm2d):
                if isinstance(layer, NoisyBatchNorm2d):
                    wrapped[path] = layer
                else:
                    new_layer = NoisyBatchNorm2d(layer)
                    setattr(parent, name, new_layer)
                    wrapped[path] = new_layer
            else:
                visit(layer, path)

    visit(model)
    if not wrapped:
        raise ValueError("ANP requires BatchNorm2d layers")
    return wrapped


def _restore_bn(model):
    """Remove ANP masks so the checkpoint contains ordinary BatchNorm."""
    for parent in model.modules():
        for name, layer in list(parent.named_children()):
            if not isinstance(layer, NoisyBatchNorm2d):
                continue

            plain = nn.BatchNorm2d(
                layer.num_features,
                eps=layer.eps,
                momentum=layer.momentum,
                track_running_stats=layer.track_running_stats,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            plain.load_state_dict({
                key: value for key, value in layer.state_dict().items()
                if key in plain.state_dict()
            })
            setattr(parent, name, plain)


def _augment(images):
    """Random crop (padding 4) and horizontal flip, matching upstream ANP's clean set."""
    n, _, h, w = images.shape
    padded = F.pad(images, (4, 4, 4, 4))
    rows = torch.randint(0, 9, (n,)).tolist()
    cols = torch.randint(0, 9, (n,)).tolist()
    cropped = torch.stack([
        padded[i, :, r:r + h, c:c + w] for i, (r, c) in enumerate(zip(rows, cols))
    ])
    flip = torch.rand(n, device=images.device) < 0.5
    cropped[flip] = cropped[flip].flip(-1)
    return cropped


def train_masks(model, clean_loader, *, eps=0.4, alpha=0.2, steps=1, iterations=2000,
                augment=True):
    """Alternate noise maximization and mask minimization on clean images."""
    if not (math.isfinite(eps) and eps >= 0):
        raise ValueError("eps must be finite and nonnegative")
    if not (math.isfinite(alpha) and 0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    # Added iterations validation 
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if len(clean_loader) == 0:
        raise ValueError("Clean training dataset is empty")

    # Changed the model's BN layers with NoisyBatchNorm2d. It freezes running stats protecting running_mean and running_var 
    layers = _replace_bn(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    masks = [bn.neuron_mask for bn in layers.values()]
    noises = []
    for bn in layers.values():
        noises.append(bn.weight_noise)
        noises.append(bn.bias_noise)

    for parameter in masks + noises:
        parameter.requires_grad_(True)

    noise_optimizer = torch.optim.SGD(noises, lr=eps / steps)
    mask_optimizer = torch.optim.SGD(masks, lr=0.2, momentum=0.9)

    device = next(model.parameters()).device
    # Loop over the clean loader for the requested number of iterations,
    # re-iterating it rather than using itertools.cycle, which caches every batch
    def batches():
        while True:
            yield from clean_loader

    clean_iter = batches()

    for _ in range(iterations):
        images, labels, *_ = next(clean_iter)
        images, labels = images.to(device), labels.to(device)
        if augment:
            images = _augment(images)

        # Step 1: find noise that makes the model perform worse.
        for bn in layers.values():
            bn.noise_limit = eps
            bn.noise_enabled = eps > 0
            with torch.no_grad():
                bn.weight_noise.uniform_(-eps, eps)
                bn.bias_noise.uniform_(-eps, eps)

        if eps > 0:
            for _ in range(steps):
                noise_optimizer.zero_grad()
                mask_optimizer.zero_grad()
                # Forward pass on model
                (-F.cross_entropy(model(images), labels)).backward()
                for noise in noises:
                    if noise.grad is not None:
                        noise.grad.sign_()
                noise_optimizer.step()
                with torch.no_grad():
                    for noise in noises:
                        noise.clamp_(-eps, eps)

        # Step 2: learn masks that work with and without the noise.
        mask_optimizer.zero_grad()
        noise_optimizer.zero_grad()
        for bn in layers.values():
            bn.noise_enabled = False
        clean_loss = F.cross_entropy(model(images), labels)

        if eps > 0:
            for bn in layers.values():
                bn.noise_enabled = True
            noisy_loss = F.cross_entropy(model(images), labels)
            loss = alpha * clean_loss + (1 - alpha) * noisy_loss
        else:
            loss = clean_loss

        loss.backward()
        mask_optimizer.step()
        with torch.no_grad():
            for mask in masks:
                mask.clamp_(0, 1)
        for bn in layers.values():
            bn.noise_enabled = False

    return {name: bn.neuron_mask.detach().clone() for name, bn in layers.items()}


def _prunable_layers(model, scores):
    """BatchNorm layers that have learned mask scores (all of them, as in upstream ANP)."""
    return [
        (name, bn) for name, bn in model.named_modules()
        if isinstance(bn, nn.BatchNorm2d) and name in scores
    ]


def _zero_channels(model, selected):
    """Zero the selected BatchNorm channel weights and remove ANP wrappers."""
    pruned = []
    for name, bn, indices in selected:
        with torch.no_grad():
            # Changed so the upstream zeroes ONLY the BN weight
            bn.weight[indices] = 0
        if len(indices):
            pruned.append({"layer": name, "indices": indices.cpu().tolist(),
                           "count": len(indices)})

    _restore_bn(model)
    return pruned


def prune_by_threshold(model, scores, threshold):
    """Zero low-scoring BatchNorm channel weights and remove ANP wrappers."""
    if not (math.isfinite(threshold) and 0 <= threshold <= 1):
        raise ValueError("prune threshold must be between 0 and 1")

    selected = [
        (name, bn, torch.where(scores[name] <= threshold)[0])
        for name, bn in _prunable_layers(model, scores)
    ]
    return _zero_channels(model, selected)


# Added prune_by_fraction to support pruning by fraction alongside threshold
def prune_by_fraction(model, scores, fraction):
    """Prune exactly the lowest-scoring fraction of prunable BatchNorm channels."""
    if not (math.isfinite(fraction) and 0 <= fraction <= 1):
        raise ValueError("prune fraction must be between 0 and 1")

    layers = _prunable_layers(model, scores)
    if fraction == 0 or not layers:
        # nothing to prune, but still unwrap the ANP layers
        return _zero_channels(model, [])

    # Rank channels globally; a stable sort breaks ties (e.g. masks clamped to
    # 0 or 1) by position so exactly k channels are pruned
    all_scores = torch.cat([scores[name].flatten().cpu() for name, _ in layers])
    k = max(1, int(len(all_scores) * fraction))
    chosen = torch.zeros(len(all_scores), dtype=torch.bool)
    chosen[torch.argsort(all_scores, stable=True)[:k]] = True

    selected = []
    offset = 0
    for name, bn in layers:
        size = scores[name].numel()
        indices = torch.where(chosen[offset:offset + size])[0].to(bn.weight.device)
        selected.append((name, bn, indices))
        offset += size
    return _zero_channels(model, selected)


@torch.no_grad()
def evaluate_clean(model, dataset):
    """Accuracy on clean examples."""
    if len(dataset) == 0:
        raise ValueError("Clean evaluation dataset is empty")
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    for images, labels, *_ in DataLoader(dataset, batch_size=64):
        predictions = model(images.to(device)).argmax(1)
        correct += (predictions == labels.to(device)).sum().item()
    return correct / len(dataset)


def _sample(dataset, count, generator):
    """Return a reproducible subset of clean examples."""
    if len(dataset) == 0 or count < 1:
        raise ValueError("A nonempty dataset and positive sample count are required")
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def run_anp(*, model_path, out, report, eps=0.4, alpha=0.2,
            steps=1, iterations=2000, prune_threshold=0.2, prune_fraction=None,
            clean_samples=500, seed=0, version="0.1.2", data="cifar10",
            eval_samples=500, triggered_eval_dataset=None, attack_target=None):
    """Train masks, prune neurons, and write a checkpoint and repair report."""
    source, destination = Path(model_path).resolve(), Path(out).resolve()
    if source == destination:
        raise ValueError("Output cannot overwrite the input checkpoint")
    if report != "-" and Path(report).resolve() in (source, destination):
        raise ValueError("Report path must differ from model paths")
    if (triggered_eval_dataset is None) != (attack_target is None):
        raise ValueError("Supply both triggered examples and attack target")

    torch.manual_seed(seed)
    config = utils.get_preprocess_config(data)
    if report == "-":
        with redirect_stdout(sys.stderr):
            model, _ = loader.detect_and_build(
                model_path, arch_hint="resnet18", num_classes=config.get_num_classes()
            )
            train_loader, _ = utils.dataloader_for(data, "train", 64)
            test_loader, _ = utils.dataloader_for(data, "test", 64)
    else:
        model, _ = loader.detect_and_build(
            model_path, arch_hint="resnet18", num_classes=config.get_num_classes()
        )
        train_loader, _ = utils.dataloader_for(data, "train", 64)
        test_loader, _ = utils.dataloader_for(data, "test", 64)

    generator = torch.Generator().manual_seed(seed)
    clean = _sample(train_loader.dataset, clean_samples, generator)
    evaluation = _sample(test_loader.dataset, eval_samples, generator)

    clean_before = evaluate_clean(model, evaluation)
    asr_before = None
    if triggered_eval_dataset is not None:
        asr_before, _ = evaluate_targeted_asr(model, triggered_eval_dataset, attack_target)

    # As in upstream ANP: full batches of 128 drawn with replacement, one per iteration.
    # train_masks wraps and freezes the model in place; only the masks are trained.
    scores = train_masks(
        model,
        DataLoader(clean, batch_size=128, sampler=RandomSampler(
            clean, replacement=True, num_samples=iterations * 128)),
        eps=eps,
        alpha=alpha,
        steps=steps,
        iterations=iterations,
    )

    if prune_fraction is not None:
        pruned = prune_by_fraction(model, scores, prune_fraction)
    else:
        pruned = prune_by_threshold(model, scores, prune_threshold)

    clean_after = evaluate_clean(model, evaluation)
    asr_after = None
    asr_count = None
    if triggered_eval_dataset is not None:
        asr_after, asr_count = evaluate_targeted_asr(
            model, triggered_eval_dataset, attack_target
        )

    results = {
        "method": "anp",
        "status": "completed",
        "verdict": "repair_applied",
        "parameters": {
            "eps": eps, "alpha": alpha, "steps": steps, "iterations": iterations,
            "prune_threshold": prune_threshold if prune_fraction is None else None,
            "prune_fraction": prune_fraction, "clean_samples": len(clean),
            "eval_samples": len(evaluation), "seed": seed,
        },
        "total_channels_pruned": sum(item["count"] for item in pruned),
        "layer_breakdown": [
            {"layer": item["layer"], "channels_pruned": item["count"],
             "indices": item["indices"]} for item in pruned
        ],
        "clean_accuracy_before": clean_before,
        "clean_accuracy_after": clean_after,
    }
    if asr_after is not None:
        results.update({
            "asr_before": asr_before,
            "asr_after": asr_after,
            "asr_evaluation_samples": asr_count,
        })
    payload = rpt.build_repair_report(model_path, "anp", data, version, results)
    rpt.validate_report_data(payload)

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), destination)
    if report == "-":
        print(json.dumps(payload, indent=2))
    else:
        report_path = Path(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
