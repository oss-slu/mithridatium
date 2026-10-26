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
from torch.utils.data import DataLoader, Subset

from mithridatium import loader, report as rpt, utils


class NoisyBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm with a learned mask and adversarial noise per channel."""

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
        self.neuron_noise = nn.Parameter(torch.zeros_like(self.weight))
        self.noise_enabled = False
        self.noise_limit = 0.0
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

    def forward(self, images):
        noise = 0
        if self.noise_enabled:
            noise = self.neuron_noise.clamp(-self.noise_limit, self.noise_limit)

        mask = self.neuron_mask + noise
        return F.batch_norm(
            images,
            self.running_mean,
            self.running_var,
            self.weight * mask,
            self.bias * mask,
            self.training or not self.track_running_stats,
            self.momentum if self.momentum is not None else 0.0,
            self.eps,
        )


def _replace_bn(model):
    """Add ANP masks to every BatchNorm2d layer."""
    wrapped = {}

    def visit(parent, prefix=""):
        for name, layer in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(layer, nn.BatchNorm2d):
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


def train_masks(model, clean_loader, *, eps=0.4, alpha=0.2, steps=10):
    """Alternate noise maximization and mask minimization on clean images."""
    if not (math.isfinite(eps) and eps >= 0):
        raise ValueError("eps must be finite and nonnegative")
    if not (math.isfinite(alpha) and 0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")

    layers = _replace_bn(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    masks = [bn.neuron_mask for bn in layers.values()]
    noises = [bn.neuron_noise for bn in layers.values()]
    for parameter in masks + noises:
        parameter.requires_grad_(True)

    mask_optimizer = torch.optim.SGD(masks, lr=0.1)
    noise_optimizer = torch.optim.SGD(noises, lr=0.1)
    device = next(model.parameters()).device
    model.eval()
    seen = 0

    for images, labels, *_ in clean_loader:
        images, labels = images.to(device), labels.to(device)
        seen += len(labels)

        # Step 1: find noise that makes the model perform worse.
        for bn in layers.values():
            bn.noise_limit = eps
            bn.noise_enabled = eps > 0
            with torch.no_grad():
                bn.neuron_noise.uniform_(-eps, eps)

        if eps > 0:
            for _ in range(steps):
                noise_optimizer.zero_grad()
                mask_optimizer.zero_grad()
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

    if seen == 0:
        raise ValueError("Clean training dataset is empty")
    return {name: bn.neuron_mask.detach().clone() for name, bn in layers.items()}


def prune_by_threshold(model, scores, threshold):
    """Zero low-scoring BatchNorm channels and remove ANP wrappers."""
    if not (math.isfinite(threshold) and 0 <= threshold <= 1):
        raise ValueError("prune threshold must be between 0 and 1")

    pruned = []
    for name, bn in model.named_modules():
        if not isinstance(bn, NoisyBatchNorm2d):
            continue
        indices = torch.where(scores[name] <= threshold)[0]
        with torch.no_grad():
            bn.weight[indices] = 0
            bn.bias[indices] = 0
        if len(indices):
            pruned.append({"layer": name, "indices": indices.cpu().tolist(),
                           "count": len(indices)})

    _restore_bn(model)
    return pruned


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


@torch.no_grad()
def evaluate_targeted_asr(model, dataset, target_label):
    """Targeted ASR on triggered images paired with original clean labels."""
    if len(dataset) == 0:
        raise ValueError("Triggered evaluation dataset is empty")
    device = next(model.parameters()).device
    model.eval()
    hits = total = 0
    for images, labels, *_ in DataLoader(dataset, batch_size=64):
        labels = labels.to(device)
        eligible = labels != target_label
        predictions = model(images.to(device)).argmax(1)
        hits += ((predictions == target_label) & eligible).sum().item()
        total += eligible.sum().item()
    if total == 0:
        raise ValueError("Triggered evaluation requires non-target-class examples")
    return hits / total, total


def _sample(dataset, count, generator):
    """Return a reproducible subset of clean examples."""
    if len(dataset) == 0 or count < 1:
        raise ValueError("A nonempty dataset and positive sample count are required")
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def run_anp(*, model_path, out, report, eps=0.4, alpha=0.2,
            steps=10, prune_threshold=0.2, clean_samples=500,
            seed=0, version="0.1.2", data="cifar10", eval_samples=500,
            triggered_eval_dataset=None, attack_target=None):
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

    scores = train_masks(model, DataLoader(clean, batch_size=64),
                         eps=eps, alpha=alpha, steps=steps)
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
            "eps": eps, "alpha": alpha, "steps": steps,
            "prune_threshold": prune_threshold, "clean_samples": len(clean),
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
