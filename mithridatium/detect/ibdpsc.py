"""
IBD-PSC input-level backdoor detection (Hou et al., ICML 2024).
https://arxiv.org/abs/2405.09786

Paper Eq. 4: PSC(x) = 1/n * sum_i softmax(F_i(x))[y'], y' = argmax F(x), flag
when PSC(x) > T. F_i multiplies gamma and beta of the last k + i - 1 BN layers
by omega, and k is the fewest such layers that misclassify more than xi of a
small clean set (Algorithm 1). Defaults are the paper's: omega = 1.5, n = 5,
T = 0.9, xi = 0.6.

Written from the paper. No code taken from THUYimingLi/BackdoorBox (GPL-2.0);
mithridatium is MIT. BackdoorBox amplifies k+1..k+n layers; we follow the
paper's k..k+n-1. Amplified models run through torch.func.functional_call, so
the model passed in is never copied or modified.
"""

import torch
import torch.nn as nn
from typing import Any, Dict
from torch.func import functional_call
from torch.utils.data import DataLoader

# ponytail: flat batch cutoff, uncalibrated; starts at SCALE-UP's value but is
# IBD-PSC's own, since PSC and SPC scores are distributed differently.
FLAGGED_FRACTION_THRESHOLD = 0.20


def bn_layers(model) -> list[tuple[str, nn.BatchNorm2d]]:
    """Affine BatchNorm layers, in the order the model defines them."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d) and m.affine]


def amplify(bns, num_layers: int, scale: float) -> Dict[str, torch.Tensor]:
    """Parameter overrides that multiply gamma and beta of the last num_layers BN layers by scale."""
    return {
        f"{name}.{param}": getattr(bn, param) * scale
        for name, bn in bns[len(bns) - num_layers:]
        for param in ("weight", "bias")
    }


def clean_error(model, overrides, data, device, batch_size: int = 100) -> float:
    """Eq. 3: the fraction of labelled clean data misclassified with the overrides applied."""
    wrong = total = 0
    for images, labels in DataLoader(data, batch_size=batch_size):
        predictions = functional_call(model, overrides, (images.to(device),)).argmax(1).cpu()
        wrong += int((predictions != labels).sum())
        total += len(predictions)
    return wrong / total


def select_start_layer(model, bns, clean_data, scale: float, xi: float, device, batch_size: int = 100) -> int:
    """Algorithm 1: the fewest amplified BN layers, counted from the last, whose clean error exceeds xi."""
    for k in range(1, len(bns) + 1):
        if clean_error(model, amplify(bns, k, scale), clean_data, device, batch_size) > xi:
            return k
    raise ValueError(
        f"amplifying all {len(bns)} BN layers by {scale} never misclassified more than {xi:.0%} "
        f"of the clean images, so IBD-PSC has no starting layer; try a larger scale"
    )


def psc_scores(model, bns, data, layer_counts, scale: float, device, batch_size: int = 100):
    """Eq. 4 per input, in dataset order. Returns (scores, original predictions)."""
    amplified = [amplify(bns, k, scale) for k in layer_counts]
    scores, preds = [], []
    for images, _ in DataLoader(data, batch_size=batch_size):
        images = images.to(device)
        predictions = model(images).argmax(1)
        confidence = torch.stack([
            functional_call(model, overrides, (images,)).softmax(1).gather(1, predictions[:, None]).squeeze(1)
            for overrides in amplified
        ])
        scores.append(confidence.mean(0).cpu())
        preds.append(predictions.cpu())
    return torch.cat(scores), torch.cat(preds)


@torch.no_grad()
def ibdpsc(
        model,
        clean_data,
        suspect_data,
        scale: float = 1.5,
        num_models: int = 5,
        threshold: float = 0.9,
        xi: float = 0.6,
        batch_size: int = 100,
        device="cpu",
        ) -> Dict[str, Any]:
    """
    Computes IBD-PSC parameter-oriented scaling consistency (PSC) scores.

    Args:
        model: The model to evaluate. Needs BatchNorm layers; never modified.
        clean_data: Known-clean, labelled Dataset; only chooses the starting layer k.
        suspect_data: Dataset of inputs to score. Labels are ignored.
        scale: omega, the factor applied to gamma and beta of the amplified layers.
        num_models: n, the number of amplified models averaged per input.
        threshold: Flag an input when its PSC exceeds this value.
        xi: Clean error rate that ends the layer selection.
        batch_size: Forward-pass batch size.
        device: Device to run the computation on.

    Returns:
        A dictionary containing per-input PSC scores and the batch verdict.
    """
    model = model.to(device).eval()
    bns = bn_layers(model)
    if not bns:
        raise ValueError("IBD-PSC amplifies BatchNorm layers, and this model has none")

    start = select_start_layer(model, bns, clean_data, scale, xi, device, batch_size)
    # k + n - 1 can pass the BN count; like the authors' code, repeat the all-layers copy.
    layer_counts = [min(start + i, len(bns)) for i in range(num_models)]
    scores, preds = psc_scores(model, bns, suspect_data, layer_counts, scale, device, batch_size)

    flagged = scores > threshold   # Sec. 4.4 is a strict inequality
    num_flagged = int(flagged.sum())

    return {
        "mode": "input-level",
        "method": "ibdpsc",
        "verdict": (
            "likely backdoored"
            if num_flagged / len(scores) >= FLAGGED_FRACTION_THRESHOLD
            else "likely clean"
        ),
        "num_inputs": len(scores),
        "num_flagged": num_flagged,
        "threshold": float(threshold),
        "parameters": {
            "scale": scale,
            "num_models": num_models,
            "xi": xi,
            "clean_samples": len(clean_data),
            "start_layer": start,
            "layers_per_model": layer_counts,
            "bn_layers": [name for name, _ in bns[len(bns) - max(layer_counts):]],
        },
        "per_sample": [
            {"index": i, "predicted_label": int(p), "score": float(s), "flagged": bool(f)}
            for i, (p, s, f) in enumerate(zip(preds, scores, flagged))
        ],
    }
