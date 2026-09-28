"""
SCALE-UP input-level backdoor detection (Guo et al., ICLR 2023).
https://openreview.net/forum?id=o0LFPcoFKnr

Paper Eq. 2: SPC(x) = sum_{n in S} I{C(n*x) == C(x)} / |S|, flag when SPC(x) > T,
with n*x clipped to [0, 1]. The paper leaves T to the defender and gives
S = {3,5,7,9,11} only as an example, so both defaults below are ours.

Written from the paper. No code taken from JunfengGo/SCALE-UP (no license) or
THUYimingLi/BackdoorBox (GPL-2.0); mithridatium is MIT.
"""

import torch
from typing import Any, Dict, Optional, Sequence
from mithridatium import utils

from mithridatium.defenses.mmbd import get_device

DEFAULT_SCALES = (3.0, 5.0, 7.0, 9.0, 11.0)

# ponytail: flat batch cutoff, uncalibrated. Swap for the paper's data-limited
# NSPC (Eq. 3-4) once a clean set exists.
FLAGGED_FRACTION_THRESHOLD = 0.20


@torch.no_grad()
def run_scaleup(
        model,
        configs,
        num_samples: int = 256,
        threshold: float = 0.5,
        scales: Sequence[float] = DEFAULT_SCALES,
        seed: Optional[int] = None,
        device=None,
        test_loader=None,
        ) -> Dict[str, Any]:
    """
    Computes SCALE-UP scaled prediction consistency (SPC) scores.

    Args:
        model: The model to evaluate.
        configs: Preprocess configuration.
        num_samples: Number of inputs to score.
        threshold: Flag an input when its SPC exceeds this value.
        scales: Pixel amplification factors applied to each input.
        seed: Seed for reproducibility.
        device: Device to run the computation on.
        test_loader: Dataloader to score; built from configs when omitted.

    Returns:
        A dictionary containing per-input SPC scores and the batch verdict.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device(0)

    model = model.to(device=device, dtype=torch.float32).eval()

    if test_loader is None:
        test_loader, _ = utils.dataloader_for(
            configs.get_dataset(),
            split="test",
            batch_size=128
        )

    scales = [float(s) for s in scales]
    lo, hi = configs.get_value_range()

    # Eq. 2 scales raw pixels, but the dataloaders yield normalized tensors, so
    # every batch round-trips through pixel space before and after scaling.
    normalize = configs.get_normalize()
    mean = torch.tensor(configs.get_mean(), device=device).view(1, -1, 1, 1) if normalize else 0.0
    std = torch.tensor(configs.get_std(), device=device).view(1, -1, 1, 1) if normalize else 1.0

    per_sample = []
    for images, _ in test_loader:
        if len(per_sample) >= num_samples:
            break

        batch = images[: num_samples - len(per_sample)].to(device, dtype=torch.float32)
        predictions = model(batch).argmax(1)
        raw = batch * std + mean

        hits = sum(
            (model((torch.clamp(raw * s, lo, hi) - mean) / std).argmax(1) == predictions).float()
            for s in scales
        )

        for label, score in zip(predictions.tolist(), (hits / len(scales)).tolist()):
            per_sample.append({
                "index": len(per_sample),
                "predicted_label": int(label),
                "score": score,
                "flagged": score > threshold,   # Eq. 2 is a strict inequality
            })

    if not per_sample:
        raise ValueError("Dataloader produced no inputs to score.")

    num_flagged = sum(s["flagged"] for s in per_sample)
    flagged_fraction = num_flagged / len(per_sample)

    return {
        "defense": "scaleup",
        "mode": "input-level",
        "method": "scaleup",
        "verdict": (
            "likely backdoored"
            if flagged_fraction >= FLAGGED_FRACTION_THRESHOLD
            else "likely clean"
        ),
        "num_inputs": len(per_sample),
        "num_flagged": num_flagged,
        "threshold": float(threshold),
        "parameters": {
            "num_samples": num_samples,
            "scales": scales,
            "seed": seed,
        },
        "dataset": str(configs.get_dataset()),
        "per_sample": per_sample,
    }
