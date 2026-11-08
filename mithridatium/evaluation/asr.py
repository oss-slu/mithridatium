"""Architecture-independent targeted attack success rate evaluation.

The supplied dataset must yield ``(triggered_input, original_clean_label)``.
It is the caller's responsibility to generate the *correct* trigger for the
attack that was used to poison the model.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_targeted_asr(model, dataset, target_label: int) -> tuple[float, int]:
    """Return (targeted ASR, eligible count) for a triggered dataset.

    Inputs with an original clean label equal to the attack target are excluded
    from the ASR denominator. Works with any torch classification model that
    outputs class logits and any map-style triggered evaluation dataset.
    """
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
