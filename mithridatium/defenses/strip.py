import torch
import random
import numpy as np
from typing import Dict, Any, List

from mithridatium import utils
from mithridatium.defenses.mmbd import get_device

def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns per-sample entropy over the softmax distribution.

    Args:
        logits: A tensor of shape (batch_size, num_classes) containing the logits.

    Returns:
        A tensor of shape (batch_size,) containing the entropy for each sample.
    """
    p = torch.nn.Softmax(dim=1)(logits) + 1e-8
    return (-p * p.log()).sum(1)

def strip_scores(model, configs, num_bases: int = 32, num_perturbations: int = 16, device=None, entropy_mean_threshold=0.45 ) -> Dict[str, Any]:
    """
    Computes STRIP-style entropy scores.

    Args:
        model: The model to evaluate.
        configs: Preprocess configuration.
        num_bases: Number of base samples to evaluate.
        num_perturbations: Number of perturbations per base sample.
        device: Device to run the computation on.

    Returns:
        A dictionary containing the raw entropy scores.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device(0)

    model = model.to(device=device, dtype=torch.float32).eval()

    # -------- Build test dataloader ----------
    # configs already contains dataset name, batch size, transforms, etc.
    test_loader, _ = utils.dataloader_for(
        configs.get_dataset(),
        split="test",
        batch_size=256
    )


    # Collect all images from the dataloader to use as a pool for mixing
    all_images = []
    for images, _ in test_loader:
        all_images.append(images)
        if len(all_images) * images.shape[0] >= num_bases + num_perturbations * 2: # Heuristic to stop early if we have enough data
             break
    
    if not all_images:
         raise ValueError("Dataloader is empty")

    all_images = torch.cat(all_images, dim=0)
    
    # Ensure we have enough images
    if len(all_images) < num_bases:
        num_bases = len(all_images)
        # raise ValueError(f"Not enough images in dataloader. Needed {num_bases}, got {len(all_images)}")

    # Select base samples
    indices = torch.randperm(len(all_images))
    base_indices = indices[:num_bases]
    base_images = all_images[base_indices].to(device, dtype=torch.float32)

    entropies_list = []

    with torch.no_grad():
        for i in range(num_bases):
            base_img = base_images[i]
            
            # Create perturbations
            # We need num_perturbations other images. 
            # We can sample from the whole pool (excluding the current base if we want, but collision prob is low)
            perturb_indices = torch.randint(0, len(all_images), (num_perturbations,))
            perturb_images = all_images[perturb_indices].to(device, dtype=torch.float32)
            
            # Superimpose: 0.5 * base + 0.5 * other
            # base_img is (C, H, W), perturb_images is (N, C, H, W)
            # Broadcast base_img
            mixed_images = 0.5 * base_img.unsqueeze(0) + 0.5 * perturb_images
            
            logits = model(mixed_images)
            entropies = prediction_entropy(logits)
            
            # Aggregate entropy for this base sample
            mean_entropy = entropies.mean().item()
            entropies_list.append(mean_entropy)

    if not entropies_list:
        raise ValueError("No entropies were computed.")

    entropy_mean = float(np.mean(entropies_list))
    entropy_min  = float(np.min(entropies_list))
    entropy_max  = float(np.max(entropies_list))

    if entropy_mean > entropy_mean_threshold:
        verdict = "likely backdoored"
    else:
        verdict = "likely clean"

    return {
        "defense": "strip",
        "entropies": entropies_list,
        "statistics": {
            "entropy_mean": entropy_mean,
            "entropy_min": entropy_min,
            "entropy_max": entropy_max,
        },
        "parameters": {
            "num_bases": num_bases,
            "num_perturbations": num_perturbations,
        },
        "dataset": str(configs.get_dataset()),
        "verdict": verdict,
        "thresholds": {
            "entropy_mean_threshold": entropy_mean_threshold
        }
    }

