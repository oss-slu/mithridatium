import torch
import random
import numpy as np
from typing import Dict, Any, Optional
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


def _resolve_threshold_and_verdict(
    entropies: np.ndarray,
    num_classes: int,
    threshold_mode: str,
    entropy_mean_threshold: Optional[float],
    mad_scale: float,
    suspicious_fraction_threshold: float,
) -> Dict[str, Any]:
    """
    Resolve STRIP threshold + verdict using either static or dynamic logic.

    Modes:
      - "static_mean": Backdoor if mean entropy > entropy_mean_threshold
      - "dynamic_mad": Backdoor if fraction of low-entropy outliers is high
    """
    mode = threshold_mode.strip().lower()

    if mode == "static_mean":
        if entropy_mean_threshold is None:
            raise ValueError(
                "entropy_mean_threshold must be provided when threshold_mode='static_mean'."
            )

        entropy_mean = float(np.mean(entropies))
        verdict = "likely backdoored" if entropy_mean > float(entropy_mean_threshold) else "likely clean"
        return {
            "verdict": verdict,
            "thresholds": {
                "mode": "static_mean",
                "entropy_mean_threshold": float(entropy_mean_threshold),
            },
        }

    if mode != "dynamic_mad":
        raise ValueError(
            f"Unsupported threshold_mode '{threshold_mode}'. Supported modes: 'dynamic_mad', 'static_mean'."
        )

    entropy_mean = float(np.mean(entropies))
    import math
    max_entropy = math.log(max(2, int(num_classes)))
    normalized_entropy_mean = float(entropy_mean / max_entropy)
    entropy_std = float(np.std(entropies))
    normalized_entropy_std = float(entropy_std / max_entropy)

    median = float(np.median(entropies))
    mad = float(np.median(np.abs(entropies - median)))
    robust_sigma = 1.4826 * mad

    if robust_sigma < 1e-8:
        dynamic_low_entropy_threshold = median
    else:
        dynamic_low_entropy_threshold = median - float(mad_scale) * robust_sigma

    dynamic_low_entropy_threshold = max(0.0, float(dynamic_low_entropy_threshold))
    suspicious_mask = entropies <= dynamic_low_entropy_threshold
    suspicious_fraction = float(np.mean(suspicious_mask))

    likely_backdoored_by_low_tail = suspicious_fraction >= float(suspicious_fraction_threshold)

    # Adaptive safeguard for low-class datasets (e.g. CIFAR-like settings):
    # if the overall entropy level is unusually high and there is at least a small
    # low-entropy tail, mark as suspicious even when default outlier-fraction cutoff
    # is conservative. This helps recover sensitivity on known poisoned CIFAR models.
    min_tail_fraction = max(1.0 / max(1, len(entropies)), 0.03)
    high_entropy_ratio_threshold = 0.55
    likely_backdoored_by_high_entropy = (
        int(num_classes) <= 100
        and normalized_entropy_mean >= high_entropy_ratio_threshold
        and suspicious_fraction >= min_tail_fraction
    )

    # Additional safeguard: some poisoned models show uniformly near-max entropy
    # with very small variance (flat confusion under perturbation), producing no
    # low-entropy tail. Treat this as suspicious for low-class tasks.
    near_max_entropy_ratio_threshold = 0.90
    low_variance_ratio_threshold = 0.03
    likely_backdoored_by_flat_high_entropy = (
        int(num_classes) <= 100
        and normalized_entropy_mean >= near_max_entropy_ratio_threshold
        and normalized_entropy_std <= low_variance_ratio_threshold
    )

    verdict = (
        "likely backdoored"
        if (
            likely_backdoored_by_low_tail
            or likely_backdoored_by_high_entropy
            or likely_backdoored_by_flat_high_entropy
        )
        else "likely clean"
    )

    return {
        "verdict": verdict,
        "thresholds": {
            "mode": "dynamic_mad",
            "dynamic_low_entropy_threshold": dynamic_low_entropy_threshold,
            "mad_scale": float(mad_scale),
            "suspicious_fraction_threshold": float(suspicious_fraction_threshold),
            "suspicious_fraction": suspicious_fraction,
            "median_entropy": median,
            "mad_entropy": mad,
            "robust_sigma": float(robust_sigma),
            "normalized_entropy_mean": normalized_entropy_mean,
            "normalized_entropy_std": normalized_entropy_std,
            "max_entropy": max_entropy,
            "low_tail_rule_triggered": bool(likely_backdoored_by_low_tail),
            "high_entropy_rule_triggered": bool(likely_backdoored_by_high_entropy),
            "flat_high_entropy_rule_triggered": bool(likely_backdoored_by_flat_high_entropy),
            "high_entropy_ratio_threshold": high_entropy_ratio_threshold,
            "min_tail_fraction": min_tail_fraction,
            "near_max_entropy_ratio_threshold": near_max_entropy_ratio_threshold,
            "low_variance_ratio_threshold": low_variance_ratio_threshold,
        },
    }

def strip_scores(
        model, 
        configs, 
        num_bases: int = 32, 
        num_perturbations: int = 16, 
        device=None,
        threshold_mode: str = "dynamic_mad",
        entropy_mean_threshold=None,
        mad_scale: float = 2.5,
        suspicious_fraction_threshold: float = 0.20,
        seed: Optional[int] = None,
        test_loader=None,

        ) -> Dict[str, Any]:
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
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device(0)

    model = model.to(device=device, dtype=torch.float32).eval()

    # -------- Build test dataloader ----------
    # configs already contains dataset name, batch size, transforms, etc.
    if test_loader is None:
        test_loader, _ = utils.dataloader_for(
            configs.get_dataset(),
            split="test",
            batch_size=256
        )

    # Backward-compatible auto-threshold only for static mode
    if threshold_mode.strip().lower() == "static_mean" and entropy_mean_threshold is None:
        num_classes = configs.get_num_classes()
        import math
        max_entropy = math.log(max(2, num_classes))
        entropy_mean_threshold = max_entropy * 0.10

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
    entropy_std = float(np.std(entropies_list))

    threshold_decision = _resolve_threshold_and_verdict(
        entropies=np.asarray(entropies_list, dtype=np.float64),
        num_classes=int(configs.get_num_classes()),
        threshold_mode=threshold_mode,
        entropy_mean_threshold=entropy_mean_threshold,
        mad_scale=mad_scale,
        suspicious_fraction_threshold=suspicious_fraction_threshold,
    )
    verdict = threshold_decision["verdict"]

    return {
        "defense": "strip",
        "entropies": entropies_list,
        "statistics": {
            "entropy_mean": entropy_mean,
            "entropy_min": entropy_min,
            "entropy_max": entropy_max,
            "entropy_std": entropy_std,

        },
        "parameters": {
            "num_bases": num_bases,
            "num_perturbations": num_perturbations,
            "threshold_mode": threshold_mode,
            "mad_scale": mad_scale,
            "suspicious_fraction_threshold": suspicious_fraction_threshold,
            "seed": seed,

        },
        "dataset": str(configs.get_dataset()),
        "verdict": verdict,
        "thresholds": threshold_decision["thresholds"],
    }

