"""
This implementation of Channel Lipschitzness-based Pruning (CLP) is adapted from 
the BackdoorBench project.
Original Authors: The Chinese University of Hong Kong, Shenzhen (CUHK-SZ) and 
Shenzhen Research Institute of Big Data (SRIBD).
Original License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
License link: https://creativecommons.org/licenses/by-nc/4.0/legalcode
Modifications: Adapted the pruning logic to fit the mithridatium repair pipeline, 
including vectorization improvements and custom CLI dispatching.
"""


import torch
import torch.nn as nn 
import json
from datetime import datetime, timezone
from pathlib import Path
from mithridatium import loader


# Gives each output channel a score to know which is backdoor 
def _uclc_score(weight: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
    w = weight.detach().float()
    mats = w.reshape(w.shape[0], w.shape[1], -1)       
    sigma = torch.linalg.matrix_norm(mats, ord=2)        
    scale = bn.weight.detach().float() / torch.sqrt(bn.running_var.float() + bn.eps)
    #Returns channel scores
    return sigma * scale.abs()


# Turns off the channels that score way above the rest and returns indexs of the possliby backdoored channels
def _prune_channels(bn: nn.BatchNorm2d, score: torch.Tensor, threshold_mult: float) -> list[int]:
    if score.numel() != bn.num_features:
        raise ValueError(f"score has {score.numel()} channels but BN has {bn.num_features}")
    if not bn.affine:
        raise ValueError("CLP needs an affine BatchNorm (gamma/beta) to prune")
    threshold = score.mean() + threshold_mult * score.std()
    idx = torch.nonzero(score > threshold).flatten()
    with torch.no_grad():
        bn.weight[idx] = 0.0
        bn.bias[idx] = 0.0
    # Return the index of pruned channels
    return idx.tolist()


def repair_clp(model_path: str, out: str, report: str, threshold_mult: float = 3.0) -> None:

    model, _ = loader.detect_and_build(str(model_path), arch_hint="resnet18", num_classes=10)
    
    model.eval()

    #  Iterates through the network to pair Conv2d and BatchNorm2d layers.
    last_conv: nn.Conv2d | None = None
    last_name: str = ""
    layer_log: list[dict] = []
    total_pruned = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # If we find a Conv2d we store it to remember it for the next step
            last_conv = module
            last_name = name
            
        elif isinstance(module, nn.BatchNorm2d) and last_conv is not None:
            
            # makes sure the output channels of the Conv match the features of the BN.
            if last_conv.out_channels == module.num_features:
                
                # Calculate the sensitivity score 
                score = _uclc_score(last_conv.weight, module)
                
                # finds outliers and sets there Batchnorm weights/biases to zero
                pruned_idx = _prune_channels(module, score, threshold_mult)
                
                # Keep track of how many channels we turned off for the report
                total_pruned += len(pruned_idx)
                if pruned_idx:
                    layer_log.append(
                        {"layer": last_name, "channels_pruned": len(pruned_idx)}
                    )
                    
            last_conv = None

    # Saves the repaired checkpoint to the output path and checks the dirctotry is real first
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)

    report_data = {
        "mithridatium_version": "0.1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "defense": "clp",
        "dataset": "n/a (data-free)", 
        "results": {
            "method": "clp",
            "parameters": {"threshold_mult": threshold_mult},
            "total_channels_pruned": total_pruned,
            "layer_breakdown": layer_log,
            "status": f"pruned {total_pruned} channels (u={threshold_mult})",
        },
    }

    # Write the report
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as f:
        json.dump(report_data, f, indent=2)
