import torch
import torch.nn as nn 


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