import torch
import torch.nn as nn
from typing import Tuple

def extract_embeddings(model: nn.Module, loader: torch.utils.data.DataLoader, feature_module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract penultimate-layer embeddings and labels from a model and dataloader.
    Args:
        model: The neural network model (e.g., resnet18).
        loader: DataLoader for the dataset.
        feature_module: The module in the model whose output is the embedding (e.g., model.avgpool or model.layer4).
    Returns:
        embs: Tensor of shape [N, D] (embeddings)
        labels: Tensor of shape [N] (labels)
    """
    model.eval()
    embs = []
    labels = []
    device = next(model.parameters()).device

    def hook_fn(module, input, output):
        hook_fn.embeddings = output.detach()
    hook_fn.embeddings = None
    hook = feature_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _ = model(x)
            emb = hook_fn.embeddings
            if emb.dim() > 2:
                emb = torch.flatten(emb, start_dim=1)
            embs.append(emb.cpu())
            labels.append(y.cpu())
    hook.remove()
    embs = torch.cat(embs, dim=0)
    labels = torch.cat(labels, dim=0)
    return embs, labels

def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.
    Args:
        model: The neural network model.
        loader: DataLoader for the dataset.
    Returns:
        loss: Average loss (float)
        accy: Accuracy (float)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / total
    accy = correct / total
    return avg_loss, accy
