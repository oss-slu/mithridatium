from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from mithridatium.loader_hf import HFImageClassifier

def load_resnet18(model_path: str | None):
    """
    Load a ResNet-18 model with optional checkpoint.
    
    Args:
        model_path: Path to checkpoint file, or None for random init.
        
    Returns:
        Tuple of (model, feature_module).
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    feature_module = model.avgpool

    # try to load a checkpoint if provided
    if model_path and Path(model_path).exists():
        model, feature_module = detect_and_build(model_path, arch_hint="resnet18", num_classes=10)
    else:
        print(f"[loader] checkpoint not found at '{model_path}'. Using randomly initialized model (ok for pipeline tests).")

    model.eval()
    return model, feature_module

def get_feature_module(model):
    """
    Returns the penultimate feature module for a given model architecture.
    
    Args:
        model: PyTorch model instance.
        
    Returns:
        The feature extraction module (e.g., model.avgpool for ResNet).
        
    Raises:
        NotImplementedError: If architecture is not supported.
    """
    arch = model.__class__.__name__
    if arch == 'ResNet':
        return model.avgpool
    else:
        raise NotImplementedError(f"Feature module not defined for architecture: {arch}")

def _build_resnet18_cifar(num_classes: int = 10):
    """
    Build a CIFAR-adapted ResNet-18.

    Many backdoor research pipelines (BackdoorBench, TrojanZoo, etc.) use a
    modified ResNet-18 for small images (32x32) with:
        - conv1: 3x3 kernel, stride=1, padding=1  (instead of 7x7, stride=2, padding=3)
        - No maxpool layer after conv1
        - avgpool adjusted with AdaptiveAvgPool2d(1) (same as standard)

    This matches the architecture used in most CIFAR-10 backdoor papers.
    """
    m = models.resnet18(weights=None)
    # Replace the ImageNet-style conv1 (7x7) with CIFAR-style (3x3)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool — CIFAR images are too small for it
    m.maxpool = nn.Identity()
    # Adjust final classifier
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def detect_and_build(ckpt_path: str, arch_hint: str = "resnet18", num_classes: int = 10):
    """
    Auto-detect the architecture variant from checkpoint weights and build
    the matching model.

    This inspects the checkpoint's conv1.weight shape to determine if the
    model was trained with a standard ImageNet architecture (7x7 kernel) or
    a CIFAR-adapted architecture (3x3 kernel), then builds and loads the
    correct variant.

    Args:
        ckpt_path: Path to the checkpoint file.
        arch_hint: Base architecture family (e.g. "resnet18"). Used as a
            fallback when auto-detection can't determine the variant.
        num_classes: Number of output classes.

    Returns:
        Tuple of (model, feature_module) with weights loaded.
    """
    # Load and unwrap the checkpoint to inspect weight shapes
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _unwrap_state_dict(ckpt)

    # Auto-detect the variant from weight shapes
    detected_arch = _detect_resnet_variant(sd)

    if detected_arch != arch_hint:
        print(f"[loader] auto-detected architecture variant: '{detected_arch}' "
              f"(hint was '{arch_hint}')")

    # Build the correct model
    model, feature_module = build_model(detected_arch, num_classes)

    # Load weights into the correctly-shaped model
    missing, unexpected = model.load_state_dict(sd, strict=False)

    total_params = len(list(model.state_dict().keys()))
    loaded_params = total_params - len(missing)

    if loaded_params == 0:
        raise RuntimeError(
            f"No weights were loaded from '{ckpt_path}' even after "
            f"auto-detecting architecture '{detected_arch}'. "
            f"The checkpoint may be incompatible."
        )

    if missing:
        print(f"[warn] detect_and_build: {len(missing)} missing keys (partial load)")
    if unexpected:
        print(f"[warn] detect_and_build: {len(unexpected)} unexpected keys ignored")
    print(f"[loader] loaded {loaded_params}/{total_params} parameter tensors "
          f"from '{ckpt_path}' into '{detected_arch}'")

    return model, feature_module

def _detect_resnet_variant(state_dict: dict) -> str:
    """
    Inspect checkpoint weights to determine if this is a standard ImageNet
    ResNet or a CIFAR-adapted variant.

    Returns:
        "resnet18"       — standard ImageNet variant (conv1 is 7x7)
        "resnet18_cifar" — CIFAR-adapted variant (conv1 is 3x3)
    """
    conv1_key = "conv1.weight"
    if conv1_key not in state_dict:
        # Can't determine — fall back to standard
        return "resnet18"

    shape = state_dict[conv1_key].shape
    # Standard ImageNet ResNet-18 conv1: (64, 3, 7, 7)
    # CIFAR-adapted ResNet-18 conv1:     (64, 3, 3, 3)
    kernel_size = shape[-1]

    if kernel_size == 3:
        return "resnet18_cifar"
    else:
        return "resnet18"

def build_model(arch: str = "resnet18", num_classes: int = 10):
    """
    Build a model with the specified architecture.

    Supported:
      - resnet18
      - resnet18_cifar
      - resnet34
      - hf_resnet50
    """
    arch_lower = arch.lower()

    if arch_lower == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m, get_feature_module(m)

    elif arch_lower == "resnet18_cifar":
        m = _build_resnet18_cifar(num_classes)
        return m, get_feature_module(m)

    elif arch_lower == "resnet34":
        from torchvision.models import resnet34
        m = resnet34(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m, get_feature_module(m)

    elif arch_lower == "hf_resnet50":
        m = HFImageClassifier("microsoft/resnet-50")
        return m, None

    else:
        raise NotImplementedError(f"Architecture '{arch}' not yet supported")
 
def _unwrap_state_dict(ckpt: dict) -> dict:
    """
    Extract the raw state dict from a checkpoint that may be wrapped
    in a training checkpoint dict.

    Handles formats like:
        {'model_state_dict': {...}, 'epoch': 50, 'args': ...}
        {'state_dict': {...}, ...}
        {'model': {...}, ...}
        {'net': {...}, ...}
    Or a raw state dict with layer keys directly.
    """
    state_dict_keys = ["model_state_dict", "state_dict", "model", "net"]
    if isinstance(ckpt, dict):
        for key in state_dict_keys:
            if key in ckpt:
                print(f"[loader] found weights under '{key}' key, unwrapping")
                return ckpt[key]
    return ckpt

def validate_model(model: torch.nn.Module, arch: str, input_size):
    """
    Basic model validation:
    - Verify input_size looks correct
    - Run a dry forward pass
    - Verify output is [batch, num_classes]
    """
    if not isinstance(input_size, (tuple, list)) or len(input_size) != 3:
        raise ValueError(f"Invalid input_size for validation: {input_size} (expected (C, H, W))")

    C, H, W = input_size
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, C, H, W)

    with torch.no_grad():
        try:
            out = model_cpu(dummy)
        except Exception as ex:
            raise RuntimeError(
                "Dry forward pass failed — model architecture or weights "
                f"are incompatible with input size {input_size}.\nReason: {ex}"
            )

    if not isinstance(out, torch.Tensor):
        raise RuntimeError(
            f"Model forward must return a torch.Tensor of logits, got {type(out)}"
        )

    if out.ndim != 2:
        raise RuntimeError(
            f"Model forward must return logits of shape [batch, num_classes], got shape {tuple(out.shape)}"
        )

    if out.shape[0] != 1:
        raise RuntimeError(
            f"Validation forward pass expected batch dimension 1, got output shape {tuple(out.shape)}"
        )

    return True
