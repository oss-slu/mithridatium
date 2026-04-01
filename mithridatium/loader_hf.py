# mithridatium/loader_hf.py

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from mithridatium.utils import PreprocessConfig


def build_huggingface_model(model_id: str):
    """
    Build a Hugging Face image classification model by model ID.
    Example model_id: 'microsoft/resnet-50'
    """
    m = HFImageClassifier(model_id)
    return m, None

class HFImageClassifier(nn.Module):
    """
    Wrap a Hugging Face image classification model so Mithridatium can treat it
    like a plain PyTorch classifier: model(x) -> logits

    Also exposes basic capability information so defenses can decide whether
    they can run or should fail clearly.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x to already be a float tensor of shape [N, C, H, W].
        Returns logits of shape [N, num_classes].

        Assumes Mithridatium dataloaders already produce correctly resized /
        normalized tensors for the model. So we pass pixel_values directly.
        """
        outputs = self.model(pixel_values=x)
        return outputs.logits

    @property
    def num_classes(self) -> int:
        return int(self.model.config.num_labels)

    def supports_feature_extraction(self) -> bool:
        """
        Current wrapper does not expose a stable intermediate feature module.
        """
        return False

    def get_feature_module(self):
        """
        Torchvision-style feature hook target is not currently exposed for HF models.
        """
        return None

    def get_preprocess_config(self, fallback_dataset: str = "cifar10_for_imagenet") -> PreprocessConfig:
        """
        Return a preprocessing config aligned as closely as possible with the HF model.
        Uses processor metadata when available, with safe defaults otherwise.
        """
        image_mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])
        size_info = getattr(self.processor, "size", None)

        height = 224
        width = 224

        if isinstance(size_info, dict):
            if "height" in size_info and "width" in size_info:
                height = int(size_info["height"])
                width = int(size_info["width"])
            elif "shortest_edge" in size_info:
                height = width = int(size_info["shortest_edge"])
        elif isinstance(size_info, int):
            height = width = int(size_info)

        return PreprocessConfig(
            input_size=(3, height, width),
            channels_first=True,
            value_range=(0.0, 1.0),
            mean=tuple(float(x) for x in image_mean),
            std=tuple(float(x) for x in image_std),
            normalize=True,
            ops=[f"resize:{height}", f"centercrop:{width}"],
            dataset=fallback_dataset,
        )