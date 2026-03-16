# mithridatium/loader_hf.py

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification


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
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x to already be preprocessed tensor of shape [N, C, H, W].
        Returns logits of shape [N, num_classes].
        """
        outputs = self.model(pixel_values=x)
        return outputs.logits

    @property
    def num_classes(self) -> int:
        return int(self.model.config.num_labels)