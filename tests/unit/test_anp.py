import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

from mithridatium.repair.anp import (
    _replace_bn,
    prune_by_threshold,
    train_masks,
)

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 2 * 2, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_fake_loader():
    # 6 fake 2x2 images and labels
    inputs = torch.randn(6, 1, 2, 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=2)

def test_masks_stay_within_zero_to_one():
    model = TinyModel()
    loader = get_fake_loader()

    scores = train_masks(model, loader, eps=0.2, alpha=0.2, steps=2)

    # Masks should stay between 0 and 1
    for name, mask in scores.items():
        assert (mask >= 0.0).all()
        assert (mask <= 1.0).all()

def test_perturbations_respect_epsilon():
    model = TinyModel()
    loader = get_fake_loader()
    eps = 0.25

    train_masks(model, loader, eps=eps, steps=2)

    # Noise added to the BatchNorm
    noise = model.bn.neuron_noise
    assert (noise >= -eps).all()
    assert (noise <= eps).all()


def test_pruning_zeroes_selected_neurons():
    model = TinyModel()

    # Set known weights and biases on the BatchNorm
    model.bn.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    model.bn.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4])

    _replace_bn(model)

    # Channels 0 and 2 have scores < 0.25 so they should be pruned
    scores = {"bn": torch.tensor([0.1, 0.8, 0.2, 0.9])}
    pruned = prune_by_threshold(model, scores, threshold=0.25)

    assert len(pruned) == 1
    assert pruned[0]["layer"] == "bn"
    assert pruned[0]["indices"] == [0, 2]

    # Selected neurons are zeroed
    assert model.bn.weight[0] == 0.0
    assert model.bn.weight[2] == 0.0
    assert model.bn.bias[0] == 0.0
    assert model.bn.bias[2] == 0.0

    # Unselected neurons keep their values
    assert model.bn.weight[1] == 2.0
    assert model.bn.weight[3] == 4.0
    assert model.bn.bias[1] == 0.2
    assert model.bn.bias[3] == 0.4


def test_exported_state_dict_loads_into_plain_resnet18():
    # Plain ResNet-18
    model = models.resnet18(weights=None)

    wrapped = _replace_bn(model)
    scores = {name: torch.ones_like(bn.weight) for name, bn in wrapped.items()}
    prune_by_threshold(model, scores, threshold=0.5)

    # Check that plain ResNet-18 can load the state cleanly
    plain_resnet = models.resnet18(weights=None)
    plain_resnet.load_state_dict(model.state_dict(), strict=True)