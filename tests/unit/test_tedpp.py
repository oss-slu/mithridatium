"""TED++ unit tests using a tiny model and synthetic image data."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mithridatium import report as rpt
from mithridatium.detect.tedpp import TEDPLUS, _rank_vectors

pytestmark = pytest.mark.unit


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            for channel in range(3):
                self.conv.weight[channel, channel, 1, 1] = 1.0
            self.fc.weight.copy_(torch.eye(3))

    def forward(self, images):
        features = self.relu(self.conv(images))
        return self.fc(features.mean(dim=(2, 3)))


def _class_data(samples_per_class=6):
    images = []
    labels = []
    generator = torch.Generator().manual_seed(7)
    for class_id in range(3):
        for _ in range(samples_per_class):
            image = torch.zeros(3, 8, 8)
            image[class_id] = 1.0 + 0.01 * torch.rand(
                8, 8, generator=generator
            )
            images.append(image)
            labels.append(class_id)
    return TensorDataset(torch.stack(images), torch.tensor(labels))


def test_tedpp_returns_valid_report_and_removes_hooks():
    model = TinyClassifier()
    reference_data = _class_data()
    suspect_data = TensorDataset(*reference_data[:9])
    detector = TEDPLUS(
        model=model,
        reference_data=reference_data,
        suspect_data=suspect_data,
        reference_size=len(reference_data),
        samples_per_class=5,
        contamination=0.1,
        batch_size=4,
        seed=3,
    )

    result = detector.detect()

    assert (result["mode"], result["method"]) == ("input-level", "tedpp")
    assert result["num_inputs"] == len(result["per_sample"]) == 9
    assert 0 <= result["num_flagged"] <= 9
    rpt.validate_report_data(
        rpt.build_report("m.pth", "tedpp", "cifar10", "0.0.0", result)
    )
    assert not any(module._forward_hooks for module in model.modules())


def test_rank_vectors_assign_worst_rank_outside_tube_or_without_class():
    reference_activations = torch.tensor([[0.0], [1.0]])
    reference_predictions = torch.tensor([0, 0])
    query_activations = torch.tensor([[0.1], [10.0], [0.0]])
    query_predictions = torch.tensor([0, 0, 1])

    ranks = _rank_vectors(
        query_activations,
        query_predictions,
        reference_activations,
        reference_predictions,
        tube_radius=1.0,
        leave_one_out=False,
    )

    assert ranks.tolist() == [0.0, 2.0, 2.0]
