"""TED++ tube and locally adaptive ranking tests."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mithridatium import report as rpt
from mithridatium.detect.tedpp import TEDPLUS, tedpp

pytestmark = pytest.mark.unit


def _ranker(samples_per_class=3, beta=0.5):
    ranker = TEDPLUS.__new__(TEDPLUS)
    ranker.device = "cpu"
    ranker.SAMPLES_PER_CLASS = samples_per_class
    ranker.beta = beta
    ranker.DEFENSE_TRAIN_SIZE = samples_per_class * 2
    ranker.hook_handles = []
    return ranker


def test_tube_radius_is_maximum_class_neighborhood_diameter():
    ranker = _ranker()
    predictions = torch.tensor([0, 0, 0, 1, 1, 1])
    activations = torch.tensor([[0.0], [1.0], [2.0], [100.0], [101.0], [103.0]])

    radius = ranker.getTubeRadius(predictions, activations, "test")

    assert radius == pytest.approx(3.0)


def test_lar_uses_ted_rank_inside_tube_and_worst_rank_outside_in_input_order():
    ranker = _ranker(samples_per_class=2)
    defense_predictions = torch.tensor([0, 1, 0, 1])
    defense_activations = torch.tensor([[0.0], [2.0], [5.0], [12.0]])
    tube_radius = ranker.getTubeRadius(
        defense_predictions, defense_activations, "test"
    )

    result = ranker.getLayerRegionDistance(
        new_prediction=torch.tensor([1, 1]),
        new_activation=torch.tensor([[4.9], [30.0]]),
        new_temp_label="Suspect",
        h_defense_prediction=defense_predictions,
        h_defense_activation=defense_activations,
        layer="test",
        layer_test_region_individual={},
        tube_radius=tube_radius,
    )

    assert tube_radius == pytest.approx(10.0)
    assert result["test"]["Suspect"] == [1, 4]


def test_suspect_ranking_rejects_prediction_without_reference_class():
    ranker = _ranker(samples_per_class=2)

    with pytest.raises(ValueError, match="No clean reference samples"):
        ranker.getLayerRegionDistance(
            new_prediction=torch.tensor([2]),
            new_activation=torch.tensor([[1.0]]),
            new_temp_label="Suspect",
            h_defense_prediction=torch.tensor([0, 1]),
            h_defense_activation=torch.tensor([[0.0], [4.0]]),
            layer="test",
            layer_test_region_individual={},
            tube_radius=1.0,
        )


def test_suspect_ranking_rejects_mismatched_predictions_and_activations():
    ranker = _ranker(samples_per_class=2)

    with pytest.raises(ValueError, match="predictions for"):
        ranker.getLayerRegionDistance(
            new_prediction=torch.tensor([0, 1]),
            new_activation=torch.tensor([[1.0]]),
            new_temp_label="Suspect",
            h_defense_prediction=torch.tensor([0, 1]),
            h_defense_activation=torch.tensor([[0.0], [4.0]]),
            layer="test",
            layer_test_region_individual={},
            tube_radius=1.0,
        )


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2, bias=False)
        self.fc2 = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([[-1.0], [1.0]]))
            self.fc2.weight.copy_(torch.eye(2))

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_tedpp_returns_schema_valid_per_sample_results():
    model = TinyClassifier()
    clean_images = torch.tensor(
        [[-5.0], [-4.0], [-1.0], [-0.5], [0.5], [1.0], [4.0], [5.0]]
    )
    clean_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    suspects = TensorDataset(
        torch.tensor([[-4.0], [-0.5], [0.5], [4.0]]),
        torch.tensor([0, 0, 1, 1]),
    )
    results = tedpp(
        model,
        TensorDataset(clean_images, clean_labels),
        suspects,
        samples_per_class=3,
        beta=0.5,
        contamination=0.1,
        seed=0,
    )

    assert set(results) == {
        "mode",
        "method",
        "verdict",
        "num_inputs",
        "num_flagged",
        "threshold",
        "parameters",
        "per_sample",
    }
    assert (results["mode"], results["method"]) == ("input-level", "ted++")
    assert results["num_inputs"] == len(results["per_sample"]) == 4
    assert [sample["index"] for sample in results["per_sample"]] == list(range(4))
    assert results["num_flagged"] == sum(
        sample["flagged"] for sample in results["per_sample"]
    )
    rpt.validate_report_data(
        rpt.build_report("m.pth", "ted++", "cifar10", "0.0.0", results)
    )
    assert not any(module._forward_hooks for module in model.modules())
