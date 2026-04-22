import pytest
import torch
from torchvision.models import resnet18

from mithridatium.defenses.freeeagle import run_freeeagle
from mithridatium.utils import get_preprocess_config


def test_run_freeeagle_resnet_outputs_contract():
    torch.manual_seed(0)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)

    config = get_preprocess_config("cifar10")
    config.freeeagle_num_classes = 3
    config.freeeagle_num_dummy = 1
    config.freeeagle_optimize_steps = 2
    config.freeeagle_learning_rate = 1e-2
    config.freeeagle_weight_decay = 0.0
    config.freeeagle_metric = "softmax_score"
    config.freeeagle_inspect_layer_position = 2

    results = run_freeeagle(model, config, device=torch.device("cpu"))

    assert results["defense"] == "freeeagle"
    assert isinstance(results["anomaly_metric"], float)
    assert isinstance(results["anomaly_matrix"], list)
    assert len(results["anomaly_matrix"]) == 3
    assert isinstance(results["tendency_per_target"], list)
    assert len(results["tendency_per_target"]) == 3
    assert results["verdict"] in ("likely clean", "likely backdoored")
    assert "parameters" in results
    assert results["parameters"]["inspect_layer_position"] == 2


def test_run_freeeagle_rejects_unknown_architecture():
    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x[:, :4])

    model = TinyNet()
    config = get_preprocess_config("cifar10")

    with pytest.raises(NotImplementedError):
        run_freeeagle(model, config, device=torch.device("cpu"))