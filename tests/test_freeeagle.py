import json
from pathlib import Path

import pytest
import torch
from torchvision.models import resnet18
from typer.testing import CliRunner

from mithridatium import loader
from mithridatium import report as rpt
from mithridatium.cli import app
from mithridatium.defenses.freeeagle import run_freeeagle
from mithridatium.utils import get_preprocess_config


runner = CliRunner()


def _configure_freeeagle(config, *, optimize_steps: int, threshold: float, num_classes: int = 10):
    config.freeeagle_num_classes = num_classes
    config.freeeagle_num_dummy = 1
    config.freeeagle_num_important_neurons = 5
    config.freeeagle_metric = "softmax_score"
    config.freeeagle_use_transpose_correction = False
    config.freeeagle_bound_on = True
    config.freeeagle_optimize_steps = optimize_steps
    config.freeeagle_learning_rate = 1e-2
    config.freeeagle_weight_decay = 5e-3
    config.freeeagle_anomaly_threshold = threshold
    config.freeeagle_inspect_layer_position = 2


def test_run_freeeagle_random_clean_model_not_flagged():
    torch.manual_seed(0)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    config = get_preprocess_config("cifar10")
    _configure_freeeagle(config, optimize_steps=2, threshold=2.0, num_classes=10)

    results = run_freeeagle(model, config, device=torch.device("cpu"))

    assert results["defense"] == "freeeagle"
    assert isinstance(results["anomaly_metric"], float)
    assert isinstance(results["anomaly_matrix"], list)
    assert len(results["anomaly_matrix"]) == 10
    assert isinstance(results["tendency_per_target"], list)
    assert len(results["tendency_per_target"]) == 10
    assert results["verdict"] == "likely clean"


def test_run_freeeagle_clean_checkpoint_executes_wrapper_contract():
    checkpoint_path = Path("models/resnet18_clean.pth")
    if not checkpoint_path.exists():
        pytest.skip("clean checkpoint not available in workspace")

    torch.manual_seed(0)
    model, _ = loader.detect_and_build(str(checkpoint_path), arch_hint="resnet18", num_classes=10)

    config = get_preprocess_config("cifar10")
    _configure_freeeagle(config, optimize_steps=5, threshold=7.0, num_classes=10)

    results = run_freeeagle(model, config, device=torch.device("cpu"))

    assert results["defense"] == "freeeagle"
    assert isinstance(results["anomaly_metric"], float)
    assert isinstance(results["anomaly_matrix"], list)
    assert len(results["anomaly_matrix"]) == 10
    assert isinstance(results["tendency_per_target"], list)
    assert len(results["tendency_per_target"]) == 10
    assert results["verdict"] in ("likely clean", "likely backdoored")


@pytest.mark.slow
def test_freeeagle_cli_detect_flags_poison_and_writes_report(tmp_path):
    checkpoint_path = Path("models/resnet18_poison.pth")
    if not checkpoint_path.exists():
        pytest.skip("poison checkpoint not available in workspace")

    out_path = tmp_path / "freeeagle_report.json"

    result = runner.invoke(
        app,
        [
            "detect",
            "-m",
            str(checkpoint_path),
            "-D",
            "freeeagle",
            "-d",
            "cifar10",
            "--freeeagle-optimize-steps",
            "5",
            "--freeeagle-anomaly-threshold",
            "2.0",
            "-o",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    assert out_path.exists()

    report_data = json.loads(out_path.read_text(encoding="utf-8"))
    rpt.validate_report_data(report_data)

    assert report_data["defense"] == "freeeagle"
    assert report_data["results"]["verdict"] == "likely backdoored"
