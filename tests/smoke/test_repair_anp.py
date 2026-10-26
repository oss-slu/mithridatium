import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typer.testing import CliRunner

import mithridatium.cli as cli
from mithridatium.cli import app
from mithridatium import report as rpt

runner = CliRunner()


def test_anp_repair_end_to_end(tmp_path, monkeypatch):
    # Fake checkpoint path so CLI validation passes 
    model_path = tmp_path / "fake.pth"
    model_path.touch()

    out_path = tmp_path / "repaired.pth"
    report_path = tmp_path / "report.json"

    # Tiny model with BatchNorm2d
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=1, bias=False),
        nn.BatchNorm2d(4),
        nn.Flatten(),
        nn.Linear(4 * 2 * 2, 3),
    )

    # Tiny clean dataset 
    dataset = TensorDataset(
        torch.randn(6, 1, 2, 2),
        torch.tensor([0, 1, 2, 0, 1, 2]),
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
    )

    class FakeConfig:
        def get_num_classes(self):
            return 3

    # Replace expensive model
    monkeypatch.setattr(
        cli.utils,
        "get_preprocess_config",
        lambda _: FakeConfig(),
    )

    monkeypatch.setattr(
        cli.loader,
        "detect_and_build",
        lambda *args, **kwargs: (model, None),
    )

    monkeypatch.setattr(
        cli.utils,
        "dataloader_for",
        lambda *args, **kwargs: (loader, None),
    )

    result = runner.invoke(
        app,
        [
            "repair",
            "--model", str(model_path),
            "--method", "anp",
            "--data", "cifar10",
            "--clean-samples", "6",
            "--seed", "0",
            "--anp-steps", "1",
            "--out", str(out_path),
            "--report", str(report_path),
        ],
    )

    assert result.exit_code == 0, (
        f"\nCLI output:\n{result.output}"
        f"\nException: {result.exception!r}"
    )
    assert out_path.exists()
    assert report_path.exists()

    payload = json.loads(report_path.read_text())
    rpt.validate_report_data(payload)
    assert payload["method"] == "anp"
    assert payload["results"]["total_channels_pruned"] >= 0