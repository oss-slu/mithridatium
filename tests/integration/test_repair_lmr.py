import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from typer.testing import CliRunner

import mithridatium.cli as cli
from mithridatium.cli import app


runner = CliRunner()


def test_lmr_repair_end_to_end(tmp_path, monkeypatch):
    # Fake checkpoint path so CLI validation passes.
    model_path = tmp_path / "fake.pth"
    model_path.touch()

    out_path = tmp_path / "repaired.pth"
    report_path = tmp_path / "report.json"

    # Tiny model instead of ResNet18.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4, 3),
    )

    # Tiny clean dataset instead of downloading CIFAR-10.
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

    # Replace expensive model/data loading.
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
            "--method", "lmr",
            "--data", "cifar10",
            "--clean-samples", "6",
            "--seed", "0",
            "--lmr-target-class", "2",
            "--lmr-prune-ratio", "0.25",
            "--out", str(out_path),
            "--report", str(report_path),
        ],
    )

    assert result.exit_code == 0, (
    f"\nCLI output:\n{result.output}"
    f"\nException: {result.exception!r}")
    assert out_path.exists()
    assert report_path.exists()