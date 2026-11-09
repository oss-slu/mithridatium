import json
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typer.testing import CliRunner

import mithridatium.cli as cli
from mithridatium.cli import app
from mithridatium import report as rpt

runner = CliRunner()


@pytest.mark.parametrize("trigger_type", [None, "white-square", "checkerboard", "cross", "blend"])
def test_anp_repair_end_to_end(tmp_path, monkeypatch, trigger_type):
    # Fake checkpoint path so CLI validation passes 
    model_path = tmp_path / "fake.pth"
    model_path.touch()

    out_path = tmp_path / "repaired.pth"
    report_path = tmp_path / "report.json"

    # Tiny model with BatchNorm2d
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1, bias=False),
        nn.BatchNorm2d(4),
        nn.Flatten(),
        nn.Linear(4 * 2 * 2, 3),
    )

    # Tiny clean dataset 
    dataset = TensorDataset(
        torch.randn(6, 3, 2, 2),
        torch.tensor([0, 1, 2, 0, 1, 2]),
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
    )

    class FakeConfig:
        def get_num_classes(self):
            return 3

        def get_normalize(self):
            return True

        def get_mean(self):
            return (0.4914, 0.4822, 0.4465)

        def get_std(self):
            return (0.2023, 0.1994, 0.2010)

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
        lambda *args, **kwargs: (loader, FakeConfig()),
    )
    monkeypatch.setattr(cli, "VERSION", "0.1.2")

    args = [
            "repair",
            "--model", str(model_path),
            "--method", "anp",
            "--data", "cifar10",
            "--clean-samples", "6",
            "--seed", "0",
            "--anp-steps", "1",
            "--out", str(out_path),
            "--report", str(report_path),
    ]
    if trigger_type is not None:
        args.extend([
            "--anp-attack-target", "0",
            "--anp-trigger-type", trigger_type,
            "--anp-asr-samples", "4",
        ])
        if trigger_type == "blend":
            args.extend(["--anp-trigger-alpha", "0.3"])
        else:
            args.extend(["--anp-trigger-size", "1", "--anp-trigger-pos", "top-left"])
    result = runner.invoke(app, args)

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
    if trigger_type is not None:
        assert 0 <= payload["results"]["asr_before"] <= 1
        assert 0 <= payload["results"]["asr_after"] <= 1
        assert payload["results"]["asr_evaluation_samples"] == 4
    else:
        assert "asr_before" not in payload["results"]
        assert "asr_after" not in payload["results"]
        assert "asr_evaluation_samples" not in payload["results"]


@pytest.mark.parametrize("flags", [
    ["--anp-trigger-size", "2"],  # target is required
    ["--anp-trigger-type", "blend"],  # target is required
    ["--anp-attack-target", "0", "--anp-trigger-type", "unknown"],
    ["--anp-attack-target", "0", "--anp-trigger-type", "blend", "--anp-trigger-size", "2"],
    ["--anp-attack-target", "0", "--anp-trigger-alpha", "0.2"],
    ["--anp-attack-target", "0", "--anp-trigger-type", "blend", "--anp-trigger-alpha", "2"],
    ["--anp-attack-target", "10"],
    ["--anp-attack-target", "0", "--anp-trigger-pos", "center"],
    ["--anp-attack-target", "0", "--anp-asr-samples", "0"],
    ["--anp-attack-target", "0", "--data", "imagenet"],
])
def test_anp_invalid_asr_options_return_usage_error(flags):
    result = runner.invoke(app, ["repair", "--method", "anp", *flags])
    assert result.exit_code == cli.EXIT_USAGE_ERROR, result.output
