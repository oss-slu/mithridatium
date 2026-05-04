"""
Unit tests for the Mithridatium CLI.

These tests mock model loading, dataloading, report validation, and defense
execution so they can verify CLI behavior without running expensive defenses
or requiring real checkpoints/datasets.
"""

from __future__ import annotations

import pytest
import torch
from typer.testing import CliRunner

from mithridatium.cli import (
    EXIT_NO_INPUT,
    EXIT_USAGE_ERROR,
    app,
)
from mithridatium.utils import get_preprocess_config


pytestmark = pytest.mark.unit

runner = CliRunner()


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.fc = torch.nn.Linear(4, 10)
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((x.shape[0], 10), device=x.device)


def _valid_result(defense: str) -> dict:
    if defense == "mmbd":
        return {
            "defense": "mmbd",
            "suspected_backdoor": False,
            "num_flagged": 0,
            "top_eigenvalue": 0.0,
            "verdict": "likely clean",
        }

    if defense == "strip":
        return {
            "defense": "strip",
            "entropies": [1.0, 1.1],
            "statistics": {
                "entropy_mean": 1.05,
                "entropy_min": 1.0,
                "entropy_max": 1.1,
                "entropy_std": 0.05,
            },
            "parameters": {
                "num_bases": 2,
                "num_perturbations": 2,
                "seed": None,
            },
            "thresholds": {
                "entropy_mean_threshold": 0.2,
            },
            "verdict": "likely clean",
        }

    if defense == "aeva":
        return {
            "defense": "aeva",
            "verdict": "likely clean",
            "anomaly_index": 0.0,
            "thresholds": {
                "anomaly_index_threshold": 4.0,
            },
            "parameters": {
                "samples_per_class": 1,
                "hsja_iterations": 1,
            },
        }

    if defense == "freeeagle":
        return {
            "defense": "freeeagle",
            "anomaly_metric": 0.0,
            "anomaly_matrix": [[0.0, 0.0], [0.0, 0.0]],
            "tendency_per_target": [0.0, 0.0],
            "verdict": "likely clean",
            "thresholds": {
                "anomaly_metric_threshold": 2.0,
            },
            "parameters": {
                "num_classes": 2,
                "inspect_layer_position": 2,
                "optimize_steps": 1,
                "input_shape": [3, 32, 32],
            },
            "dataset": "cifar10",
        }

    raise AssertionError(f"Unknown test defense: {defense}")


@pytest.fixture
def cli_mocks(monkeypatch):
    """
    Patch the expensive pieces used by CLI detect.

    The CLI still runs its argument parsing, provider checks, defense selection,
    report building, output writing, and routing logic.
    """

    cfg = get_preprocess_config("cifar10")
    model = DummyModel()

    monkeypatch.setattr(
        "mithridatium.cli.utils.get_preprocess_config",
        lambda _data: cfg,
    )
    monkeypatch.setattr(
        "mithridatium.cli.loader.detect_and_build",
        lambda *_args, **_kwargs: (model, None),
    )
    monkeypatch.setattr(
        "mithridatium.cli.loader.build_huggingface_model",
        lambda *_args, **_kwargs: (model, None),
    )
    monkeypatch.setattr(
        "mithridatium.cli.loader.validate_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mithridatium.cli.validate_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mithridatium.cli.loader.ensure_defense_compatibility",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "mithridatium.cli.utils.dataloader_for",
        lambda *_args, **_kwargs: (None, cfg),
    )
    monkeypatch.setattr(
        "mithridatium.cli.utils.dataloader_for_config",
        lambda *_args, **_kwargs: (None, cfg),
    )
    monkeypatch.setattr(
        "mithridatium.cli.get_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )

    # Keep schema validation out of this file. Schema-specific behavior belongs
    # in report tests.
    monkeypatch.setattr(
        "mithridatium.cli.rpt.validate_report_data",
        lambda *_args, **_kwargs: None,
    )

    return {"cfg": cfg, "model": model}


def test_cli_version_flag():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip()


def test_cli_defenses_lists_current_defenses():
    result = runner.invoke(app, ["defenses"])

    assert result.exit_code == 0

    stdout = result.stdout.lower()

    assert "mmbd" in stdout
    assert "strip" in stdout
    assert "aeva" in stdout
    assert "freeeagle" in stdout

    # Spectral was removed and should not come back accidentally.
    assert "spectral" not in stdout


def test_detect_rejects_unsupported_provider():
    result = runner.invoke(
        app,
        [
            "detect",
            "--provider",
            "bad-provider",
            "--defense",
            "mmbd",
        ],
    )

    assert result.exit_code == EXIT_USAGE_ERROR

    output = (result.stdout or "") + (result.stderr or "")
    assert "unsupported --provider" in output.lower()


def test_detect_rejects_missing_local_model(tmp_path):
    missing_model = tmp_path / "missing.pth"

    result = runner.invoke(
        app,
        [
            "detect",
            "--provider",
            "torchvision",
            "--model",
            str(missing_model),
            "--defense",
            "mmbd",
        ],
    )

    assert result.exit_code == EXIT_NO_INPUT

    output = (result.stdout or "") + (result.stderr or "")
    assert "model path not found" in output.lower()


def test_detect_rejects_unsupported_defense(tmp_path):
    model_path = tmp_path / "fake.pth"
    model_path.write_bytes(b"ok")

    result = runner.invoke(
        app,
        [
            "detect",
            "--provider",
            "torchvision",
            "--model",
            str(model_path),
            "--defense",
            "spectral",
            "--data",
            "cifar10",
            "--out",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == EXIT_USAGE_ERROR

    output = (result.stdout or "") + (result.stderr or "")
    assert "unsupported --defense" in output.lower()
    assert "spectral" in output.lower()


@pytest.mark.parametrize("defense", ["mmbd", "strip", "aeva", "freeeagle"])
def test_detect_routes_torchvision_defenses(tmp_path, monkeypatch, cli_mocks, defense):
    model_path = tmp_path / "fake.pth"
    model_path.write_bytes(b"ok")

    out_path = tmp_path / f"{defense}.json"

    called = {}

    def fake_runner(*_args, **_kwargs):
        called["defense"] = defense
        return _valid_result(defense)

    if defense == "mmbd":
        monkeypatch.setattr("mithridatium.cli.run_mmbd", fake_runner)
    elif defense == "strip":
        monkeypatch.setattr("mithridatium.cli.strip_scores", fake_runner)
    elif defense == "aeva":
        monkeypatch.setattr("mithridatium.cli.run_aeva", fake_runner)
    elif defense == "freeeagle":
        monkeypatch.setattr("mithridatium.cli.run_freeeagle", fake_runner)

    result = runner.invoke(
        app,
        [
            "detect",
            "--provider",
            "torchvision",
            "--model",
            str(model_path),
            "--defense",
            defense,
            "--data",
            "cifar10",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    assert called["defense"] == defense
    assert out_path.exists()


@pytest.mark.parametrize("defense", ["mmbd", "strip", "aeva"])
def test_detect_routes_huggingface_logits_only_defenses(tmp_path, monkeypatch, cli_mocks, defense):
    """
    Hugging Face routing should work for defenses that only need logits/model calls.

    FreeEagle is intentionally excluded here because it currently requires
    stable internal feature access.
    """

    out_path = tmp_path / f"hf_{defense}.json"

    called = {}

    def fake_runner(*_args, **_kwargs):
        called["defense"] = defense
        return _valid_result(defense)

    if defense == "mmbd":
        monkeypatch.setattr("mithridatium.cli.run_mmbd", fake_runner)
    elif defense == "strip":
        monkeypatch.setattr("mithridatium.cli.strip_scores", fake_runner)
    elif defense == "aeva":
        monkeypatch.setattr("mithridatium.cli.run_aeva", fake_runner)

    result = runner.invoke(
        app,
        [
            "detect",
            "--provider",
            "huggingface",
            "--hf-model-id",
            "fake/resnet",
            "--defense",
            defense,
            "--data",
            "cifar10_for_imagenet",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    assert called["defense"] == defense
    assert out_path.exists()