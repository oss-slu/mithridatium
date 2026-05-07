"""
Integration tests for FreeEagle detection with real checkpoints.

These tests are optional/slow because they run the actual defense.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mithridatium import report as rpt
from mithridatium.cli import app


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.requires_model,
]

runner = CliRunner()


def _find_checkpoint(models_root: Path, stem: str) -> Path | None:
    """Find a model checkpoint by stem, accepting either .pt or .pth."""
    for suffix in (".pt", ".pth"):
        candidate = models_root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def test_freeeagle_cli_detect_runs_on_poison_checkpoint(tmp_path, models_root):
    checkpoint_path = _find_checkpoint(models_root, "resnet18_poison")

    if checkpoint_path is None:
        pytest.skip(
            "poison checkpoint not available. Expected either "
            f"{models_root / 'resnet18_poison.pt'} or "
            f"{models_root / 'resnet18_poison.pth'}"
        )

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

    assert result.exit_code == 0, (result.stdout or "") + (result.stderr or "")
    assert out_path.exists()

    report_data = json.loads(out_path.read_text(encoding="utf-8"))
    rpt.validate_report_data(report_data)

    assert report_data["defense"] == "freeeagle"
    assert report_data["dataset"] == "cifar10"
    assert report_data["results"]["verdict"] in ("likely clean", "likely backdoored")