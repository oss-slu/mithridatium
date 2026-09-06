"""
Smoke tests for the Mithridatium CLI.

These tests verify that the Typer CLI boots and exposes the expected commands.
They do not run full defenses or require datasets/checkpoints.
"""

from __future__ import annotations

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from mithridatium.cli import app, VERSION


pytestmark = pytest.mark.smoke

runner = CliRunner()


def test_cli_version_flag():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert VERSION in result.stdout


def test_cli_defenses_lists_current_supported_defenses():
    result = runner.invoke(app, ["defenses"])

    assert result.exit_code == 0

    stdout = result.stdout.lower()

    assert "mmbd" in stdout
    assert "strip" in stdout
    assert "aeva" in stdout
    assert "freeeagle" in stdout

    # Spectral was removed; this guards against stale test expectations.
    assert "spectral" not in stdout


def test_cli_help_loads():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "mithridatium" in result.stdout.lower()
    assert "detect" in result.stdout.lower()
    assert "defenses" in result.stdout.lower()


def test_cli_detect_help_loads():
    result = runner.invoke(app, ["detect", "--help"])

    assert result.exit_code == 0


def test_cli_detect_declares_expected_options():
    # Rich highlights option names per-token, so the rendered help can split
    # "--model" with escape codes on a color-capable terminal. Assert the
    # declared options instead of substrings of the rendered panel.
    detect_command = get_command(app).commands["detect"]
    declared = {opt for param in detect_command.params for opt in param.opts}

    assert {"--model", "--data", "--defense", "--provider", "--out"} <= declared


def test_cli_detect_rejects_unknown_defense_before_running_model(tmp_path):
    fake_model = tmp_path / "fake.pth"
    fake_model.write_bytes(b"not a real checkpoint")

    result = runner.invoke(
        app,
        [
            "detect",
            "--model",
            str(fake_model),
            "--defense",
            "spectral",
            "--data",
            "cifar10",
            "--out",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code != 0
    assert "unsupported" in result.stdout.lower() or "unsupported" in result.stderr.lower()
    assert "spectral" in result.stdout.lower() or "spectral" in result.stderr.lower()