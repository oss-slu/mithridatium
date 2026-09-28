"""
Smoke tests for the Mithridatium CLI.

These tests verify that the Typer CLI boots and exposes the expected commands.
They do not run full defenses or require datasets/checkpoints.
"""

from __future__ import annotations

import json

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from mithridatium import report as rpt
from mithridatium.cli import EXIT_CANT_CREATE, EXIT_NO_INPUT, EXIT_USAGE_ERROR, app, VERSION


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
    assert "audit" in result.stdout.lower()
    assert "detect" in result.stdout.lower()
    assert "defenses" in result.stdout.lower()


def test_cli_audit_help_loads():
    result = runner.invoke(app, ["audit", "--help"])

    assert result.exit_code == 0


def test_cli_audit_declares_expected_options():
    # Rich highlights option names per-token, so the rendered help can split
    # "--model" with escape codes on a color-capable terminal. Assert the
    # declared options instead of substrings of the rendered panel.
    audit_command = get_command(app).commands["audit"]
    declared = {opt for param in audit_command.params for opt in param.opts}

    assert {"--model", "--data", "--defense", "--provider", "--out"} <= declared


def test_cli_repair_help_loads():
    result = runner.invoke(app, ["repair", "--help"])

    assert result.exit_code == 0
    assert "method" in result.stdout.lower()
    assert "lmr" in result.stdout.lower()


def test_cli_repair_rejects_unknown_method():
    result = runner.invoke(app, ["repair", "--method", "unsupported_method"])

    assert result.exit_code != 0
    assert "unsupported repair method" in result.stderr.lower()


def test_cli_detect_help_loads():
    result = runner.invoke(app, ["detect", "--help"])

    assert result.exit_code == 0
    assert "scaleup" in result.stdout.lower()
    assert "method" in result.stdout.lower()


def test_cli_detect_declares_expected_options():
    detect_command = get_command(app).commands["detect"]
    declared = {opt for param in detect_command.params for opt in param.opts}

    assert {"--model", "--data", "--method", "--out", "--scaleup-num-samples"} <= declared


def test_cli_detect_rejects_missing_model():
    result = runner.invoke(
        app,
        [
            "detect",
            "--model",
            "models/does_not_exist.pth",
            "--method",
            "scaleup",
            "--out",
            "reports/detect.json",
        ],
    )

    assert result.exit_code == EXIT_NO_INPUT
    assert "model path not found" in result.stderr.lower()


def test_cli_detect_rejects_unknown_method(tmp_path):
    fake_model = tmp_path / "fake.pth"
    fake_model.write_bytes(b"not a real checkpoint")

    result = runner.invoke(
        app,
        [
            "detect",
            "--model",
            str(fake_model),
            "--method",
            "spectral",
            "--out",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "unsupported" in result.stderr.lower()
    assert "spectral" in result.stderr.lower()


def test_cli_detect_scaleup_stub_writes_schema_valid_report(tmp_path):
    fake_model = tmp_path / "fake.pth"
    fake_model.write_bytes(b"not a real checkpoint")
    out_path = tmp_path / "detect_scaleup.json"

    result = runner.invoke(
        app,
        [
            "detect",
            "--model",
            str(fake_model),
            "--method",
            "scaleup",
            "--data",
            "cifar10",
            "--scaleup-num-samples",
            "32",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    rpt.validate_report_data(payload)

    results = payload["results"]
    assert results["mode"] == "input-level"
    assert results["method"] == "scaleup"
    assert results["status"] == "stub_complete"
    assert results["num_inputs"] == 32
    assert results["parameters"]["scales"]


def test_cli_detect_refuses_overwrite_without_force(tmp_path):
    fake_model = tmp_path / "fake.pth"
    fake_model.write_bytes(b"ok")
    out_path = tmp_path / "detect.json"
    out_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "detect",
            "--model",
            str(fake_model),
            "--method",
            "scaleup",
            "--scaleup-num-samples",
            "1",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == EXIT_CANT_CREATE
    assert "already exists" in (result.stdout + result.stderr).lower()


def test_cli_audit_rejects_unknown_defense_before_running_model(tmp_path):
    fake_model = tmp_path / "fake.pth"
    fake_model.write_bytes(b"not a real checkpoint")

    result = runner.invoke(
        app,
        [
            "audit",
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