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

def test_cli_repair_requires_target_class():
    result = runner.invoke(app, ["repair","--method","lmr"])

    assert result.exit_code != 0
    assert "lmr-target-class" in result.stderr.lower()

def test_cli_repair_requires_prune_ratio():
    result = runner.invoke(app,["repair","--method","lmr","--lmr-target-class","0"])

    assert result.exit_code != 0
    assert "lmr-prune-ratio" in result.stderr.lower()

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


def test_cli_detect_scaleup_writes_schema_valid_report(tmp_path, monkeypatch):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mithridatium import cli
    from mithridatium.utils import get_preprocess_config

    fake_model = tmp_path / "fake.pth"
    fake_model.write_bytes(b"not a real checkpoint")
    out_path = tmp_path / "detect_scaleup.json"

    cfg = get_preprocess_config("cifar10")
    data = TensorDataset(torch.randn(32, 3, 32, 32), torch.zeros(32, dtype=torch.long))
    toy = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))

    monkeypatch.setattr(cli.loader, "detect_and_build", lambda *a, **k: (toy, None))
    monkeypatch.setattr(cli.utils, "dataloader_for", lambda *a, **k: (DataLoader(data, batch_size=16), cfg))

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
    assert results["num_inputs"] == 32
    assert len(results["per_sample"]) == 32
    assert results["parameters"]["scales"] == [3.0, 5.0, 7.0, 9.0, 11.0]


def _patch_detect_inputs(monkeypatch, toy, data):
    from torch.utils.data import DataLoader

    from mithridatium import cli
    from mithridatium.utils import get_preprocess_config

    cfg = get_preprocess_config("cifar10")
    monkeypatch.setattr(cli.loader, "detect_and_build", lambda *a, **k: (toy, None))
    monkeypatch.setattr(cli.utils, "dataloader_for", lambda *a, **k: (DataLoader(data, batch_size=16), cfg))


def test_cli_detect_ibdpsc_writes_schema_valid_report(tmp_path, monkeypatch):
    import torch
    from torch.utils.data import TensorDataset

    fake_model = tmp_path / "fake.pth"
    fake_model.write_bytes(b"not a real checkpoint")
    out_path = tmp_path / "detect_ibdpsc.json"

    nn = torch.nn
    toy = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 32 * 32, 10)).eval()
    x = torch.randn(32, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        wrong = (toy(x).argmax(1) + 1) % 10   # always-wrong clean labels, so layer selection stops at k = 1
    _patch_detect_inputs(monkeypatch, toy, TensorDataset(x, wrong))

    result = runner.invoke(
        app,
        [
            "detect",
            "--model",
            str(fake_model),
            "--method",
            "ibdpsc",
            "--data",
            "cifar10",
            "--ibdpsc-num-samples",
            "8",
            "--ibdpsc-clean-samples",
            "16",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    rpt.validate_report_data(payload)

    results = payload["results"]
    assert results["mode"] == "input-level"
    assert results["method"] == "ibdpsc"
    assert results["num_inputs"] == 8
    assert len(results["per_sample"]) == 8
    assert results["parameters"]["scale"] == 1.5
    assert results["parameters"]["start_layer"] == 1


def test_cli_detect_ibdpsc_rejects_model_without_batchnorm(tmp_path, monkeypatch):
    import torch
    from torch.utils.data import TensorDataset

    fake_model = tmp_path / "fake.pth"
    fake_model.write_bytes(b"not a real checkpoint")

    toy = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    _patch_detect_inputs(monkeypatch, toy, TensorDataset(torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long)))

    result = runner.invoke(
        app,
        ["detect", "--model", str(fake_model), "--method", "ibdpsc", "--out", str(tmp_path / "r.json")],
    )

    assert result.exit_code == EXIT_USAGE_ERROR
    assert "batchnorm" in result.stderr.lower()



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


def test_cli_repair_clp_passes_arguments(tmp_path, monkeypatch):
    """CLP should be selected and receive the requested threshold and paths."""
    model = tmp_path / "model.pth"
    model.write_bytes(b"placeholder")
    out = tmp_path / "repaired.pth"
    report = tmp_path / "clp.json"
    calls = []

    monkeypatch.setattr(
        "mithridatium.cli.repair_clp", lambda **kwargs: calls.append(kwargs)
    )
    result = runner.invoke(app, [
        "repair", "--method", "clp", "--model", str(model),
        "--clp-threshold-mult", "1.5", "--out", str(out),
        "--report", str(report),
    ])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert calls == [{
        "model_path": str(model),
        "out": str(out),
        "report": str(report),
        "threshold_mult": 1.5,
        "version": VERSION,
    }]


def test_cli_repair_clp_overwrite_requires_force(tmp_path, monkeypatch):
    """An existing output is protected unless --force is provided."""
    model = tmp_path / "model.pth"
    model.write_bytes(b"placeholder")
    out = tmp_path / "repaired.pth"
    out.write_bytes(b"original")
    report = tmp_path / "clp.json"
    report.write_text("original", encoding="utf-8")
    calls = []

    def fake_repair(**kwargs):
        calls.append(kwargs)
        out.write_bytes(b"new")
        report.write_text("new", encoding="utf-8")

    monkeypatch.setattr("mithridatium.cli.repair_clp", fake_repair)
    args = [
        "repair", "--method", "clp", "--model", str(model),
        "--out", str(out), "--report", str(report),
    ]

    refused = runner.invoke(app, args)
    assert refused.exit_code == EXIT_CANT_CREATE
    assert calls == []
    assert out.read_bytes() == b"original"
    assert report.read_text(encoding="utf-8") == "original"

    allowed = runner.invoke(app, [*args, "--force"])
    assert allowed.exit_code == 0, allowed.stdout + allowed.stderr
    assert len(calls) == 1
    assert out.read_bytes() == b"new"
    assert report.read_text(encoding="utf-8") == "new"
def test_cli_detect_ted_writes_schema_valid_report(tmp_path, monkeypatch):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mithridatium import cli
    from mithridatium.utils import get_preprocess_config

    fake_model = tmp_path / "fake.pth"
    fake_model.write_bytes(b"not a real checkpoint")
    out_path = tmp_path / "detect_ted.json"

    toy = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Flatten(), torch.nn.Linear(4 * 8 * 8, 3),
    ).eval()
    x = torch.randn(150, 3, 8, 8, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        y = toy(x).argmax(1)
        # Upstream's TED code assumes every predicted class has at least two
        # reference samples, so drop classes the untrained toy predicts only once.
        keep = torch.bincount(y, minlength=3)[y] >= 2
        x, y = x[keep], y[keep]

    data = TensorDataset(x, y)
    cfg = get_preprocess_config("cifar10")

    monkeypatch.setattr(cli.loader, "detect_and_build", lambda *a, **k: (toy, None))
    monkeypatch.setattr(cli.utils, "dataloader_for", lambda *a, **k: (DataLoader(data, batch_size=16), cfg))

    result = runner.invoke(app, [
        "detect", "--model", str(fake_model), "--method", "ted", "--data", "cifar10",
        "--ted-num-samples", "10", "--ted-reference-size", "150",
        "--ted-contamination", "0.1", "--out", str(out_path),
    ])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    rpt.validate_report_data(payload)
    assert payload["results"]["method"] == "ted"
    assert payload["results"]["num_inputs"] == 10
