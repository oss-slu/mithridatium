# tests/test_cli_v2.py
import json
from pathlib import Path
from typer.testing import CliRunner

from mithridatium.cli import (
    app,
    VERSION,
    EXIT_NO_INPUT,
    EXIT_IO_ERROR,
    EXIT_USAGE_ERROR,
    EXIT_CANT_CREATE,
)

from mithridatium import report as rpt

runner = CliRunner()


def _write_model(tmp_path: Path) -> Path:
    """Create a tiny dummy model file that is readable."""
    model = tmp_path / "fake.pth"
    model.write_bytes(b"ok")
    return model


def test_version_flag():
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert VERSION.strip() in res.stdout


def test_defenses_lists_spectral_and_mmbd():
    res = runner.invoke(app, ["defenses"])
    assert res.exit_code == 0
    # order not guaranteed; check both are present
    assert "spectral" in res.stdout
    assert "mmbd" in res.stdout

def test_detect_spectral_stdout(tmp_path):
    model = (tmp_path / "fake.pth"); model.write_bytes(b"ok")
    res = runner.invoke(app, ["detect", "-m", str(model), "-D", "spectral", "-d", "cifar10", "-o", "-"])
    assert res.exit_code == 0
    assert '"results"' in res.stdout
    assert '"top_eigenvalue"' in res.stdout
    assert "defense=spectral" in res.stdout or '"defense": "spectral"' in res.stdout

def test_detect_stdout_json_then_summary(tmp_path):
    model = _write_model(tmp_path)
    res = runner.invoke(
        app,
        ["detect", "-m", str(model), "-D", "mmbd", "-d", "cifar10", "-o", "-"],
    )
    assert res.exit_code == 0
    # JSON bits
    assert '"mithridatium_version"' in res.stdout
    assert '"defense": "mmbd"' in res.stdout
    assert '"dataset": "cifar10"' in res.stdout
    assert '"results"' in res.stdout
    assert '"suspected_backdoor"' in res.stdout
    # summary bits
    assert "defense=mmbd" in res.stdout
    assert "dataset=cifar10" in res.stdout


def test_detect_to_file_json_schema(tmp_path):
    model = _write_model(tmp_path)
    out = tmp_path / "report.json"
    res = runner.invoke(
        app,
        ["detect", "-m", str(model), "-D", "mmbd", "-d", "cifar10", "-o", str(out)],
    )
    assert res.exit_code == 0
    assert out.exists()
    rep = json.loads(out.read_text(encoding="utf-8"))
    # top-level keys
    for k in ("mithridatium_version", "model_path", "defense", "dataset", "results"):
        assert k in rep
    assert rep["defense"] == "mmbd"
    assert rep["dataset"] == "cifar10"
    # results keys + types
    r = rep["results"]
    assert isinstance(r["suspected_backdoor"], bool)
    assert isinstance(r["num_flagged"], int)
    assert isinstance(r["top_eigenvalue"], (int, float))


def test_missing_model_errors_with_code(tmp_path):
    missing = tmp_path / "nope.pth"
    out = tmp_path / "r.json"
    res = runner.invoke(
        app, ["detect", "-m", str(missing), "-D", "mmbd", "-o", str(out)]
    )
    assert res.exit_code == EXIT_NO_INPUT
    assert "model path not found" in res.stdout


def test_unreadable_model_errors_with_code(tmp_path, monkeypatch):
    model = _write_model(tmp_path)

    # Patch Path.open to raise OSError when opening this file in 'rb'
    from pathlib import Path as _P
    _orig_open = _P.open

    def bad_open(self, mode="r", *args, **kwargs):
        if self == model and "rb" in mode:
            raise OSError("permission denied")
        return _orig_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(_P, "open", bad_open)

    res = runner.invoke(
        app, ["detect", "-m", str(model), "-D", "mmbd", "-o", str(tmp_path / "r.json")]
    )
    assert res.exit_code == EXIT_IO_ERROR
    assert "could not be opened" in res.stdout
    assert "permission denied" in res.stdout


def test_unsupported_defense(tmp_path):
    model = _write_model(tmp_path)
    res = runner.invoke(
        app, ["detect", "-m", str(model), "-D", "not_a_defense", "-o", str(tmp_path / "r.json")]
    )
    assert res.exit_code == EXIT_USAGE_ERROR
    assert "unsupported --defense" in res.stdout
    # should list supported defenses
    assert "spectral" in res.stdout and "mmbd" in res.stdout


def test_force_overwrite(tmp_path):
    model = _write_model(tmp_path)
    out = tmp_path / "r.json"

    # First write
    res1 = runner.invoke(app, ["detect", "-m", str(model), "-D", "mmbd", "-o", str(out)])
    assert res1.exit_code == 0 and out.exists()

    # Overwrite should fail without --force
    res2 = runner.invoke(app, ["detect", "-m", str(model), "-D", "mmbd", "-o", str(out)])
    assert res2.exit_code == EXIT_CANT_CREATE
    assert "already exists" in res2.stdout

    # Overwrite with --force should succeed
    res3 = runner.invoke(
        app, ["detect", "-m", str(model), "-D", "mmbd", "-o", str(out), "--force"]
    )
    assert res3.exit_code == 0


def test_build_report_schema_helper():
    res = {"suspected_backdoor": True, "num_flagged": 500, "top_eigenvalue": 42.3}
    rep = rpt.build_report("models/resnet18_bd.pth", "mmbd", "cifar10", "0.1.1", res)
    for k in ("mithridatium_version", "model_path", "defense", "dataset", "results"):
        assert k in rep
    r = rep["results"]
    assert isinstance(r["suspected_backdoor"], bool)
    assert isinstance(r["num_flagged"], int)
    assert isinstance(r["top_eigenvalue"], (int, float))
