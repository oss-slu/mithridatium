import pytest
import torch
from typer.testing import CliRunner

from mithridatium import report as rpt
from mithridatium.cli import EXIT_IO_ERROR, app


runner = CliRunner()


def _valid_freeeagle_results() -> dict:
    return {
        "defense": "freeeagle",
        "anomaly_metric": 2.5,
        "anomaly_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "tendency_per_target": [1.2, 0.8],
        "verdict": "likely backdoored",
        "thresholds": {"anomaly_metric_threshold": 2.0},
        "parameters": {
            "num_classes": 2,
            "inspect_layer_position": 2,
            "optimize_steps": 5,
            "input_shape": [3, 32, 32],
        },
        "dataset": "cifar10",
    }


def test_freeeagle_report_validates_against_schema():
    report = rpt.build_report(
        model_path="models/resnet18.pth",
        defense="freeeagle",
        dataset="cifar10",
        version="0.1.1",
        results=_valid_freeeagle_results(),
    )

    rpt.validate_report_data(report)


def test_freeeagle_report_missing_metric_fails_schema():
    invalid_results = _valid_freeeagle_results()
    invalid_results.pop("anomaly_metric")

    report = rpt.build_report(
        model_path="models/resnet18.pth",
        defense="freeeagle",
        dataset="cifar10",
        version="0.1.1",
        results=invalid_results,
    )

    with pytest.raises(Exception):
        rpt.validate_report_data(report)


def test_detect_fails_when_freeeagle_results_do_not_match_schema(tmp_path, monkeypatch):
    model_path = tmp_path / "fake.pth"
    model_path.write_bytes(b"ok")

    class DummyCfg:
        def get_num_classes(self):
            return 10

        def get_input_size(self):
            return (3, 32, 32)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x

    monkeypatch.setattr("mithridatium.cli.utils.get_preprocess_config", lambda _data: DummyCfg())
    monkeypatch.setattr("mithridatium.cli.loader.detect_and_build", lambda *_args, **_kwargs: (DummyModel(), None))
    monkeypatch.setattr("mithridatium.cli.loader.validate_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mithridatium.cli.utils.dataloader_for", lambda *_args, **_kwargs: (None, DummyCfg()))
    monkeypatch.setattr("mithridatium.cli.get_device", lambda *_args, **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(
        "mithridatium.cli.run_freeeagle",
        lambda *_args, **_kwargs: {
            "defense": "freeeagle",
            "verdict": "likely clean",
            "thresholds": {"anomaly_metric_threshold": 2.0},
            "parameters": {
                "num_classes": 10,
                "inspect_layer_position": 2,
                "optimize_steps": 10,
                "input_shape": [3, 32, 32],
            },
            "dataset": "cifar10",
        },
    )

    result = runner.invoke(
        app,
        [
            "detect",
            "-m",
            str(model_path),
            "-D",
            "freeeagle",
            "-d",
            "cifar10",
            "-o",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == EXIT_IO_ERROR
    output = (result.stdout or "") + (getattr(result, "stderr", "") or "")
    assert "failed schema validation" in output


def test_detect_passes_freeeagle_cli_overrides_to_config(tmp_path, monkeypatch):
    model_path = tmp_path / "fake.pth"
    model_path.write_bytes(b"ok")

    class DummyCfg:
        def get_num_classes(self):
            return 10

        def get_input_size(self):
            return (3, 32, 32)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x

    monkeypatch.setattr("mithridatium.cli.utils.get_preprocess_config", lambda _data: DummyCfg())
    monkeypatch.setattr("mithridatium.cli.loader.detect_and_build", lambda *_args, **_kwargs: (DummyModel(), None))
    monkeypatch.setattr("mithridatium.cli.loader.validate_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mithridatium.cli.utils.dataloader_for", lambda *_args, **_kwargs: (None, DummyCfg()))
    monkeypatch.setattr("mithridatium.cli.get_device", lambda *_args, **_kwargs: torch.device("cpu"))

    captured = {}

    def _run_freeeagle(_model, config, **_kwargs):
        captured["num_classes"] = getattr(config, "freeeagle_num_classes", None)
        captured["num_dummy"] = getattr(config, "freeeagle_num_dummy", None)
        captured["num_important_neurons"] = getattr(config, "freeeagle_num_important_neurons", None)
        captured["metric"] = getattr(config, "freeeagle_metric", None)
        captured["use_transpose_correction"] = getattr(config, "freeeagle_use_transpose_correction", None)
        captured["bound_on"] = getattr(config, "freeeagle_bound_on", None)
        captured["optimize_steps"] = getattr(config, "freeeagle_optimize_steps", None)
        captured["learning_rate"] = getattr(config, "freeeagle_learning_rate", None)
        captured["weight_decay"] = getattr(config, "freeeagle_weight_decay", None)
        captured["anomaly_threshold"] = getattr(config, "freeeagle_anomaly_threshold", None)
        captured["inspect_layer_position"] = getattr(config, "freeeagle_inspect_layer_position", None)
        return _valid_freeeagle_results()

    monkeypatch.setattr("mithridatium.cli.run_freeeagle", _run_freeeagle)

    result = runner.invoke(
        app,
        [
            "detect",
            "-m",
            str(model_path),
            "-D",
            "freeeagle",
            "-d",
            "cifar10",
            "--freeeagle-num-classes",
            "12",
            "--freeeagle-num-dummy",
            "2",
            "--freeeagle-num-important-neurons",
            "7",
            "--freeeagle-metric",
            "softmax_score",
            "--freeeagle-use-transpose-correction",
            "--freeeagle-no-bound-on",
            "--freeeagle-optimize-steps",
            "20",
            "--freeeagle-learning-rate",
            "0.02",
            "--freeeagle-weight-decay",
            "0.004",
            "--freeeagle-anomaly-threshold",
            "2.7",
            "--freeeagle-inspect-layer-position",
            "3",
            "-o",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == 0
    assert captured["num_classes"] == 12
    assert captured["num_dummy"] == 2
    assert captured["num_important_neurons"] == 7
    assert captured["metric"] == "softmax_score"
    assert captured["use_transpose_correction"] is True
    assert captured["bound_on"] is False
    assert captured["optimize_steps"] == 20
    assert captured["learning_rate"] == 0.02
    assert captured["weight_decay"] == 0.004
    assert captured["anomaly_threshold"] == 2.7
    assert captured["inspect_layer_position"] == 3
