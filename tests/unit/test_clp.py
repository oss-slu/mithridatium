"""Fast CLP tests: no checkpoint download and no training required."""

import json

import pytest
import torch
import torch.nn as nn

from mithridatium import report as rpt
from mithridatium.repair.clp import _prune_channels, _uclc_score, repair_clp

pytestmark = pytest.mark.unit


class TinyResNetLike(nn.Module):
    """Just enough Conv2d -> BatchNorm2d structure to exercise CLP."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        with torch.no_grad():
            self.conv1.weight.zero_()
            self.conv1.weight[:, 0, 0, 0] = torch.tensor([1.0, 1.0, 1.0, 10.0])

    def forward(self, x):
        return self.bn1(self.conv1(x))


def test_uclc_score_scales_with_conv_weight_and_bn():
    model = TinyResNetLike()
    with torch.no_grad():
        model.bn1.weight.fill_(2.0)
        model.bn1.running_var.fill_(3.0)
    scores = _uclc_score(model.conv1.weight, model.bn1)
    expected = torch.tensor([1., 1., 1., 10.]) * 2.0 / (3.0 + model.bn1.eps) ** 0.5
    torch.testing.assert_close(scores, expected)


def test_prune_only_outlier_channel():
    bn = nn.BatchNorm2d(4)
    with torch.no_grad():
        bn.bias.fill_(0.5)
    pruned = _prune_channels(bn, torch.tensor([1., 1., 1., 10.]), threshold_mult=1.0)
    assert pruned == [3]
    assert bn.weight[3].item() == 0.0
    assert bn.bias[3].item() == 0.0
    torch.testing.assert_close(bn.weight[:3], torch.ones(3))
    torch.testing.assert_close(bn.bias[:3], torch.full((3,), 0.5))


def test_prune_rejects_channel_mismatch():
    with pytest.raises(ValueError, match="score has"):
        _prune_channels(nn.BatchNorm2d(4), torch.tensor([1., 2.]), threshold_mult=1.0)


def test_prune_rejects_non_affine_bn():
    with pytest.raises(ValueError, match="affine"):
        _prune_channels(nn.BatchNorm2d(4, affine=False), torch.ones(4), threshold_mult=1.0)


def test_repair_clp_creates_checkpoint_and_valid_report(tmp_path, monkeypatch):
    model = TinyResNetLike()
    monkeypatch.setattr(
        "mithridatium.repair.clp.loader.detect_and_build",
        lambda *_args, **_kwargs: (model, None),
    )
    out = tmp_path / "subdir" / "repaired.pth"
    report_file = tmp_path / "reports" / "clp.json"
    repair_clp("dummy.pth", str(out), str(report_file), threshold_mult=1.0, version="test")

    assert out.exists()
    saved = torch.load(out, map_location="cpu", weights_only=True)
    assert saved["bn1.weight"][3].item() == 0.0
    assert saved["bn1.bias"][3].item() == 0.0
    torch.testing.assert_close(saved["bn1.weight"][:3], torch.ones(3))

    payload = json.loads(report_file.read_text(encoding="utf-8"))
    rpt.validate_report_data(payload)
    assert payload["timestamp_utc"].endswith("Z")
    assert payload["defense"] == "clp"
    assert payload["results"]["method"] == "clp"
    assert payload["results"]["parameters"]["threshold_mult"] == 1.0
    assert payload["results"]["total_channels_pruned"] == 1
    assert payload["results"]["layer_breakdown"] == [
        {"layer": "conv1", "channels_pruned": 1}
    ]
    assert "pruned 1 channels" in payload["results"]["status"]


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_repair_rejects_invalid_threshold_before_loading(tmp_path, monkeypatch, invalid):
    def should_not_load(*_args, **_kwargs):
        raise AssertionError("loader should not run for invalid threshold")
    monkeypatch.setattr("mithridatium.repair.clp.loader.detect_and_build", should_not_load)
    with pytest.raises(ValueError, match="nonnegative"):
        repair_clp("dummy.pth", str(tmp_path / "out.pth"), str(tmp_path / "r.json"), invalid, version="test")


def test_clp_schema_requires_prune_count():
    payload = {
        "mithridatium_version": "0.1.0",
        "timestamp_utc": "2026-10-11T00:00:00+00:00",
        "model_path": "dummy.pth",
        "defense": "clp",
        "dataset": "n/a (data-free)",
        "results": {
            "method": "clp",
            "parameters": {"threshold_mult": 3.0},
            "status": "pruning attempted",
        },
    }
    with pytest.raises(Exception):
        rpt.validate_report_data(payload)
    payload["results"]["total_channels_pruned"] = 0
    rpt.validate_report_data(payload)
    payload["results"]["total_channels_pruned"] = -1
    with pytest.raises(Exception):
        rpt.validate_report_data(payload)
