"""IBD-PSC unit tests. Tiny models and synthetic data, no checkpoint."""

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mithridatium import report as rpt
from mithridatium.detect import ibdpsc as ibdpsc_mod
from mithridatium.detect.ibdpsc import amplify, bn_layers, ibdpsc, psc_scores, select_start_layer

pytestmark = pytest.mark.unit


class Flip(nn.Module):
    """One BN layer. Near-zero inputs score 1.0 < 1.25 and predict class 1;
    amplifying the BN by 1.5 lifts them to 1.5 > 1.25, so they flip to class 0."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        nn.init.ones_(self.bn.bias)

    def forward(self, x):
        z = self.bn(x[:, :1]).mean((1, 2, 3))
        return torch.stack([z, torch.full_like(z, 1.25)], 1)


class TinyBN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 8 * 8, num_classes)
        for bn in (self.bn1, self.bn2):   # non-trivial gamma and beta, so scaling beta matters too
            nn.init.normal_(bn.weight)
            nn.init.normal_(bn.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.fc(x.flatten(1))


class NoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 3)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _flip_data(n=16):
    """Inputs Flip predicts as class 1, labelled 1: 0% clean error until the BN is amplified."""
    x = 0.01 * torch.randn(n, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    return TensorDataset(x, torch.ones(n, dtype=torch.long))


def _random_data(n=16):
    x = torch.randn(n, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    return TensorDataset(x, torch.zeros(n, dtype=torch.long))


def test_original_model_is_not_modified():
    model = Flip().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    r = ibdpsc(model, _flip_data(), _flip_data())

    assert r["parameters"]["start_layer"] == 1    # layer selection really ran
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert model.training                         # handed back in the mode it came in


def test_psc_scores_are_in_unit_interval_and_report_validates(monkeypatch):
    monkeypatch.setattr(ibdpsc_mod, "select_start_layer", lambda *a, **k: 1)
    torch.manual_seed(0)
    model = TinyBN().eval()

    r = ibdpsc(model, _random_data(), _random_data())

    assert all(0.0 <= s["score"] <= 1.0 for s in r["per_sample"])
    assert r["parameters"]["layers_per_model"] == [1, 2, 2, 2, 2]   # paper's k..k+n-1, capped at 2 BN layers
    assert (r["mode"], r["method"]) == ("input-level", "ibdpsc")
    assert r["num_inputs"] == len(r["per_sample"]) == 16
    assert [s["index"] for s in r["per_sample"]] == list(range(16))
    rpt.validate_report_data(rpt.build_report("m.pth", "ibdpsc", "cifar10", "0.0.0", r))


@pytest.mark.parametrize("model", [
    NoBN(),
    nn.Sequential(nn.BatchNorm2d(3, affine=False), nn.Flatten(), nn.Linear(3 * 8 * 8, 3)),   # BN without gamma/beta
])
def test_model_without_batchnorm_raises(model):
    model.eval()
    data = _random_data(4)

    assert bn_layers(model) == []
    with pytest.raises(ValueError, match="BatchNorm"):
        ibdpsc(model, data, data)


def test_threshold_is_strict_per_sec_4_4(monkeypatch):
    """Paper flags when PSC(x) > T, so PSC == T is not flagged."""
    fake = torch.tensor([0.95, 0.9, 0.5, 0.2, 0.1])
    monkeypatch.setattr(ibdpsc_mod, "select_start_layer", lambda *a, **k: 1)
    monkeypatch.setattr(ibdpsc_mod, "psc_scores", lambda *a, **k: (fake, torch.tensor([0, 1, 2, 0, 1])))

    r = ibdpsc(TinyBN().eval(), _random_data(5), _random_data(5), threshold=0.9)

    assert [s["flagged"] for s in r["per_sample"]] == [True, False, False, False, False]
    assert r["verdict"] == "likely backdoored"   # 1/5 flagged sits exactly on the 20% cutoff, which counts


def test_psc_matches_manually_scaled_copies():
    torch.manual_seed(0)
    model = TinyBN().eval()
    x = torch.randn(8, 3, 8, 8)

    scores, preds = psc_scores(model, bn_layers(model), TensorDataset(x, torch.zeros(8)), [1, 2], 1.5, "cpu")

    with torch.no_grad():
        original = model(x).argmax(1)   # computed here, not taken from psc_scores, so the check is independent
    assert torch.equal(preds, original)
    expected = torch.zeros(8)
    for k in (1, 2):
        scaled = copy.deepcopy(model)
        for bn in [scaled.bn2, scaled.bn1][:k]:   # the last k BN layers
            bn.weight.data *= 1.5
            bn.bias.data *= 1.5
        with torch.no_grad():
            expected += scaled(x).softmax(1).gather(1, original[:, None]).squeeze(1)
    assert torch.allclose(scores, expected / 2)


def test_psc_is_confidence_in_the_unamplified_label():
    """Flip predicts 1 unamplified and 0 amplified: PSC must stay on label 1, softmax([1.5, 1.25])[1]."""
    r = ibdpsc(Flip(), _flip_data(), _flip_data())

    assert {s["predicted_label"] for s in r["per_sample"]} == {1}
    assert all(abs(s["score"] - 0.4378) < 5e-3 for s in r["per_sample"])   # input noise moves it ~0.001; a wrong label gives 0.562


def test_layer_selection_stops_at_first_error_strictly_above_xi(monkeypatch):
    bns = [(f"b{i}", nn.BatchNorm2d(1)) for i in range(4)]
    errors = {0: 0.1, 1: 0.2, 2: 0.6, 3: 0.61, 4: 0.9}   # 0.6 equals xi and does not stop the search
    monkeypatch.setattr(ibdpsc_mod, "clean_error", lambda model, overrides, *a, **k: errors[len(overrides) // 2])

    assert select_start_layer(None, bns, None, 1.5, 0.6, "cpu") == 3

    errors.update({3: 0.5, 4: 0.6})
    with pytest.raises(ValueError, match="no starting layer"):
        select_start_layer(None, bns, None, 1.5, 0.6, "cpu")

    errors[0] = 0.7   # already past xi before amplifying: a broken model or mismatched data
    with pytest.raises(ValueError, match="before any amplification"):
        select_start_layer(None, bns, None, 1.5, 0.6, "cpu")


def test_amplified_layer_counts_stop_at_the_bn_count():
    r = ibdpsc(Flip(), _flip_data(), _flip_data(), num_models=3)

    assert r["parameters"]["layers_per_model"] == [1, 1, 1]
    assert r["parameters"]["bn_layers"] == ["bn"]
    bns = bn_layers(TinyBN())
    assert len(amplify(bns, len(bns) + 1, 1.5)) == 2 * len(bns)   # one past the count must not wrap to the last layer
