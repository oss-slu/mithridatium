"""SCALE-UP unit tests. Synthetic tensors and a toy model, no checkpoint."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mithridatium import report as rpt
from mithridatium.detect.scaleup import run_scaleup
from mithridatium.utils import get_preprocess_config

pytestmark = pytest.mark.unit


class Toy(torch.nn.Module):
    """cutoff=None gives a constant label (scale-invariant). Otherwise the label
    keys on mean pixel value, so amplification flips it."""

    def __init__(self, cutoff=None):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 10)
        if self.cutoff is None:
            logits[:, 3] = 1.0
        else:
            logits[x.flatten(1).mean(1) > self.cutoff, 1] = 1.0
        return logits


def _loader(n=32):
    torch.manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(n, 3, 32, 32) * 0.5,
                                    torch.zeros(n, dtype=torch.long)), batch_size=16)


def _run(cutoff=None, **kw):
    return run_scaleup(Toy(cutoff), get_preprocess_config("cifar10"),
                       num_samples=32, seed=42, test_loader=_loader(), **kw)


def test_scale_invariant_input_scores_one():
    r = _run()
    assert all(s["score"] == 1.0 and s["flagged"] for s in r["per_sample"])
    assert (r["num_inputs"], r["num_flagged"]) == (32, 32)
    assert r["verdict"] == "likely backdoored"


def test_amplification_actually_changes_the_input():
    r = _run(cutoff=1.0)
    assert all(s["score"] == 0.0 and not s["flagged"] for s in r["per_sample"])
    assert r["verdict"] == "likely clean"


def test_threshold_is_strict_per_eq2():
    """Paper flags when SPC(x) > T, so SPC == T flags nothing."""
    r = _run(threshold=1.0)
    assert all(s["score"] == 1.0 for s in r["per_sample"]) and r["num_flagged"] == 0


def test_report_validates_against_schema():
    rep = rpt.build_report("m.pth", "scaleup", "cifar10", "0.1.1", _run())
    rpt.validate_report_data(rep)
    rep["results"].pop("per_sample")
    with pytest.raises(Exception):
        rpt.validate_report_data(rep)
