"""Fast tests for the single-CSV ANP sweep (no real checkpoints needed)."""

import csv

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts import sweep_anp as sweep


def test_stage_candidates(monkeypatch):
    a = sweep.parse_args(["--stage", "threshold"])
    assert len(sweep.candidates(a)) == 7
    assert all(c[1:] == (0.4, 0.2, 1, 500) for c in sweep.candidates(a))
    a = sweep.parse_args(["--stage", "eps-alpha"])
    assert len(sweep.candidates(a)) == 9
    a = sweep.parse_args(["--stage", "steps-samples"])
    assert len(sweep.candidates(a)) == 6
    a = sweep.parse_args(["--stage", "final"])
    assert sweep.candidates(a) == [(0.2, 0.4, 0.2, 1, 500)]


def test_smoke_does_not_modify_normal_settings():
    quick = sweep.parse_args(["--smoke"])
    normal = sweep.parse_args([])
    assert quick.iterations == 2
    assert quick.output_dir != normal.output_dir
    assert normal.iterations == 2000


def test_single_csv_resume_and_new_stage(tmp_path, monkeypatch):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)
            self.bn = nn.BatchNorm2d(4)
            self.fc = nn.Linear(4 * 4 * 4, 10)

        def forward(self, x):
            return self.fc(self.bn(self.conv(x)).flatten(1))

    torch.manual_seed(44)
    weights = TinyModel().state_dict()

    def load(*args, **kwargs):
        model = TinyModel()
        model.load_state_dict(weights)
        return model, None

    class Config:
        def get_num_classes(self):
            return 10

        def get_normalize(self):
            return False

    x = torch.rand(30, 3, 4, 4)
    y = torch.tensor([i % 10 for i in range(30)])
    dataset = TensorDataset(x, y)
    dl = DataLoader(dataset, batch_size=8)
    monkeypatch.setattr(sweep.loader, "detect_and_build", load)
    monkeypatch.setattr(sweep.utils, "dataloader_for", lambda *args: (dl, Config()))
    mask_calls = []

    def train_masks(model, data, **kwargs):
        mask_calls.append(kwargs)
        return {"bn": torch.tensor([0.01, 0.03, 0.08, 0.4])}

    monkeypatch.setattr(sweep, "train_masks", train_masks)
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    out = tmp_path / "results"
    for key, value in {
        "model": checkpoint, "output_dir": out, "device": "cpu", "iterations": 1,
        "eval_samples": 12, "asr_samples": 12, "clean_samples": 10, "trigger_size": 1,
    }.items():
        monkeypatch.setitem(sweep.CONFIG, key, value)
    monkeypatch.setitem(sweep.SWEEPS, "threshold", [0, 0.05, 0.2])
    sweep.main(["--stage", "threshold"])
    assert len(mask_calls) == 1
    with (out / "all_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert [int(r["neurons_pruned"]) for r in rows] == [0, 2, 3]
    assert all(r["ca_after_pct"] and r["asr_after_pct"] and r["runtime_seconds"] for r in rows)
    assert list(out.glob("*.csv")) == [out / "all_results.csv"]

    # Repeating a stage resumes/skips completed rows; a different stage appends.
    sweep.main(["--stage", "threshold"])
    assert len(mask_calls) == 1
    monkeypatch.setitem(sweep.CONFIG, "threshold", 0.05)
    monkeypatch.setitem(sweep.SWEEPS, "eps", [0.4])
    monkeypatch.setitem(sweep.SWEEPS, "alpha", [0.2])
    sweep.main(["--stage", "eps-alpha"])
    with (out / "all_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {row["stage"] for row in rows} == {"threshold", "eps-alpha"}
    assert len(mask_calls) == 2


def test_reject_bad_samples_and_old_schema(tmp_path, monkeypatch):
    monkeypatch.setitem(sweep.CONFIG, "iterations", 0)
    with pytest.raises(ValueError, match="positive"):
        sweep.main([])
    path = tmp_path / "all_results.csv"
    path.write_text("old_column\n1\n")
    with pytest.raises(ValueError, match="different column"):
        sweep.saved_keys(path)
