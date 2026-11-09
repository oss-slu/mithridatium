from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mithridatium.defenses import aeva


pytestmark = pytest.mark.unit


class _FakeStream:
    def __init__(self, is_tty: bool) -> None:
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


class _ProgressBar:
    def __init__(
        self,
        *,
        total: int,
        desc: str,
        unit: str,
        disable: bool,
        leave: bool,
    ) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.leave = leave
        self.updated = 0

    def __enter__(self) -> _ProgressBar:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def update(self, amount: int = 1) -> None:
        self.updated += amount


def _make_detector(tmp_path: Path) -> aeva.BDDetect:
    detector = aeva.BDDetect.__new__(aeva.BDDetect)
    detector.num_labels = 4
    detector.task = "cifar10"
    detector.clean_accuracy = 0.9
    detector.anomaly_index_threshold = 4.0
    detector.samples_per_class = 1
    detector.hsja_iterations = 1
    detector.hsja_max_num_evals = 10
    detector.hsja_init_num_evals = 5
    detector.hsja_query_batch_size = 8
    detector.device = torch.device("cpu")
    detector.output_dir = tmp_path
    detector.get_vec = Mock()

    scores = np.full((4, 4), np.nan, dtype=np.float32)
    for source in range(4):
        for target in range(4):
            if source != target:
                scores[source, target] = float((source + 1) * (target + 1))
    detector._load_pair_matrices = Mock(return_value=(scores, scores.copy()))
    return detector


def _install_progress_spy(monkeypatch: pytest.MonkeyPatch) -> list[_ProgressBar]:
    progress_bars: list[_ProgressBar] = []

    def fake_tqdm(**kwargs) -> _ProgressBar:
        progress = _ProgressBar(**kwargs)
        progress_bars.append(progress)
        return progress

    monkeypatch.setattr(aeva, "tqdm", fake_tqdm)
    return progress_bars


def test_progress_counts_only_selected_source_target_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress_bars = _install_progress_spy(monkeypatch)
    monkeypatch.setattr(aeva.sys, "stdout", _FakeStream(True))
    monkeypatch.setattr(aeva.sys, "stderr", _FakeStream(True))
    detector = _make_detector(tmp_path)

    report = detector.detect(sp=1, ep=3)

    assert len(progress_bars) == 1
    progress = progress_bars[0]
    assert progress.total == (3 - 1) * (4 - 1) == 6
    assert progress.updated == 6
    assert progress.disable is False
    assert detector.get_vec.call_count == 6
    assert report["parameters"]["sp"] == 1
    assert report["parameters"]["ep"] == 3
    assert report["clean_accuracy"] == 0.9


def test_progress_is_disabled_when_stdout_is_not_a_tty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress_bars = _install_progress_spy(monkeypatch)
    monkeypatch.setattr(aeva.sys, "stdout", _FakeStream(False))
    monkeypatch.setattr(aeva.sys, "stderr", _FakeStream(True))
    detector = _make_detector(tmp_path)

    detector.detect(sp=0, ep=1)

    assert len(progress_bars) == 1
    assert progress_bars[0].disable is True
    assert progress_bars[0].total == 3
