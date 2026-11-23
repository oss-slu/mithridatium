"""Fast tests for the single-CSV ANP sweep (no real checkpoints needed)."""

import csv

import pytest
from scripts import sweep_anp as sweep


def test_stage_candidates():
    a = sweep.parse_args(["--stage", "threshold"])
    assert sweep.candidates(a) == [
        (t, a.eps, a.alpha, a.steps, a.clean_samples)
        for t in sweep.SWEEPS["threshold"]
    ]

    a = sweep.parse_args(["--stage", "eps-alpha"])
    assert sweep.candidates(a) == [
        (a.threshold, e, alpha, a.steps, a.clean_samples)
        for e in sweep.SWEEPS["eps"]
        for alpha in sweep.SWEEPS["alpha"]
    ]

    a = sweep.parse_args(["--stage", "steps-samples"])
    assert sweep.candidates(a) == [
        (a.threshold, a.eps, a.alpha, steps, samples)
        for steps in sweep.SWEEPS["steps"]
        for samples in sweep.SWEEPS["clean_samples"]
    ]

    a = sweep.parse_args(["--stage", "threshold-seeds"])
    assert sweep.candidates(a) == [
        (t, a.eps, a.alpha, a.steps, a.clean_samples)
        for t in sweep.SWEEPS["threshold"]
    ]

    a = sweep.parse_args(["--stage", "final"])
    assert sweep.candidates(a) == [
        (a.threshold, a.eps, a.alpha, a.steps, a.clean_samples)
    ]


def test_smoke_does_not_modify_normal_settings():
    quick = sweep.parse_args(["--smoke"])
    normal = sweep.parse_args([])
    assert quick.iterations == 2
    assert quick.output_dir != normal.output_dir
    assert normal.iterations == 2000


def test_single_csv_resume_and_new_stage(tmp_path, monkeypatch):
    """Run the normal sweep through the real repair API (mocked, no training)."""
    from types import SimpleNamespace

    checkpoint = tmp_path / "poison.pth"
    checkpoint.touch()
    calls = []

    def fake_run_anp(**kwargs):
        calls.append(kwargs)
        return {"results": {
            "clean_accuracy_before": 0.8, "clean_accuracy_after": 0.79,
            "asr_before": 0.9, "asr_after": 0.1,
            "total_channels_pruned": 17,
        }}

    monkeypatch.setattr(sweep, "run_anp", fake_run_anp)
    monkeypatch.setattr(sweep.utils, "dataloader_for", lambda *args, **kw:
                        (SimpleNamespace(dataset=list(range(30))), object()))
    monkeypatch.setattr(sweep, "build_triggered_dataset", lambda *a, **kw: list(range(12)))
    out = tmp_path / "results"
    for key, value in {
        "model": checkpoint, "output_dir": out, "device": "cpu",
        "iterations": 1, "eval_samples": 12, "asr_samples": 12,
        "clean_samples": 10, "trigger_size": 1,
    }.items():
        monkeypatch.setitem(sweep.CONFIG, key, value)
    monkeypatch.setitem(sweep.SWEEPS, "threshold", [0, 0.05, 0.2])

    sweep.main(["--stage", "threshold"])
    assert len(calls) == 3
    assert all(call["seed"] == 0 for call in calls)
    assert all(call["clean_subset_seed"] == 1234 for call in calls)
    with (out / "all_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert [float(r["prune_threshold"]) for r in rows] == [0, 0.05, 0.2]
    assert all(r["ca_after_pct"] == "79.0" and r["asr_after_pct"] == "10.0" for r in rows)
    assert all(r["neurons_pruned"] == "17" and r["runtime_seconds"] for r in rows)

    sweep.main(["--stage", "threshold"])
    assert len(calls) == 3  # resume does not rerun repairs
    monkeypatch.setitem(sweep.CONFIG, "threshold", 0.05)
    monkeypatch.setitem(sweep.SWEEPS, "eps", [0.4])
    monkeypatch.setitem(sweep.SWEEPS, "alpha", [0.2])
    sweep.main(["--stage", "eps-alpha"])
    assert len(calls) == 4
    with (out / "all_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {row["stage"] for row in rows} == {"threshold", "eps-alpha"}


def test_subset_variability_uses_run_anp_and_full_test(tmp_path, monkeypatch):
    """Five clean subsets use one optimization seed and a held-out full test set."""
    from types import SimpleNamespace

    poison = tmp_path / "poison.pth"
    clean = tmp_path / "clean.pth"
    poison.touch()
    clean.touch()
    calls = []

    def fake_run_anp(**kwargs):
        calls.append(kwargs)
        return {"results": {
            "clean_accuracy_before": 0.8, "clean_accuracy_after": 0.79,
            "asr_before": 0.9, "asr_after": 0.1,
            "total_channels_pruned": 17,
        }}

    monkeypatch.setattr(sweep, "run_anp", fake_run_anp)
    monkeypatch.setattr(sweep.utils, "dataloader_for", lambda *a, **kw:
                        (SimpleNamespace(dataset=list(range(10000))), object()))
    monkeypatch.setattr(sweep, "build_triggered_dataset", lambda *a, **kw: list(range(9000)))
    monkeypatch.setitem(sweep.CONFIG, "model", poison)
    monkeypatch.setitem(sweep.CONFIG, "clean_model", clean)
    monkeypatch.setitem(sweep.CONFIG, "iterations", 2)
    monkeypatch.setitem(sweep.CONFIG, "device", "cpu")
    original_parse_args = sweep.parse_args

    def temporary_output(argv=None):
        args = original_parse_args(argv)
        args.output_dir = tmp_path / "review"
        return args

    monkeypatch.setattr(sweep, "parse_args", temporary_output)
    args = sweep.parse_args(["--stage", "subset-variability"])
    assert args.eval_samples == args.asr_samples == 10000
    sweep.main(["--stage", "subset-variability"])
    assert len(calls) == 10  # 5 subsets per model
    assert {call["clean_subset_seed"] for call in calls} == set(range(5))
    assert {call["seed"] for call in calls} == {0}
    assert {call["eval_seed"] for call in calls} == {1234}
    assert all(call["eval_samples"] == 10000 for call in calls)
    assert all(call["device"] == "cpu" for call in calls)
    with (tmp_path / "review" / "all_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    assert len({(r["model"], r["clean_subset_seed"]) for r in rows}) == 10
    sweep.main(["--stage", "subset-variability"])
    assert len(calls) == 10  # no duplicate runs


def test_reject_bad_samples_and_old_schema(tmp_path, monkeypatch):
    monkeypatch.setitem(sweep.CONFIG, "iterations", 0)
    with pytest.raises(ValueError, match="positive"):
        sweep.main([])
    path = tmp_path / "all_results.csv"
    path.write_text("old_column\n1\n")
    with pytest.raises(ValueError, match="different column"):
        sweep.saved_keys(path)
