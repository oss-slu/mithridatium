"""Keep ANP's CLI, MCP and Python defaults in sync (issue #184)."""

import ast
import inspect
from pathlib import Path

from mithridatium.repair import anp
from mithridatium.repair import anp_defaults as defaults

ROOT = Path(__file__).resolve().parents[2]

# Python option name, CLI option name, shared constant.
CASES = [
    ("eps", "anp_eps", "ANP_EPS"),
    ("alpha", "anp_alpha", "ANP_ALPHA"),
    ("steps", "anp_steps", "ANP_STEPS"),
    ("prune_threshold", "anp_prune_threshold", "ANP_PRUNE_THRESHOLD"),
    ("clean_samples", "clean_samples", "ANP_CLEAN_SAMPLES"),
    ("iterations", "anp_iterations", "ANP_ITERATIONS"),
    ("clean_subset_seed", "anp_clean_subset_seed", "ANP_CLEAN_SUBSET_SEED"),
]


def _source_defaults(file, function):
    """Return AST defaults without requiring optional MCP dependencies to be installed."""
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    positional = node.args.posonlyargs + node.args.args
    values = {
        argument.arg: default
        for argument, default in zip(positional[-len(node.args.defaults):], node.args.defaults)
    } if node.args.defaults else {}
    values.update({
        argument.arg: default
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
    })
    return values


def _constant_name(node):
    # CLI defaults are declared via typer.Option(CONSTANT, '--flag', ...).
    if isinstance(node, ast.Call):
        node = node.args[0]
    return node.id if isinstance(node, ast.Name) else None


def test_anp_defaults_match_all_entry_points():
    cli = _source_defaults("mithridatium/cli.py", "repair")
    mcp = _source_defaults("mithridatium/mcp/repair_mcp.py", "run_anp")
    mcp_helper = _source_defaults("mithridatium/mcp/repair_mcp.py", "_repair_kwargs")
    python_defaults = inspect.signature(anp.run_anp).parameters

    for python_name, option_name, constant in CASES:
        expected = getattr(defaults, constant)
        assert python_defaults[python_name].default == expected
        assert _constant_name(cli[option_name]) == constant
        assert _constant_name(mcp[option_name]) == constant
        assert _constant_name(mcp_helper[option_name]) == constant


def test_train_masks_uses_shared_training_defaults():
    parameters = inspect.signature(anp.train_masks).parameters
    assert parameters["eps"].default == defaults.ANP_EPS
    assert parameters["alpha"].default == defaults.ANP_ALPHA
    assert parameters["steps"].default == defaults.ANP_STEPS
    assert parameters["iterations"].default == defaults.ANP_ITERATIONS


def test_clean_subset_seed_follows_optimizer_seed_unless_overridden():
    from mithridatium.repair.anp import _resolve_clean_subset_seed

    assert defaults.ANP_CLEAN_SUBSET_SEED is None
    assert _resolve_clean_subset_seed(0, None) == 0
    assert _resolve_clean_subset_seed(1, None) == 1
    assert _resolve_clean_subset_seed(1, 1234) == 1234


def test_run_anp_uses_seeded_clean_subset_by_default(tmp_path, monkeypatch):
    """Exercise the actual backend (without loading checkpoints or training masks)."""
    from types import SimpleNamespace

    import pytest
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset, TensorDataset

    dataset = TensorDataset(torch.zeros(20, 3, 4, 4), torch.zeros(20, dtype=torch.long))
    seen = []

    def record_sample(data, count, generator):
        seen.append(generator.initial_seed())
        return Subset(data, list(range(count)))

    monkeypatch.setattr(anp, "_sample", record_sample)
    monkeypatch.setattr(anp.loader, "detect_and_build", lambda *args, **kw: (nn.Linear(1, 1), None))
    monkeypatch.setattr(anp.utils, "get_preprocess_config", lambda name: SimpleNamespace(get_num_classes=lambda: 10))
    monkeypatch.setattr(anp.utils, "dataloader_for", lambda *args, **kw: (DataLoader(dataset), None))
    monkeypatch.setattr(anp, "evaluate_clean", lambda model, data: 0.75)
    monkeypatch.setattr(anp, "train_masks", lambda *args, **kw: {})
    monkeypatch.setattr(anp, "prune_by_threshold", lambda *args, **kw: [])
    monkeypatch.setattr(anp.rpt, "build_repair_report", lambda model_path, method, data, version, results: results)
    monkeypatch.setattr(anp.rpt, "validate_report_data", lambda report: None)

    for override, expected in [(None, 3), (9, 9)]:
        seen.clear()
        kwargs = dict(model_path=str(tmp_path / "original.pth"),
                      out=str(tmp_path / "repaired.pth"),
                      report=str(tmp_path / "report.json"),
                      clean_samples=5, eval_samples=7, seed=3, eval_seed=1234)
        if override is not None:
            kwargs["clean_subset_seed"] = override
        report = anp.run_anp(**kwargs)
        assert seen == [expected, 1234]
        assert report["parameters"]["clean_subset_seed"] == expected
        assert report["parameters"]["seed"] == 3

    with pytest.raises(ValueError, match="nonnegative"):
        anp.run_anp(model_path=str(tmp_path / "original.pth"),
                    out=str(tmp_path / "repaired.pth"),
                    report=str(tmp_path / "report.json"),
                    clean_subset_seed=-1)
