import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mithridatium.repair.lmr import run_lmr


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 8),
            nn.ReLU(),
        )

        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


def test_lmr_runs():
    torch.manual_seed(0)

    # 12 fake 2x2 images
    inputs = torch.randn(12, 1, 2, 2)

    # Three fake classes
    labels = torch.tensor([
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
    ])

    clean_loader = DataLoader(
        TensorDataset(inputs, labels),
        batch_size=4,
        shuffle=False,
    )

    model = TinyModel()

    repaired_model, results = run_lmr(
        model,
        clean_loader,
        target_class=2,
        prune_ratio=0.25,
        seed=0,
        repair_steps=2,
        fine_tune_steps=1,
    )

    assert repaired_model is model
    assert results["method"] == "lmr"
    assert results["verdict"] == "repair_applied"
    assert results["pruning"]["columns_pruned"] > 0


def test_lmr_defaults_match():
    # Verify defaults match across function, MCP, and CLI
    import inspect
    from mithridatium.repair.lmr import (
        DEFAULT_FINE_TUNE_STEPS,
        DEFAULT_PRUNE_RATIO,
        DEFAULT_REPAIR_STEPS,
        DEFAULT_SEED,
    )
    from typer.main import get_command
    from mithridatium.cli import app

    # Check function signature defaults
    fn_params = inspect.signature(run_lmr).parameters
    assert fn_params["prune_ratio"].default == DEFAULT_PRUNE_RATIO
    assert fn_params["repair_steps"].default == DEFAULT_REPAIR_STEPS
    assert fn_params["fine_tune_steps"].default == DEFAULT_FINE_TUNE_STEPS
    assert fn_params["seed"].default == DEFAULT_SEED

    # Check MCP tool signature defaults if MCP dependencies are installed
    try:
        from mithridatium.mcp.repair_mcp import run_lmr as mcp_run_lmr
        mcp_params = inspect.signature(mcp_run_lmr).parameters
        assert mcp_params["lmr_prune_ratio"].default == DEFAULT_PRUNE_RATIO
        assert mcp_params["lmr_repair_steps"].default == DEFAULT_REPAIR_STEPS
        assert mcp_params["lmr_fine_tune_steps"].default == DEFAULT_FINE_TUNE_STEPS
        assert mcp_params["seed"].default == DEFAULT_SEED
    except ImportError:
        pass

    # Check CLI option defaults
    repair_cmd = get_command(app).commands["repair"]
    cli_defaults = {p.name: p.default for p in repair_cmd.params}
    assert cli_defaults["lmr_prune_ratio"] == DEFAULT_PRUNE_RATIO
    assert cli_defaults["lmr_repair_steps"] == DEFAULT_REPAIR_STEPS
    assert cli_defaults["lmr_fine_tune_steps"] == DEFAULT_FINE_TUNE_STEPS
    assert cli_defaults["seed"] == DEFAULT_SEED
