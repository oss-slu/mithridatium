from __future__ import annotations
from typing import Any
from mithridatium.cli import repair
import json
from pathlib import Path

from mithridatium.mcp import mcp
from mithridatium.mcp.audit_mcp import _catch_output

# repair parameters
def _repair_kwargs(
    #general parameters
    method: str,
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    out: str | None = None,
    force: bool = False,

    #lmr parameters
    clean_samples: int = 500,
    seed: int | None = None,
    lmr_target_class: int | None = None,
    lmr_prune_ratio: float | None = None,

    #clp parameters
    clp_threshold_mult: float = 3.0,

    # anp parameters
    anp_eps: float = 0.4,
    anp_alpha: float = 0.2,
    anp_steps: int = 10,
    anp_prune_threshold: float = 0.2,
) -> dict[str, Any]:
    stem = Path(model).stem
    return {
        "method": method,
        "model": model,
        "data": data,
        "out": Path(out) if out else Path("models") / f"{stem}_{method}_repaired.pth",
        "report": Path("reports") / f"{stem}_{method}_repair.json",
        "force": force,
        #lmr parameters
        "clean_samples": clean_samples,
        "seed": seed,
        "lmr_target_class": lmr_target_class,
        "lmr_prune_ratio": lmr_prune_ratio,
        #clp parameters
        "clp_threshold_mult": clp_threshold_mult,
        # anp parameters
        "anp_eps": anp_eps,
        "anp_alpha": anp_alpha,
        "anp_steps": anp_steps,
        "anp_prune_threshold": anp_prune_threshold,
    }

def _run_repair(repair_kwargs: dict[str, Any]) -> dict[str, Any]:
    out, report = repair_kwargs["out"], repair_kwargs["report"]
    # never let an agent overwrite the checkpoint it was asked to repair
    if out.resolve() == Path(repair_kwargs["model"]).resolve():
        raise ValueError(f"out must differ from the input model: {out}")
    report.parent.mkdir(exist_ok=True)
    with _catch_output():
        repair(**repair_kwargs)
    return {
        "checkpoint": str(out),
        "report_path": str(report),
        "report": json.loads(report.read_text()),
    }

@mcp.tool(run_in_thread=False)
def run_lmr(
    lmr_target_class: int,
    lmr_prune_ratio: float,
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    out: str | None = None,
    force: bool = False,
    clean_samples: int = 500,
    seed: int | None = None,
):
    """Run LMR repair, save the repaired checkpoint, and return its path and the mithridatium repair report."""
    return _run_repair(_repair_kwargs(
        "lmr",
        model=model,
        data=data,
        out=out,
        force=force,
        clean_samples=clean_samples,
        seed=seed,
        lmr_target_class=lmr_target_class,
        lmr_prune_ratio=lmr_prune_ratio,
    ))

@mcp.tool(run_in_thread=False)
def run_clp(
    model: str = "models/resnet18.pth",
    out: str | None = None,
    force: bool = False,
    clp_threshold_mult: float = 3.0,
):
    """Run data-free CLP repair, save the repaired checkpoint, and return its path and the mithridatium repair report."""
    return _run_repair(_repair_kwargs(
        "clp",
        model=model,
        out=out,
        force=force,
        clp_threshold_mult=clp_threshold_mult,
    ))

@mcp.tool(run_in_thread=False)
def run_anp(
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    out: str | None = None,
    force: bool = False,
    clean_samples: int = 500,
    seed: int | None = None,
    anp_eps: float = 0.4,
    anp_alpha: float = 0.2,
    anp_steps: int = 10,
    anp_prune_threshold: float = 0.2,
):
    """Run ANP repair and return the repaired checkpoint and report."""
    return _run_repair(_repair_kwargs(
        "anp",
        model=model,
        data=data,
        out=out,
        force=force,
        clean_samples=clean_samples,
        seed=seed,
        anp_eps=anp_eps,
        anp_alpha=anp_alpha,
        anp_steps=anp_steps,
        anp_prune_threshold=anp_prune_threshold,
    ))
