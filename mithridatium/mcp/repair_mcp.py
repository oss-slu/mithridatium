from __future__ import annotations
from typing import Any
from mithridatium.cli import repair
from mithridatium.repair.anp_defaults import (
    ANP_ALPHA, ANP_CLEAN_SAMPLES, ANP_CLEAN_SUBSET_SEED, ANP_EPS,
    ANP_ITERATIONS, ANP_PRUNE_THRESHOLD, ANP_STEPS,
)
from mithridatium.repair import (
    DEFAULT_FINE_TUNE_STEPS,
    DEFAULT_PRUNE_RATIO,
    DEFAULT_REPAIR_STEPS,
)
import json
from pathlib import Path

from mithridatium.mcp import mcp
from mithridatium.mcp.audit_mcp import _catch_output
from mithridatium.repair.clp import CLP_DEFAULT_THRESHOLD_MULT

# repair parameters
def _repair_kwargs(
    #general parameters
    method: str,
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    out: str | None = None,
    force: bool = False,

    #lmr parameters
    clean_samples: int = ANP_CLEAN_SAMPLES,
    seed: int | None = None,
    lmr_target_class: int | None = None,
    lmr_prune_ratio: float = DEFAULT_PRUNE_RATIO,
    lmr_repair_steps: int = DEFAULT_REPAIR_STEPS,
    lmr_fine_tune_steps: int = DEFAULT_FINE_TUNE_STEPS,

    #clp parameters
    clp_threshold_mult: float = CLP_DEFAULT_THRESHOLD_MULT,

    # anp parameters
    anp_eps: float = ANP_EPS,
    anp_alpha: float = ANP_ALPHA,
    anp_steps: int = ANP_STEPS,
    anp_prune_threshold: float = ANP_PRUNE_THRESHOLD,
    anp_iterations: int = ANP_ITERATIONS,
    anp_clean_subset_seed: int | None = ANP_CLEAN_SUBSET_SEED,
    anp_prune_fraction: float | None = None,
    anp_attack_target: int | None = None,
    anp_trigger_type: str | None = None,
    anp_trigger_alpha: float | None = None,
    anp_trigger_size: int | None = None,
    anp_trigger_pos: str | None = None,
    anp_asr_samples: int | None = None,
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
        "lmr_repair_steps": lmr_repair_steps,
        "lmr_fine_tune_steps": lmr_fine_tune_steps,
        #clp parameters
        "clp_threshold_mult": clp_threshold_mult,
        # anp parameters
        "anp_eps": anp_eps,
        "anp_alpha": anp_alpha,
        "anp_steps": anp_steps,
        "anp_prune_threshold": anp_prune_threshold,
        "anp_iterations": anp_iterations,
        "anp_clean_subset_seed": anp_clean_subset_seed,
        "anp_prune_fraction": anp_prune_fraction,
        "anp_attack_target": anp_attack_target,
        "anp_trigger_type": anp_trigger_type,
        "anp_trigger_alpha": anp_trigger_alpha,
        "anp_trigger_size": anp_trigger_size,
        "anp_trigger_pos": anp_trigger_pos,
        "anp_asr_samples": anp_asr_samples,
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

# LMR tool
@mcp.tool(run_in_thread=False)
def run_lmr(
    lmr_target_class: int,
    lmr_prune_ratio: float = DEFAULT_PRUNE_RATIO,
    lmr_repair_steps: int = DEFAULT_REPAIR_STEPS,
    lmr_fine_tune_steps: int = DEFAULT_FINE_TUNE_STEPS,
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    out: str | None = None,
    force: bool = False,
    clean_samples: int = 500,
    seed: int = 0,
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
        lmr_repair_steps=lmr_repair_steps,
        lmr_fine_tune_steps=lmr_fine_tune_steps,
    ))

@mcp.tool(run_in_thread=False)
def run_clp(
    model: str = "models/resnet18.pth",
    out: str | None = None,
    force: bool = False,
    clp_threshold_mult: float = CLP_DEFAULT_THRESHOLD_MULT,
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
    clean_samples: int = ANP_CLEAN_SAMPLES,
    seed: int | None = None,
    anp_eps: float = ANP_EPS,
    anp_alpha: float = ANP_ALPHA,
    anp_steps: int = ANP_STEPS,
    anp_prune_threshold: float = ANP_PRUNE_THRESHOLD,
    anp_iterations: int = ANP_ITERATIONS,
    anp_clean_subset_seed: int | None = ANP_CLEAN_SUBSET_SEED,
    anp_prune_fraction: float | None = None,
    anp_attack_target: int | None = None,
    anp_trigger_type: str | None = None,
    anp_trigger_alpha: float | None = None,
    anp_trigger_size: int | None = None,
    anp_trigger_pos: str | None = None,
    anp_asr_samples: int | None = None,
):
    """Run ANP repair. Default clean-subset selection uses seed (or 0 if omitted).

    Set anp_clean_subset_seed only to override the subset seed independently.
    Supply a known CIFAR-10 trigger type and target to include ASR.
    """
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
        anp_iterations=anp_iterations,
        anp_clean_subset_seed=anp_clean_subset_seed,
        anp_prune_fraction=anp_prune_fraction,
        anp_attack_target=anp_attack_target,
        anp_trigger_type=anp_trigger_type,
        anp_trigger_alpha=anp_trigger_alpha,
        anp_trigger_size=anp_trigger_size,
        anp_trigger_pos=anp_trigger_pos,
        anp_asr_samples=anp_asr_samples,
    ))
