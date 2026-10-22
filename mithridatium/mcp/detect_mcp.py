from __future__ import annotations
from typing import Any
from mithridatium.cli import detect
import json
from pathlib import Path

from mithridatium.mcp import mcp
from mithridatium.mcp.audit_mcp import _catch_output

# detect parameters
def _detect_kwargs(
    #general parameters
    method: str,
    model: str = "models/resnet18.pth",
    data: str = "cifar10",

    #scaleup parameters (sample count reduced from the CLI's 256 for smoke runs)
    scaleup_num_samples: int = 128,
    scaleup_threshold: float = 0.5,
    scaleup_scales: list[float] | None = None,

    #ted parameters (reduced from the CLI's 256 / 1000 for smoke runs)
    ted_num_samples: int = 128,
    ted_reference_size: int = 250,
    ted_contamination: float = 0.01,

    #ibdpsc parameters (sample count reduced from the CLI's 256 for smoke runs)
    ibdpsc_num_samples: int = 128,
    ibdpsc_clean_samples: int = 100,
    ibdpsc_scale: float = 1.5,
    ibdpsc_num_models: int = 5,
    ibdpsc_threshold: float = 0.9,
) -> dict[str, Any]:
    scales = tuple(scaleup_scales) if scaleup_scales else (3, 5, 7, 9, 11)
    return {
        "method": method,
        "model": model,
        "data": data,
        #scaleup parameters (the CLI takes these as strings)
        "num_samples": str(scaleup_num_samples),
        "scaleup_threshold": str(scaleup_threshold),
        "scaleup_scales": str(scales),
        #ted parameters
        "ted_num_samples": ted_num_samples,
        "ted_reference_size": ted_reference_size,
        "ted_contamination": ted_contamination,
        #ibdpsc parameters
        "ibdpsc_num_samples": ibdpsc_num_samples,
        "ibdpsc_clean_samples": ibdpsc_clean_samples,
        "ibdpsc_scale": ibdpsc_scale,
        "ibdpsc_num_models": ibdpsc_num_models,
        "ibdpsc_threshold": ibdpsc_threshold,
    }

def _run_detect(detect_kwargs: dict[str, Any]) -> dict[str, Any]:
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    saved = reports / f"{Path(detect_kwargs['model']).stem}_{detect_kwargs['method']}.json"
    with _catch_output():
        detect(**detect_kwargs, out=str(saved), force=True)
    return json.loads(Path(saved).read_text())

@mcp.tool(run_in_thread=False)
def run_scaleup(
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    scaleup_num_samples: int = 128,
    scaleup_threshold: float = 0.5,
    scaleup_scales: list[float] | None = None,
):
    """Run SCALE-UP detection and return the mithridatium detect report."""
    return _run_detect(_detect_kwargs(
        "scaleup",
        model=model,
        data=data,
        scaleup_num_samples=scaleup_num_samples,
        scaleup_threshold=scaleup_threshold,
        scaleup_scales=scaleup_scales,
    ))

@mcp.tool(run_in_thread=False)
def run_ted(
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    ted_num_samples: int = 128,
    ted_reference_size: int = 250,
    ted_contamination: float = 0.01,
):
    """Run TED detection and return the mithridatium detect report."""
    return _run_detect(_detect_kwargs(
        "ted",
        model=model,
        data=data,
        ted_num_samples=ted_num_samples,
        ted_reference_size=ted_reference_size,
        ted_contamination=ted_contamination,
    ))

@mcp.tool(run_in_thread=False)
def run_ibdpsc(
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    ibdpsc_num_samples: int = 128,
    ibdpsc_clean_samples: int = 100,
    ibdpsc_scale: float = 1.5,
    ibdpsc_num_models: int = 5,
    ibdpsc_threshold: float = 0.9,
):
    """Run IBD-PSC detection and return the mithridatium detect report."""
    return _run_detect(_detect_kwargs(
        "ibdpsc",
        model=model,
        data=data,
        ibdpsc_num_samples=ibdpsc_num_samples,
        ibdpsc_clean_samples=ibdpsc_clean_samples,
        ibdpsc_scale=ibdpsc_scale,
        ibdpsc_num_models=ibdpsc_num_models,
        ibdpsc_threshold=ibdpsc_threshold,
    ))
