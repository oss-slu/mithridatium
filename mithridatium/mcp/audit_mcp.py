from __future__ import annotations
from typing import Any
from mithridatium.cli import audit
import contextlib #to send output to buffer instead of the MCP stdio channel
import io
import json
import tempfile #to create a temporary file to store the audit results
from pathlib import Path
import typer

from mithridatium.mcp import mcp

# audit parameters
def _audit_kwargs(
    #general parameters
    defense: str,
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    arch: str = "resnet18",
    provider: str = "torchvision",
    hf_model_id: str = "microsoft/resnet-50",

    #freeeagle parameters
    freeeagle_num_classes: int = 0,
    freeeagle_num_dummy: int = 1,
    freeeagle_num_important_neurons: int = 5,
    freeeagle_metric: str = "softmax_score",
    freeeagle_use_transpose_correction: bool = False,
    freeeagle_bound_on: bool = True,
    freeeagle_optimize_steps: int = 300,
    freeeagle_learning_rate: float = 1e-2,
    freeeagle_weight_decay: float = 5e-3,
    freeeagle_anomaly_threshold: float = 2.0,
    freeeagle_inspect_layer_position: int = 2,

    #aeva parameters
    aeva_samples_per_class: int = 5,
    aeva_hsja_iterations: int = 5,
    aeva_hsja_max_num_evals: int = 2000,
    aeva_hsja_init_num_evals: int = 50,
    aeva_hsja_query_batch_size: int = 256,
    aeva_anomaly_index_threshold: float = 4.0,
    aeva_verbose: bool = False,
    aeva_sp: int = 0,
    aeva_ep: int = 1,

    #strip parameters
    strip_threshold_mode: str = "dynamic_mad",
    strip_entropy_mean_threshold: float | None = None,
    strip_mad_scale: float = 2.5,
    strip_suspicious_fraction_threshold: float = 0.20,
) -> dict[str, Any]:
    return {
        "defense": defense,
        "model": model,
        "data": data,
        "arch": arch,
        "provider": provider,
        "hf_model_id": hf_model_id,
        #freeeagle parameters
        "freeeagle_num_classes": freeeagle_num_classes,
        "freeeagle_num_dummy": freeeagle_num_dummy,
        "freeeagle_num_important_neurons": freeeagle_num_important_neurons,
        "freeeagle_metric": freeeagle_metric,
        "freeeagle_use_transpose_correction": freeeagle_use_transpose_correction,
        "freeeagle_bound_on": freeeagle_bound_on,
        "freeeagle_optimize_steps": freeeagle_optimize_steps,
        "freeeagle_learning_rate": freeeagle_learning_rate,
        "freeeagle_weight_decay": freeeagle_weight_decay,
        "freeeagle_anomaly_threshold": freeeagle_anomaly_threshold,
        "freeeagle_inspect_layer_position": freeeagle_inspect_layer_position,
        #aeva parameters
        "aeva_samples_per_class": aeva_samples_per_class,
        "aeva_hsja_iterations": aeva_hsja_iterations,
        "aeva_hsja_max_num_evals": aeva_hsja_max_num_evals,
        "aeva_hsja_init_num_evals": aeva_hsja_init_num_evals,
        "aeva_hsja_query_batch_size": aeva_hsja_query_batch_size,
        "aeva_anomaly_index_threshold": aeva_anomaly_index_threshold,
        "aeva_verbose": aeva_verbose,
        "aeva_sp": aeva_sp,
        "aeva_ep": aeva_ep,
        #strip parameters
        "strip_threshold_mode": strip_threshold_mode,
        "strip_entropy_mean_threshold": strip_entropy_mean_threshold,
        "strip_mad_scale": strip_mad_scale,
        "strip_suspicious_fraction_threshold": strip_suspicious_fraction_threshold,
    }

# setup context manager to redirect stdout and stderr away from the MCP stdio channel
@contextlib.contextmanager
def _catch_output():
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            yield output
        except typer.Exit as err:
            raise RuntimeError(output.getvalue().strip()) from err

def _run_audit(audit_kwargs: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = str(Path(tmp_dir) / "report.json")
        with _catch_output():
            audit(**audit_kwargs, out=out_path, force=True)
        return json.loads(Path(out_path).read_text())

@mcp.tool()
def run_mmbd(
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    arch: str = "resnet18",
    provider: str = "torchvision",
    hf_model_id: str = "microsoft/resnet-50",
):
    """Run a MBBD audit and return the mithridatium audit report."""
    return _run_audit(_audit_kwargs(
        "mmbd", 
        model=model,
        data=data,
        arch=arch,
        provider=provider,
        hf_model_id=hf_model_id,
    ))

@mcp.tool()
def run_freeeagle(
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    arch: str = "resnet18",
    provider: str = "torchvision",
    hf_model_id: str = "microsoft/resnet-50",
    freeeagle_num_classes: int = 0,
    freeeagle_num_dummy: int = 1,
    freeeagle_num_important_neurons: int = 5,
    freeeagle_metric: str = "softmax_score",
    freeeagle_use_transpose_correction: bool = False,
    freeeagle_bound_on: bool = True,
    freeeagle_optimize_steps: int = 300,
    freeeagle_learning_rate: float = 1e-2,
    freeeagle_weight_decay: float = 5e-3,
    freeeagle_anomaly_threshold: float = 2.0,
    freeeagle_inspect_layer_position: int = 2,
):
    """Run a FreeEagle audit and return the mithridatium audit report."""
    return _run_audit(_audit_kwargs(
        "freeeagle",
        model=model,
        data=data,
        arch=arch,
        provider=provider,
        hf_model_id=hf_model_id,
        freeeagle_num_classes=freeeagle_num_classes,
        freeeagle_num_dummy=freeeagle_num_dummy,
        freeeagle_num_important_neurons=freeeagle_num_important_neurons,
        freeeagle_metric=freeeagle_metric,
        freeeagle_use_transpose_correction=freeeagle_use_transpose_correction,
        freeeagle_bound_on=freeeagle_bound_on,
        freeeagle_optimize_steps=freeeagle_optimize_steps,
        freeeagle_learning_rate=freeeagle_learning_rate,
        freeeagle_weight_decay=freeeagle_weight_decay,
        freeeagle_anomaly_threshold=freeeagle_anomaly_threshold,
        freeeagle_inspect_layer_position=freeeagle_inspect_layer_position,
    ))

@mcp.tool()
def run_aeva(
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    arch: str = "resnet18",
    provider: str = "torchvision",
    hf_model_id: str = "microsoft/resnet-50",
    aeva_samples_per_class: int = 5,
    aeva_hsja_iterations: int = 5,
    aeva_hsja_max_num_evals: int = 2000,
    aeva_hsja_init_num_evals: int = 50,
    aeva_hsja_query_batch_size: int = 256,
    aeva_anomaly_index_threshold: float = 4.0,
    aeva_verbose: bool = False,
    aeva_sp: int = 0,
    aeva_ep: int = 1,
):
    """Run an AEVA audit and return the mithridatium audit report."""
    return _run_audit(_audit_kwargs(
        "aeva",
        model=model,
        data=data,
        arch=arch,
        provider=provider,
        hf_model_id=hf_model_id,
        aeva_samples_per_class=aeva_samples_per_class,
        aeva_hsja_iterations=aeva_hsja_iterations,
        aeva_hsja_max_num_evals=aeva_hsja_max_num_evals,
        aeva_hsja_init_num_evals=aeva_hsja_init_num_evals,
        aeva_hsja_query_batch_size=aeva_hsja_query_batch_size,
        aeva_anomaly_index_threshold=aeva_anomaly_index_threshold,
        aeva_verbose=aeva_verbose,
        aeva_sp=aeva_sp,
        aeva_ep=aeva_ep,
    ))

@mcp.tool()
def run_strip(
    model: str = "models/resnet18.pth",
    data: str = "cifar10",
    arch: str = "resnet18",
    provider: str = "torchvision",
    hf_model_id: str = "microsoft/resnet-50",
    strip_threshold_mode: str = "dynamic_mad",
    strip_entropy_mean_threshold: float | None = None,
    strip_mad_scale: float = 2.5,
    strip_suspicious_fraction_threshold: float = 0.20,
):
    """Run a STRIP audit and return the mithridatium audit report."""
    return _run_audit(_audit_kwargs(
        "strip",
        model=model,
        data=data,
        arch=arch,
        provider=provider,
        hf_model_id=hf_model_id,
        strip_threshold_mode=strip_threshold_mode,
        strip_entropy_mean_threshold=strip_entropy_mean_threshold,
        strip_mad_scale=strip_mad_scale,
        strip_suspicious_fraction_threshold=strip_suspicious_fraction_threshold,
    ))

