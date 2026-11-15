"""TED/TED++ defaults stay aligned across methods, CLI, and MCP."""

import inspect

import pytest
import typer.main

from mithridatium.cli import app
from mithridatium.detect.ted import ted
from mithridatium.detect.ted_defaults import TED_DEFAULTS, TEDPP_DEFAULTS
from mithridatium.detect.tedpp import TEDPLUS, tedpp
from mithridatium.mcp.detect_mcp import _detect_kwargs, run_ted, run_tedpp

pytestmark = pytest.mark.unit


def _signature_defaults(function, names):
    signature = inspect.signature(function)
    return {name: signature.parameters[name].default for name in names}


def test_ted_defaults_match_function_cli_and_mcp():
    names = ("num_samples", "reference_size", "contamination")
    expected = TED_DEFAULTS

    assert _signature_defaults(ted, names[1:]) == {
        name: expected[name] for name in names[1:]
    }
    assert _signature_defaults(
        _detect_kwargs,
        ("ted_num_samples", "ted_reference_size", "ted_contamination"),
    ) == {
        "ted_num_samples": expected["num_samples"],
        "ted_reference_size": expected["reference_size"],
        "ted_contamination": expected["contamination"],
    }
    assert _signature_defaults(
        run_ted,
        ("ted_num_samples", "ted_reference_size", "ted_contamination"),
    ) == {
        "ted_num_samples": expected["num_samples"],
        "ted_reference_size": expected["reference_size"],
        "ted_contamination": expected["contamination"],
    }

    detect_command = typer.main.get_command(app).get_command(None, "detect")
    assert detect_command is not None
    cli_defaults = {parameter.name: parameter.default for parameter in detect_command.params}
    assert cli_defaults["ted_num_samples"] == expected["num_samples"]
    assert cli_defaults["ted_reference_size"] == expected["reference_size"]
    assert cli_defaults["ted_contamination"] == expected["contamination"]


def test_tedpp_defaults_match_function_cli_and_mcp():
    names = ("reference_size", "contamination", "samples_per_class", "beta")
    expected = TEDPP_DEFAULTS
    expected_values = {name: expected[name] for name in names}

    assert _signature_defaults(tedpp, names) == expected_values
    assert _signature_defaults(TEDPLUS.__init__, names) == expected_values
    mcp_names = (
        "ted_num_samples",
        "ted_reference_size",
        "ted_contamination",
        "tedpp_samples_per_class",
        "tedpp_beta",
    )
    assert _signature_defaults(run_tedpp, mcp_names) == {
        "ted_num_samples": expected["num_samples"],
        "ted_reference_size": expected["reference_size"],
        "ted_contamination": expected["contamination"],
        "tedpp_samples_per_class": expected["samples_per_class"],
        "tedpp_beta": expected["beta"],
    }
    assert _signature_defaults(
        _detect_kwargs,
        (
            "ted_num_samples",
            "ted_reference_size",
            "ted_contamination",
            "tedpp_samples_per_class",
            "tedpp_beta",
        ),
    ) == {
        "ted_num_samples": expected["num_samples"],
        "ted_reference_size": expected["reference_size"],
        "ted_contamination": expected["contamination"],
        "tedpp_samples_per_class": expected["samples_per_class"],
        "tedpp_beta": expected["beta"],
    }

    detect_command = typer.main.get_command(app).get_command(None, "detect")
    assert detect_command is not None
    cli_defaults = {parameter.name: parameter.default for parameter in detect_command.params}
    assert cli_defaults["ted_num_samples"] == expected["num_samples"]
    assert cli_defaults["ted_reference_size"] == expected["reference_size"]
    assert cli_defaults["ted_contamination"] == expected["contamination"]
    assert cli_defaults["tedpp_samples_per_class"] == expected["samples_per_class"]
    assert cli_defaults["tedpp_beta"] == expected["beta"]
