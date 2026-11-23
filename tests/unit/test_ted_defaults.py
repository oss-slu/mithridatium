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


def _cli_defaults():
    command = typer.main.get_command(app).get_command(None, "detect")
    assert command is not None
    return {parameter.name: parameter.default for parameter in command.params}


def _assert_cli_defaults(expected):
    cli_defaults = _cli_defaults()
    for name, value in expected.items():
        assert cli_defaults[name] == value


def _assert_mcp_defaults(function, expected):
    assert _signature_defaults(function, expected) == expected


def test_ted_defaults_match_function_cli_and_mcp():
    shared = {
        "ted_num_samples": TED_DEFAULTS.num_samples,
        "ted_reference_size": TED_DEFAULTS.reference_size,
        "ted_contamination": TED_DEFAULTS.contamination,
    }
    ted_function = {
        "reference_size": TED_DEFAULTS.reference_size,
        "contamination": TED_DEFAULTS.contamination,
    }

    assert _signature_defaults(ted, ted_function) == ted_function
    _assert_mcp_defaults(run_ted, shared)
    _assert_mcp_defaults(_detect_kwargs, shared)
    _assert_cli_defaults(shared)


def test_tedpp_defaults_match_function_cli_and_mcp():
    expected = {
        "contamination": TED_DEFAULTS.contamination,
        "samples_per_class": TEDPP_DEFAULTS.samples_per_class,
        "beta": TEDPP_DEFAULTS.beta,
    }
    mcp = {
        "ted_num_samples": TED_DEFAULTS.num_samples,
        "ted_reference_size": TED_DEFAULTS.reference_size,
        "ted_contamination": TED_DEFAULTS.contamination,
        "tedpp_samples_per_class": TEDPP_DEFAULTS.samples_per_class,
        "tedpp_beta": TEDPP_DEFAULTS.beta,
    }

    assert _signature_defaults(tedpp, expected) == expected
    assert _signature_defaults(TEDPLUS.__init__, expected) == expected
    assert "reference_size" not in inspect.signature(tedpp).parameters
    assert "reference_size" not in inspect.signature(TEDPLUS.__init__).parameters
    _assert_mcp_defaults(run_tedpp, mcp)
    _assert_mcp_defaults(_detect_kwargs, mcp)
    _assert_cli_defaults(
        {
            **mcp,
            "ted_num_samples": TED_DEFAULTS.num_samples,
        }
    )
