"""
Unit tests for `mithridatium agent`. The agent modules are replaced with fakes,
so no [agent] extra, API key, or GPU is needed.
"""

from __future__ import annotations

import sys
import types

import pytest
from typer.testing import CliRunner

from mithridatium.cli import AGENTS, EXIT_USAGE_ERROR, app

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture
def fake_agents(monkeypatch):
    """Install a fake mithridatium.mcp.<name>_agent module per agent; record calls."""
    calls = []
    for name in AGENTS:
        mod = types.ModuleType(f"mithridatium.mcp.{name}_agent")
        mod.DEFAULT_PROMPT = f"default {name} prompt"

        async def run(prompt, _name=name):
            calls.append((_name, prompt))
            return f"{_name} done"

        setattr(mod, f"{name}_agent", run)
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return calls


def test_default_routes_to_supervisor(fake_agents):
    result = runner.invoke(app, ["agent", "check models/x.pth"])
    assert result.exit_code == 0, result.output
    assert fake_agents == [("supervisor", "check models/x.pth")]
    assert "supervisor done" in result.output


@pytest.mark.parametrize("name", AGENTS)
@pytest.mark.parametrize("flag", ["--agent", "-a"])
def test_agent_flag_routes(fake_agents, name, flag):
    result = runner.invoke(app, ["agent", flag, name, "hello"])
    assert result.exit_code == 0, result.output
    assert fake_agents == [(name, "hello")]


def test_no_prompt_uses_agent_default(fake_agents):
    result = runner.invoke(app, ["agent", "-a", "detect"])
    assert result.exit_code == 0, result.output
    assert fake_agents == [("detect", "default detect prompt")]


def test_unknown_agent_errors(fake_agents):
    result = runner.invoke(app, ["agent", "-a", "bogus", "hi"])
    assert result.exit_code == EXIT_USAGE_ERROR
    for name in AGENTS:
        assert name in result.output
    assert fake_agents == []


def test_missing_extra_errors(monkeypatch):
    # a None entry in sys.modules makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "mithridatium.mcp.supervisor_agent", None)
    result = runner.invoke(app, ["agent", "hi"])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "pip install -e '.[agent]'" in result.output


def test_missing_api_key_errors(fake_agents, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    result = runner.invoke(app, ["agent", "hi"])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "OPENAI_API_KEY" in result.output
    assert "Traceback" not in result.output
    assert fake_agents == []
