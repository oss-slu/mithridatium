"""
Unit tests for `mithridatium agent`. The agent modules are replaced with fakes,
so no [agent] extra, API key, or GPU is needed.
"""

from __future__ import annotations

import os
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
        # mirrors mithridatium.mcp.llm.resolve_llm without importing the [agent] extra
        mod.resolve_llm = lambda llm=None: llm or os.environ.get("MITHRIDATIUM_LLM") or "gpt-4o-mini"

        async def run(prompt, llm=None, _name=name):
            calls.append((_name, prompt, llm))
            return f"{_name} done"

        setattr(mod, f"{name}_agent", run)
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("MITHRIDATIUM_LLM", raising=False)
    return calls


def test_default_routes_to_supervisor(fake_agents):
    result = runner.invoke(app, ["agent", "check models/x.pth"])
    assert result.exit_code == 0, result.output
    assert fake_agents == [("supervisor", "check models/x.pth", "gpt-4o-mini")]
    assert "supervisor done" in result.output


@pytest.mark.parametrize("name", AGENTS)
@pytest.mark.parametrize("flag", ["--agent", "-a"])
def test_agent_flag_routes(fake_agents, name, flag):
    result = runner.invoke(app, ["agent", flag, name, "hello"])
    assert result.exit_code == 0, result.output
    assert fake_agents == [(name, "hello", "gpt-4o-mini")]


def test_no_prompt_uses_agent_default(fake_agents):
    result = runner.invoke(app, ["agent", "-a", "detect"])
    assert result.exit_code == 0, result.output
    assert fake_agents == [("detect", "default detect prompt", "gpt-4o-mini")]


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


def test_llm_flag_is_passed_to_agent(fake_agents):
    result = runner.invoke(app, ["agent", "--llm", "gpt-4o", "hi"])
    assert result.exit_code == 0, result.output
    assert fake_agents == [("supervisor", "hi", "gpt-4o")]


def test_llm_env_var_is_used(fake_agents, monkeypatch):
    monkeypatch.setenv("MITHRIDATIUM_LLM", "gpt-4.1")
    result = runner.invoke(app, ["agent", "-a", "audit", "hi"])
    assert result.exit_code == 0, result.output
    assert fake_agents == [("audit", "hi", "gpt-4.1")]


def test_non_openai_llm_does_not_need_openai_key(fake_agents, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    result = runner.invoke(app, ["agent", "--llm", "ollama:llama3.1", "hi"])
    assert result.exit_code == 0, result.output
    assert fake_agents == [("supervisor", "hi", "ollama:llama3.1")]
