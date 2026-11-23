"""
Unit tests for `mithridatium agent`. The agent modules are replaced with fakes,
so no [agent] extra, API key, or GPU is needed.
"""

from __future__ import annotations

import os
import json
import sys
import types

import pytest
from typer.testing import CliRunner

from mithridatium.cli import (
    AGENTS,
    EXIT_CANT_CREATE,
    EXIT_USAGE_ERROR,
    app,
)

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture
def fake_agents(monkeypatch):
    calls = []
    for name in AGENTS:
        mod = types.ModuleType(f"mithridatium.mcp.{name}_agent")
        mod.DEFAULT_PROMPT = f"default {name} prompt"
        # mirrors mithridatium.mcp.llm.resolve_llm without importing the [agent] extra
        mod.resolve_llm = lambda llm=None: llm or os.environ.get("MITHRIDATIUM_LLM") or "gpt-4o-mini"

        async def run(prompt, llm=None, _name=name, **kwargs):
            calls.append((_name, prompt, llm))
            if kwargs.get("return_summary"):
                fake_summary = {
                    "prompt": prompt,
                    "agent": _name,
                    "chat_model": llm,
                    "steps": [],
                    "final_message": f"{_name} done",
                }
                return f"{_name} done", fake_summary
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

# verify report flag does not allow stdout dash
def test_report_rejects_stdout_dash(fake_agents):
    result = runner.invoke(app, ["agent", "hi", "--report", "-"])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "does not support '-' stdout" in result.output


# verify report flag can only be used with supervisor agent
def test_report_rejects_non_supervisor(fake_agents, tmp_path):
    out_file = tmp_path / "summary.json"
    result = runner.invoke(app, ["agent", "-a", "detect", "hi", "--report", str(out_file)])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "only supported for the supervisor agent" in result.output


# verify report writes summary json and respects the force flag
def test_report_writes_file_and_respects_force(fake_agents, tmp_path):
    report_file = tmp_path / "summary.json"
    result = runner.invoke(app, ["agent", "check model", "--report", str(report_file)])
    assert result.exit_code == 0
    assert report_file.exists()
    saved = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved["prompt"] == "check model"
    assert saved["agent"] == "supervisor"

    # conflicts when file already exists without --force
    result_conflict = runner.invoke(app, ["agent", "check model", "--report", str(report_file)])
    assert result_conflict.exit_code == EXIT_CANT_CREATE

    # overwrites with --force
    result_forced = runner.invoke(app, ["agent", "check model", "--report", str(report_file), "--force"])
    assert result_forced.exit_code == 0
