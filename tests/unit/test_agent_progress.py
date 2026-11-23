"""Unit tests for the agent progress lines (mithridatium/mcp/progress.py)."""

import asyncio
import io

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_mcp_adapters")

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from mithridatium.mcp import progress
from mithridatium.mcp import supervisor_agent as sup

pytestmark = pytest.mark.unit


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


@tool
def run_mmbd(model: str) -> str:
    """Fake MMBD tool."""
    return "verdict: attack"


async def _nested_audit(request, llm=None, **kwargs):
    # a real nested agent, invoked without a config, like the specialists
    model = ScriptedModel(messages=iter([_call("run_mmbd", {"model": "models/x.pth"}, 0), AIMessage(content="attack")]))
    agent = create_agent(model, [run_mmbd])
    result = await agent.ainvoke({"messages": request})
    return result["messages"][-1].content


def test_progress_shows_specialists_and_their_tool_calls(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(progress, "progress_config", lambda on: {"callbacks": [progress.ProgressPrinter(stream)]} if on else None)
    monkeypatch.setattr(sup, "progress_config", progress.progress_config)
    monkeypatch.setattr(sup, "audit_agent", _nested_audit)
    script = [_call("audit", {"request": "Audit models/x.pth"}, 0), AIMessage(content="summary")]
    monkeypatch.setattr(sup, "chat_model", lambda llm=None: ScriptedModel(messages=iter(script)))

    assert asyncio.run(sup.supervisor_agent("check x", llm="gpt-4o-mini", memory=False, progress=True)) == "summary"
    lines = stream.getvalue().splitlines()
    assert lines[0] == "▶ audit: Audit models/x.pth"
    assert lines[1] == "    · run_mmbd: model=models/x.pth"
    assert lines[2].startswith("    ✓ run_mmbd done (")
    assert lines[3].startswith("✓ audit done (")


def test_no_progress_by_default(monkeypatch, capsys):
    monkeypatch.setattr(sup, "audit_agent", _nested_audit)
    script = [_call("audit", {"request": "Audit models/x.pth"}, 0), AIMessage(content="summary")]
    monkeypatch.setattr(sup, "chat_model", lambda llm=None: ScriptedModel(messages=iter(script)))
    asyncio.run(sup.supervisor_agent("check x", llm="gpt-4o-mini", memory=False))
    assert capsys.readouterr().err == ""


def test_failed_tool_is_marked():
    stream = io.StringIO()
    printer = progress.ProgressPrinter(stream)
    printer.on_tool_start({"name": "run_aeva"}, "", run_id=1, inputs={"model": "m.pth"})
    printer.on_tool_error(RuntimeError("CUDA out of memory"), run_id=1)
    assert stream.getvalue().splitlines()[-1].startswith("    ✗ run_aeva failed (")
    assert "CUDA out of memory" in stream.getvalue()
