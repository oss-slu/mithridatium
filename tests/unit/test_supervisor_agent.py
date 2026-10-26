import asyncio
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_mcp_adapters")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from mithridatium.mcp import supervisor_agent as sup


class ScriptedModel(GenericFakeChatModel):
    # GenericFakeChatModel can't bind tools; the script decides the tool calls
    def bind_tools(self, tools, **kwargs):
        return self


def _call(name, request, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {"request": request}, "id": f"call_{i}"}])


def _run(monkeypatch, prompt, script, audit_reply="audit: clean"):
    mocks = {
        "audit_agent": AsyncMock(return_value=audit_reply),
        "detect_agent": AsyncMock(return_value="detect: clean"),
        "repair_agent": AsyncMock(return_value="repaired: models/x_clp_repaired.pth"),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(sup, name, mock)
    messages = [_call(name, req, i) for i, (name, req) in enumerate(script)]
    messages.append(AIMessage(content="summary"))
    monkeypatch.setattr(sup, "init_chat_model", lambda **_: ScriptedModel(messages=iter(messages)))
    assert asyncio.run(sup.supervisor_agent(prompt)) == "summary"
    return mocks


@pytest.mark.parametrize("prompt, tool, agent", [
    ("Audit models/resnet18_poison.pth", "audit", "audit_agent"),
    ("Run TED on models/resnet18_poison.pth", "detect", "detect_agent"),
    ("Repair models/resnet18_poison.pth with CLP", "repair", "repair_agent"),
])
def test_single_agent_routing(monkeypatch, prompt, tool, agent):
    mocks = _run(monkeypatch, prompt, [(tool, prompt)])
    mocks[agent].assert_awaited_once_with(prompt)
    for other, mock in mocks.items():
        if other != agent:
            mock.assert_not_awaited()


def test_conditional_repair_chains_after_flagged_audit(monkeypatch):
    prompt = "Check models/resnet18_poison.pth and repair it if it's poisoned"
    mocks = _run(
        monkeypatch, prompt,
        [("audit", "Audit models/resnet18_poison.pth"), ("repair", "Repair models/resnet18_poison.pth")],
        audit_reply="mmbd: attack",
    )
    mocks["audit_agent"].assert_awaited_once()
    mocks["repair_agent"].assert_awaited_once_with("Repair models/resnet18_poison.pth")


def test_clean_audit_skips_repair(monkeypatch):
    prompt = "Check models/resnet18_clean.pth and repair it if it's poisoned"
    mocks = _run(monkeypatch, prompt, [("audit", "Audit models/resnet18_clean.pth")])
    mocks["audit_agent"].assert_awaited_once()
    mocks["repair_agent"].assert_not_awaited()


def test_supervisor_tools_are_the_three_agents():
    assert {t.name for t in (sup.audit, sup.detect, sup.repair)} == {"audit", "detect", "repair"}
