import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

pytest.importorskip("langchain")
pytest.importorskip("langchain_mcp_adapters")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mithridatium import cli
from mithridatium.cli import EXIT_CANT_CREATE
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


def test_build_agent_run_summary_from_fake_message_history():
    prompt = "Check models/resnet18_poison.pth and repair it if it's poisoned"
    messages = [
        HumanMessage(content=prompt),
        AIMessage(
            content="",
            tool_calls=[{"name": "audit", "args": {"request": "Audit models/resnet18_poison.pth"}, "id": "call_0"}],
        ),
        ToolMessage(
            content='{"report_path": "reports/resnet18_poison_mmbd.json", "report": {"results": {"verdict": "attack"}}}',
            tool_call_id="call_0",
            name="audit",
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "repair", "args": {"request": "Repair models/resnet18_poison.pth"}, "id": "call_1"}],
        ),
        ToolMessage(
            content='{"checkpoint": "models/resnet18_poison_clp_repaired.pth", "report_path": "reports/resnet18_poison_clp.json", "verdict": "repaired"}',
            tool_call_id="call_1",
            name="repair",
        ),
        AIMessage(content="Final agent answer: model was backdoored and has been repaired."),
    ]

    summary = sup.build_agent_run_summary(
        prompt=prompt,
        messages=messages,
        agent_name="supervisor",
        model_name="gpt-4o-mini",
    )

    assert summary["prompt"] == prompt
    assert summary["agent"] == "supervisor"
    assert summary["chat_model"] == "gpt-4o-mini"
    assert summary["final_message"] == "Final agent answer: model was backdoored and has been repaired."
    assert len(summary["steps"]) == 2

    step1 = summary["steps"][0]
    assert step1["step"] == 1
    assert step1["specialist"] == "audit"
    assert step1["request"] == "Audit models/resnet18_poison.pth"
    assert "reports/resnet18_poison_mmbd.json" in step1["report_paths"]
    assert step1["verdict"] == "attack"

    step2 = summary["steps"][1]
    assert step2["step"] == 2
    assert step2["specialist"] == "repair"
    assert step2["request"] == "Repair models/resnet18_poison.pth"
    assert "models/resnet18_poison_clp_repaired.pth" in step2["checkpoint_paths"]
    assert "reports/resnet18_poison_clp.json" in step2["report_paths"]
    assert step2["verdict"] == "repaired"


def test_supervisor_agent_run_with_summary(monkeypatch):
    prompt = "Check models/resnet18_poison.pth and repair it if it's poisoned"
    mocks = {
        "audit_agent": AsyncMock(return_value="mmbd: attack (reports/resnet18_poison_mmbd.json)"),
        "detect_agent": AsyncMock(return_value="detect: clean"),
        "repair_agent": AsyncMock(return_value="repaired: models/resnet18_clp_repaired.pth, report: reports/resnet18_clp.json"),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(sup, name, mock)

    script = [
        ("audit", "Audit models/resnet18_poison.pth"),
        ("repair", "Repair models/resnet18_poison.pth"),
    ]
    messages = [_call(name, req, i) for i, (name, req) in enumerate(script)]
    messages.append(AIMessage(content="Run completed successfully."))
    monkeypatch.setattr(sup, "init_chat_model", lambda **_: ScriptedModel(messages=iter(messages)))

    final_msg, summary = asyncio.run(sup.supervisor_agent(prompt, return_summary=True))
    assert final_msg == "Run completed successfully."
    assert summary["prompt"] == prompt
    assert summary["agent"] == "supervisor"
    assert summary["chat_model"] == "gpt-4o-mini"
    assert len(summary["steps"]) == 2
    assert summary["steps"][0]["specialist"] == "audit"
    assert summary["steps"][0]["report_paths"] == ["reports/resnet18_poison_mmbd.json"]
    assert summary["steps"][1]["specialist"] == "repair"
    assert summary["steps"][1]["checkpoint_paths"] == ["models/resnet18_clp_repaired.pth"]


def test_cli_agent_without_report(monkeypatch):
    runner = CliRunner()
    fake_summary = {"prompt": "test", "agent": "supervisor", "chat_model": "gpt-4o-mini", "steps": []}
    monkeypatch.setattr(
        sup,
        "supervisor_agent",
        AsyncMock(return_value=("fake final message", fake_summary)),
    )

    result = runner.invoke(cli.app, ["agent", "test prompt"])
    assert result.exit_code == 0
    assert "fake final message" in result.output


def test_cli_agent_with_report_and_force(monkeypatch, tmp_path):
    runner = CliRunner()
    report_file = tmp_path / "agent_summary.json"
    fake_summary = {
        "prompt": "test prompt",
        "agent": "supervisor",
        "chat_model": "gpt-4o-mini",
        "steps": [
            {
                "step": 1,
                "specialist": "audit",
                "request": "check model",
                "report_paths": ["reports/model_mmbd.json"],
                "checkpoint_paths": [],
                "verdict": "clean",
            }
        ],
        "final_message": "all clean",
    }
    monkeypatch.setattr(
        sup,
        "supervisor_agent",
        AsyncMock(return_value=("all clean", fake_summary)),
    )

    result = runner.invoke(cli.app, ["agent", "test prompt", "--report", str(report_file)])
    assert result.exit_code == 0
    assert report_file.exists()
    saved = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved["prompt"] == "test prompt"
    assert saved["steps"][0]["specialist"] == "audit"

    result_conflict = runner.invoke(cli.app, ["agent", "test prompt", "--report", str(report_file)])
    assert result_conflict.exit_code == EXIT_CANT_CREATE

    result_forced = runner.invoke(cli.app, ["agent", "test prompt", "--report", str(report_file), "--force"])
    assert result_forced.exit_code == 0

