"""Unit tests for the audit MCP tools. The audit command itself is replaced with a fake."""

import inspect
import json

import pytest

pytest.importorskip("fastmcp")

from mithridatium.mcp import audit_mcp

pytestmark = pytest.mark.unit


def test_fail_on_backdoor_defaults_to_false_when_audit_is_called_as_a_function():
    # with typer.Option(False, ...) as the default, an omitted flag was a truthy
    # OptionInfo, so every flagged model came back from run_mmbd as a tool error
    assert inspect.signature(audit_mcp.audit).parameters["fail_on_backdoor"].default is False


def test_run_audit_returns_the_report(monkeypatch, tmp_path):
    seen = {}

    def fake_audit(**kwargs):
        seen.update(kwargs)
        with open(kwargs["out"], "w") as f:
            json.dump({"results": {"verdict": "attack"}}, f)

    monkeypatch.setattr(audit_mcp, "audit", fake_audit)
    monkeypatch.chdir(tmp_path)
    result = audit_mcp.run_mmbd(model="models/x.pth")

    assert "fail_on_backdoor" not in seen
    assert result["report"]["results"]["verdict"] == "attack"
    assert result["report_path"].endswith("x_mmbd.json")
