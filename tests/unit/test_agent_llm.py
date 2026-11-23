"""Chat model selection for the agents. init_chat_model is mocked, so no model is called."""

import pytest

pytest.importorskip("langchain")

from mithridatium.mcp import llm as llm_mod

pytestmark = pytest.mark.unit


@pytest.fixture
def init_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_mod, "init_chat_model", lambda **kw: calls.append(kw) or object())
    monkeypatch.delenv("MITHRIDATIUM_LLM", raising=False)
    return calls


def test_default_is_gpt_4o_mini(init_calls):
    llm_mod.chat_model()
    assert init_calls == [{"model": "gpt-4o-mini"}]


def test_env_var_overrides_default(init_calls, monkeypatch):
    monkeypatch.setenv("MITHRIDATIUM_LLM", "ollama:llama3.1")
    llm_mod.chat_model()
    assert init_calls == [{"model": "ollama:llama3.1"}]


def test_argument_overrides_env_var(init_calls, monkeypatch):
    monkeypatch.setenv("MITHRIDATIUM_LLM", "ollama:llama3.1")
    llm_mod.chat_model("gpt-4o")
    assert init_calls == [{"model": "gpt-4o"}]
