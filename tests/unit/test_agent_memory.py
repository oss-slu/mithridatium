"""
Unit tests for agent memory (mithridatium/mcp/memory.py) and how the supervisor
and specialists use it. Uses the fake chat model; no API key or MCP server.
"""

import asyncio
import json

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_mcp_adapters")
pytest.importorskip("langmem")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from mithridatium.mcp import memory as mem
from mithridatium.mcp import supervisor_agent as sup

pytestmark = pytest.mark.unit


class ScriptedModel(GenericFakeChatModel):
    # records what each model call saw so tests can check the recalled context
    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


def _call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def _supervise(monkeypatch, prompt, script, **kwargs):
    """Run the supervisor with scripted tool calls; returns (summary, model)."""
    messages = [_call(name, args, i) for i, (name, args) in enumerate(script)]
    messages.append(AIMessage(content="summary"))
    model = ScriptedModel(messages=iter(messages), seen=[])
    monkeypatch.setattr(sup, "chat_model", lambda llm=None: model)
    _, summary = asyncio.run(sup.supervisor_agent(prompt, llm="gpt-4o-mini", return_summary=True, **kwargs))
    return summary, model


def _system_prompt(model) -> str:
    return model.seen[0][0].content


def _fake_specialist(received, tool="run_mmbd", status="success", content='{"verdict": "attack", "report_path": "reports/x_mmbd.json"}'):
    """A specialist that records the memory it got and leaves a one-tool trace."""

    async def run(request, llm=None, memory=None, trace=None):
        received.append(memory)
        if trace is not None:
            trace.extend([
                _call(tool, {"model_path": "models/x.pth"}, 0),
                ToolMessage(content=content, tool_call_id="call_0", name=tool, status=status),
                AIMessage(content=f"{tool} done"),
            ])
        return f"{tool} done"

    return run


def test_memory_written_in_one_run_is_found_in_the_next(monkeypatch, isolated_agent_memory):
    fact = "models/resnet18_poison.pth is a BadNets checkpoint targeting class 0"
    _supervise(monkeypatch, "remember this", [
        ("manage_semantic_memory", {"content": fact, "action": "create"}),
    ])
    assert isolated_agent_memory.exists()

    summary, model = _supervise(monkeypatch, "what do you know about resnet18_poison?", [
        ("search_semantic_memory", {"query": "resnet18_poison"}),
    ])
    assert fact in summary["steps"][0]["result"]
    # recalled into the prompt too
    assert fact in _system_prompt(model)
    assert mem.MEMORY_RULE in _system_prompt(model)


def test_specialists_get_read_only_memory(monkeypatch):
    received = []
    monkeypatch.setattr(sup, "audit_agent", _fake_specialist(received))
    _supervise(monkeypatch, "audit models/x.pth", [("audit", {"request": "Audit models/x.pth"})])

    (view,) = received
    assert not view.writable
    assert {t.name for t in view.tools()} == {"search_semantic_memory", "search_episodic_memory"}
    with pytest.raises(PermissionError):
        view.manage_tools()
    with pytest.raises(PermissionError):
        view.record_episode("audit", "should not be written")
    with pytest.raises(PermissionError):
        view.save()


def test_supervisor_gets_manage_and_search_tools(monkeypatch):
    names = []
    real = sup.create_agent
    monkeypatch.setattr(sup, "create_agent", lambda model, tools, **kw: names.extend(t.name for t in tools) or real(model, tools, **kw))
    _supervise(monkeypatch, "hi", [])
    assert {"manage_semantic_memory", "manage_episodic_memory", "search_semantic_memory", "search_episodic_memory"} <= set(names)


def test_specialist_episodes_are_recalled_by_that_specialist(monkeypatch):
    received = []
    monkeypatch.setattr(sup, "audit_agent", _fake_specialist(received))
    _supervise(monkeypatch, "audit models/x.pth", [("audit", {"request": "Audit models/x.pth"})])

    memory = mem.Memory.open()
    audit_context = memory.recall_context("Audit models/x.pth", agent="audit")
    assert "run_mmbd ok" in audit_context
    assert "verdict attack" in audit_context
    assert "reports/x_mmbd.json" in audit_context
    # other specialists don't see audit's tool history; the supervisor sees which specialist ran
    assert "run_mmbd" not in memory.recall_context("Detect models/x.pth", agent="detect")
    assert "audit ok" in memory.recall_context("audit models/x.pth", agent="supervisor")


def test_failed_tool_is_recorded_as_failed(monkeypatch):
    monkeypatch.setattr(sup, "audit_agent", _fake_specialist([], tool="run_aeva", status="error", content="Error: CUDA out of memory"))
    _supervise(monkeypatch, "audit models/x.pth with AEVA", [("audit", {"request": "Run AEVA on models/x.pth"})])
    context = mem.Memory.open().recall_context("AEVA models/x.pth", agent="audit")
    assert "run_aeva FAILED" in context
    assert "CUDA out of memory" in context


def test_recent_runs_are_listed_newest_first_even_when_older_ones_match_better():
    memory = mem.Memory.open()
    memory.record_episode("supervisor", "[10:00] supervisor asked 'what happened in the last run?' -> audit ok")
    memory.record_episode("supervisor", "[11:00] supervisor asked 'Audit models/x.pth' -> audit ok (verdict attack)")
    memory.record_episode("supervisor", "[12:00] supervisor asked 'Detect models/y.pth' -> detect ok (verdict clean)")
    memory.record_episode("supervisor", "[13:00] supervisor asked 'Repair models/y.pth' -> repair ok")

    context = memory.recall_context("what happened in the last run?", agent="supervisor")
    recent, older = context.split("Older supervisor runs")
    assert "newest first" in recent
    assert recent.index("[13:00]") < recent.index("[12:00]") < recent.index("[11:00]")
    # the word-for-word match is older than the three shown, so it goes in its own section
    assert "[10:00]" in older and "[10:00]" not in recent


def test_runs_without_specialist_calls_are_not_recorded(monkeypatch):
    _supervise(monkeypatch, "what happened in the last run?", [])
    _supervise(monkeypatch, "what happened before?", [("search_episodic_memory", {"query": "last run"})])
    assert mem.Memory.open().recall("anything", mem.EPISODIC_NAMESPACE, limit=10) == []


def test_episodes_record_tool_results_not_the_model_reply(monkeypatch):
    monkeypatch.setattr(sup, "audit_agent", _fake_specialist([]))
    _supervise(monkeypatch, "audit models/x.pth", [("audit", {"request": "Audit models/x.pth"})])
    (episode,) = mem.Memory.open().recall("x", mem.EPISODIC_NAMESPACE, agent="supervisor")
    assert "audit ok" in episode
    assert "Replied" not in episode and "summary" not in episode


def test_failed_tool_records_the_error_line_not_the_log(monkeypatch):
    blocks = json.dumps([{"type": "text", "text": (
        "Error calling tool 'run_mmbd': [cli] loading model from provider=torchvision\n"
        "[loader] epoch 1/1\nError: failed to run 'mmbd' on model models/x.pth.\nReason: CUDA out of memory"
    )}])
    monkeypatch.setattr(sup, "audit_agent", _fake_specialist([], status="error", content=blocks))
    _supervise(monkeypatch, "audit models/x.pth", [("audit", {"request": "Audit models/x.pth"})])
    (episode,) = mem.Memory.open().recall("x", mem.EPISODIC_NAMESPACE, agent="audit")
    assert "run_mmbd FAILED" in episode
    assert "Reason: CUDA out of memory" in episode
    assert "[loader]" not in episode and '"type"' not in episode


def test_no_memory_does_not_create_the_store(monkeypatch, isolated_agent_memory):
    received = []
    monkeypatch.setattr(sup, "audit_agent", _fake_specialist(received))
    monkeypatch.setattr(sup.Memory, "open", classmethod(lambda cls, path=None: pytest.fail("store opened")))
    _supervise(monkeypatch, "audit models/x.pth", [("audit", {"request": "Audit models/x.pth"})], memory=False)
    assert received == [None]
    assert not isolated_agent_memory.exists()


def test_keyword_search_ranks_matches_then_recency():
    store = mem.MemoryStore()
    ns = mem.SEMANTIC_NAMESPACE
    store.put(ns, "old", {"content": "resnet18_poison targets class 0"})
    store.put(ns, "other", {"content": "user wants reports in reports/"})
    store.put(ns, "new", {"content": "resnet18_poison was repaired with CLP"})
    store.put(ns, "newest", {"content": "unrelated note"})

    assert [i.key for i in store.search(ns, query="resnet18_poison class", limit=2)] == ["old", "new"]
    assert [i.key for i in store.search(ns, query="resnet18_poison", limit=2)] == ["new", "old"]
    # no query or no match falls back to newest first
    assert [i.key for i in store.search(ns, limit=2)] == ["newest", "new"]
    assert [i.key for i in asyncio.run(store.asearch(ns, query="zzz", limit=1))] == ["newest"]


def test_save_and_load_keep_recency(isolated_agent_memory):
    memory = mem.Memory.open()
    memory.record_episode("audit", "first episode")
    memory.record_episode("audit", "second episode")
    memory.save()

    reloaded = mem.Memory.open()
    assert reloaded.recall("anything", mem.EPISODIC_NAMESPACE, agent="audit", limit=1) == ["second episode"]


def test_save_caps_episodes(monkeypatch, isolated_agent_memory):
    monkeypatch.setattr(mem, "MAX_EPISODES", 3)
    memory = mem.Memory.open()
    for i in range(5):
        memory.record_episode("audit", f"episode {i}")
    memory.save()
    records = json.loads(isolated_agent_memory.read_text())
    assert [r["value"]["content"] for r in records] == ["episode 2", "episode 3", "episode 4"]


def test_corrupt_store_names_the_fix(isolated_agent_memory):
    isolated_agent_memory.parent.mkdir(parents=True)
    isolated_agent_memory.write_text("{not json")
    with pytest.raises(ValueError, match="--forget"):
        mem.Memory.open()


def test_forget_deletes_the_store(isolated_agent_memory):
    memory = mem.Memory.open()
    memory.record_episode("supervisor", "x")
    memory.save()
    assert mem.forget() == (isolated_agent_memory, True)
    assert not isolated_agent_memory.exists()
    assert mem.forget() == (isolated_agent_memory, False)


def test_default_path_is_repo_memory_dir(monkeypatch):
    monkeypatch.delenv("MITHRIDATIUM_MEMORY_PATH")
    assert mem.memory_path() == mem.DEFAULT_MEMORY_PATH
    assert mem.DEFAULT_MEMORY_PATH.parent.name == "memory"
    assert (mem.DEFAULT_MEMORY_PATH.parents[1] / "pyproject.toml").exists()
