"""LangMem-backed semantic and episodic memory for the agents.

Adapted from sebfroning/AI-Pipeline-Intrusion-Detection-Agent
(app/memory/langmem_store.py). Two namespaces keep recall scoped:

- semantic: facts that stay true across runs (model paths and what they are,
  dataset/architecture, user preferences, which methods suit which models).
- episodic: what happened in past runs. The supervisor records one episode per
  run and one per specialist call (which tools ran, verdicts, errors, paths).

The supervisor is the only writer. Specialists get a read-only view: recall
plus the search tools. The store is an InMemoryStore saved to JSON between
runs. Vector search is optional: set MITHRIDATIUM_MEMORY_EMBED to a
"provider:model" embedding spec (e.g. "ollama:nomic-embed-text" or
"openai:text-embedding-3-small"). Without it, search ranks by keyword overlap,
then recency, so memory works offline with any chat model.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.store.base import SearchItem
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

SEMANTIC_NAMESPACE = ("memories", "semantic")
EPISODIC_NAMESPACE = ("memories", "episodic")
NAMESPACES = {"semantic": SEMANTIC_NAMESPACE, "episodic": EPISODIC_NAMESPACE}

DEFAULT_MEMORY_PATH = Path(__file__).resolve().parents[2] / "memory" / "memory.json"

# oldest episodes are dropped on save past this; with embeddings every memory is
# re-embedded on load, so the cap also bounds startup cost
MAX_EPISODES = 500

MEMORY_RULE = (
    "Memory is context only: current tool results always win, and never report "
    "a past verdict as a current one."
)

SEMANTIC_INSTRUCTIONS = (
    "Record durable facts that stay true across runs: model identities and "
    "paths, dataset and architecture details, which methods work on which "
    "architectures, and user preferences. Do not record one-off verdicts."
)
EPISODIC_INSTRUCTIONS = (
    "Each run and specialist call is recorded automatically. Use this only to "
    "add a lesson from this run that the automatic record would miss, such as "
    "why a method failed or which settings fixed it."
)


def memory_path() -> Path:
    """MITHRIDATIUM_MEMORY_PATH, otherwise <repo>/memory/memory.json."""
    return Path(os.environ.get("MITHRIDATIUM_MEMORY_PATH") or DEFAULT_MEMORY_PATH)


def embedding_spec() -> str | None:
    return os.environ.get("MITHRIDATIUM_MEMORY_EMBED") or None


def _build_index() -> dict[str, Any] | None:
    """Vector index config, or None for keyword search."""
    spec = embedding_spec()
    if not spec:
        return None
    from langchain.embeddings import init_embeddings

    embed = init_embeddings(spec)
    # probe once so the dimension never has to be configured by hand
    return {"dims": len(embed.embed_query("dimension probe")), "embed": embed}


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "the", "on", "of", "to", "in", "for", "with", "it", "is",
    "if", "run", "check", "please", "model", "models", "pth", "pt",
}


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOPWORDS


def _text(value: dict[str, Any]) -> str:
    content = value.get("content", value)
    return content if isinstance(content, str) else json.dumps(content, default=str)


class MemoryStore(InMemoryStore):
    """InMemoryStore that ranks by keyword overlap, then recency, without an index.

    A plain InMemoryStore ignores `query` when it has no index and returns the
    oldest items first, which is the opposite of what recall needs.
    """

    def _ranked(self, namespace_prefix, query, filter, limit, offset) -> list[SearchItem]:
        items = super().search(namespace_prefix, filter=filter, limit=10**9)
        words = _tokens(query or "")
        scored = [(len(words & _tokens(_text(item.value))), item) for item in items]
        scored.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)
        return [
            SearchItem(
                namespace=item.namespace,
                key=item.key,
                value=item.value,
                created_at=item.created_at,
                updated_at=item.updated_at,
                score=float(score),
            )
            for score, item in scored[offset : offset + limit]
        ]

    def search(self, namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0, refresh_ttl=None):
        if self.index_config and query:
            return super().search(
                namespace_prefix, query=query, filter=filter, limit=limit, offset=offset, refresh_ttl=refresh_ttl
            )
        return self._ranked(namespace_prefix, query, filter, limit, offset)

    async def asearch(self, namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0, refresh_ttl=None):
        if self.index_config and query:
            return await super().asearch(
                namespace_prefix, query=query, filter=filter, limit=limit, offset=offset, refresh_ttl=refresh_ttl
            )
        return self._ranked(namespace_prefix, query, filter, limit, offset)


class Memory:
    """The store plus the tools and helpers the agents use.

    `read_only()` shares the same store, so a specialist sees what the
    supervisor recorded earlier in the same run, but it hands out no write tools
    and refuses record_episode and save.
    """

    def __init__(self, store: MemoryStore, path: Path, writable: bool = True):
        self.store = store
        self.path = path
        self.writable = writable

    @classmethod
    def open(cls, path: Path | None = None) -> "Memory":
        memory = cls(MemoryStore(index=_build_index()), Path(path) if path else memory_path())
        memory.load()
        return memory

    def read_only(self) -> "Memory":
        return Memory(self.store, self.path, writable=False)

    def _require_writable(self) -> None:
        if not self.writable:
            raise PermissionError("this memory view is read-only; only the supervisor writes memory")

    # tools

    def search_tools(self) -> list:
        return [
            create_search_memory_tool(
                namespace,
                store=self.store,
                name=f"search_{label}_memory",
                instructions=f"Searches {label} memory. {MEMORY_RULE}",
            )
            for label, namespace in NAMESPACES.items()
        ]

    def manage_tools(self) -> list:
        self._require_writable()
        return [
            create_manage_memory_tool(
                SEMANTIC_NAMESPACE, store=self.store, name="manage_semantic_memory", instructions=SEMANTIC_INSTRUCTIONS
            ),
            create_manage_memory_tool(
                EPISODIC_NAMESPACE, store=self.store, name="manage_episodic_memory", instructions=EPISODIC_INSTRUCTIONS
            ),
        ]

    def tools(self) -> list:
        """Manage + search for a writable view, search only for a read-only one."""
        return (self.manage_tools() if self.writable else []) + self.search_tools()

    # recall

    def _items(self, namespace: tuple[str, ...], query: str | None = None, agent: str | None = None) -> list[SearchItem]:
        """Ranked items; episodic ones can be scoped to one agent.

        Episodes the supervisor wrote through manage_episodic_memory have no
        agent field and count as the supervisor's.
        """
        items = self.store.search(namespace, query=query, limit=10**9)
        if agent:
            items = [i for i in items if i.value.get("agent", "supervisor") == agent]
        return items

    def recall(self, query: str, namespace: tuple[str, ...], *, agent: str | None = None, limit: int = 3) -> list[str]:
        """Best matches for `query` (newest first when nothing matches)."""
        return [_text(i.value) for i in self._items(namespace, query, agent)[:limit]]

    def recall_context(self, query: str, *, agent: str, limit: int = 3) -> str:
        """Prompt block of known facts and this agent's past runs. Empty when there are none.

        Episodes come in two labelled sections: this agent's most recent runs,
        always shown so "the last run" means the last run, then older runs that
        match the request. Mixing them in one relevance-ranked list made models
        read the top match as the latest run.
        """
        facts = self.recall(query, SEMANTIC_NAMESPACE, limit=limit)
        recent_items = sorted(self._items(EPISODIC_NAMESPACE, agent=agent), key=lambda i: i.updated_at, reverse=True)
        recent = recent_items[:limit]
        shown = {i.key for i in recent}
        related = [
            i for i in self._items(EPISODIC_NAMESPACE, query, agent)
            if i.key not in shown and (i.score or 0) > 0
        ][:limit]
        if not facts and not recent:
            return ""
        sections = []
        if facts:
            sections.append("Known facts:\n" + "\n".join(f"- {_clip(f)}" for f in facts))
        if recent:
            sections.append(
                f"Most recent {agent} runs (newest first):\n" + "\n".join(f"- {_clip(_text(i.value))}" for i in recent)
            )
        if related:
            sections.append(
                f"Older {agent} runs related to this request:\n" + "\n".join(f"- {_clip(_text(i.value))}" for i in related)
            )
        return f"\n<memories>\n{MEMORY_RULE}\n\n" + "\n\n".join(sections) + "\n</memories>\n"

    # writes

    def record_episode(self, agent: str, content: str) -> None:
        self._require_writable()
        now = datetime.now(timezone.utc)
        key = f"{now.strftime('%Y%m%dT%H%M%S%f')}-{agent}"
        self.store.put(EPISODIC_NAMESPACE, key, {"content": content, "agent": agent, "recorded_at": now.isoformat()})

    def save(self) -> None:
        """Write every memory to JSON, oldest first, so load() keeps recency order."""
        self._require_writable()
        records = []
        for namespace in NAMESPACES.values():
            items = sorted(self.store.search(namespace, limit=10**9), key=lambda i: i.updated_at)
            if namespace == EPISODIC_NAMESPACE:
                items = items[-MAX_EPISODES:]
            records.extend(
                {
                    "namespace": list(namespace),
                    "key": item.key,
                    "value": item.value,
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in items
            )
        records.sort(key=lambda r: r["updated_at"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> int:
        """Restore memories from JSON. Returns how many were loaded."""
        if not self.path.exists():
            return 0
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as ex:
            raise ValueError(
                f"agent memory at {self.path} is not valid JSON ({ex}); "
                "clear it with `mithridatium agent --forget`"
            ) from ex
        # put() stamps the current time, so insert oldest first to keep relative recency
        for record in sorted(records, key=lambda r: r.get("updated_at", "")):
            self.store.put(tuple(record["namespace"]), record["key"], record["value"])
        return len(records)


def forget(path: Path | None = None) -> tuple[Path, bool]:
    """Delete the saved store. Returns the path and whether a file was there."""
    target = Path(path) if path else memory_path()
    existed = target.exists()
    target.unlink(missing_ok=True)
    return target, existed


def _clip(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def with_memory(
    memory: Memory | None, agent: str, prompt: str, system_prompt: str, tools: list
) -> tuple[str, list]:
    """Add recalled context and this view's memory tools to an agent's setup."""
    if memory is None:
        return system_prompt, tools
    return system_prompt + memory.recall_context(prompt, agent=agent), [*tools, *memory.tools()]
