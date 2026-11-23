"""One-line progress output for agent runs, written to stderr.

A LangChain callback handler: passed in the supervisor's config, it also sees
the tool calls of the specialists the supervisor runs, since nested agents
inherit the callbacks of the tool call they run inside. Each nested tool line
is tagged with the specialist it runs under ([audit], [detect], [repair]), found
by walking the parent run ids, so specialists running in parallel stay readable.
"""

from __future__ import annotations

import sys
import time
from typing import Any, TextIO

from langchain_core.callbacks import BaseCallbackHandler

SPECIALISTS = {"audit", "detect", "repair"}


def _clip(text: Any, limit: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _args(inputs: dict[str, Any] | None, fallback: str) -> str:
    if not inputs:
        return _clip(fallback)
    if set(inputs) == {"request"}:
        return _clip(inputs["request"])
    return _clip(", ".join(f"{k}={v}" for k, v in inputs.items() if k != "runtime"))


class ProgressPrinter(BaseCallbackHandler):
    """Prints a line when each tool starts and when it finishes or fails."""

    def __init__(self, stream: TextIO | None = None):
        self.stream = stream or sys.stderr
        self.started: dict[Any, tuple[str, float]] = {}
        # run id -> parent run id, for every chain and tool run seen
        self.parents: dict[Any, Any] = {}
        # run id -> specialist name, for the supervisor's audit/detect/repair calls
        self.specialist_runs: dict[Any, str] = {}

    def _specialist_for(self, run_id: Any) -> str | None:
        """The specialist call this run is nested under, if any."""
        seen = set()
        run_id = self.parents.get(run_id)
        while run_id is not None and run_id not in seen:
            if run_id in self.specialist_runs:
                return self.specialist_runs[run_id]
            seen.add(run_id)
            run_id = self.parents.get(run_id)
        return None

    def _line(self, name: str, run_id: Any, text: str) -> None:
        if name in SPECIALISTS:
            print(text, file=self.stream, flush=True)
            return
        owner = self._specialist_for(run_id)
        tag = f"[{owner}] " if owner else ""
        print(f"    {text[:2]}{tag}{text[2:]}", file=self.stream, flush=True)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self.parents[run_id] = parent_run_id

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self.parents[run_id] = parent_run_id
        self.started[run_id] = (name, time.monotonic())
        if name in SPECIALISTS:
            self.specialist_runs[run_id] = name
        marker = "▶" if name in SPECIALISTS else "·"
        self._line(name, run_id, f"{marker} {name}: {_args(inputs, input_str)}")

    def on_tool_end(self, output, *, run_id, **kwargs):
        name, start = self.started.pop(run_id, ("tool", time.monotonic()))
        failed = getattr(output, "status", "success") == "error"
        status = "failed" if failed else "done"
        self._line(name, run_id, f"{'✗' if failed else '✓'} {name} {status} ({time.monotonic() - start:.1f}s)")

    def on_tool_error(self, error, *, run_id, **kwargs):
        name, start = self.started.pop(run_id, ("tool", time.monotonic()))
        self._line(name, run_id, f"✗ {name} failed ({time.monotonic() - start:.1f}s): {_clip(error)}")


def progress_config(progress: bool) -> dict[str, Any] | None:
    """Run config that prints progress, or None to inherit the caller's callbacks."""
    return {"callbacks": [ProgressPrinter()]} if progress else None
