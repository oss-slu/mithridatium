"""One-line progress output for agent runs, written to stderr.

A LangChain callback handler: passed in the supervisor's config, it also sees
the tool calls of the specialists the supervisor runs, since nested agents
inherit the callbacks of the tool call they run inside.
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

    def _line(self, name: str, text: str) -> None:
        indent = "" if name in SPECIALISTS else "    "
        print(f"{indent}{text}", file=self.stream, flush=True)

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self.started[run_id] = (name, time.monotonic())
        marker = "▶" if name in SPECIALISTS else "·"
        self._line(name, f"{marker} {name}: {_args(inputs, input_str)}")

    def on_tool_end(self, output, *, run_id, **kwargs):
        name, start = self.started.pop(run_id, ("tool", time.monotonic()))
        failed = getattr(output, "status", "success") == "error"
        self._line(name, f"{'✗' if failed else '✓'} {name} {'failed' if failed else 'done'} ({time.monotonic() - start:.1f}s)")

    def on_tool_error(self, error, *, run_id, **kwargs):
        name, start = self.started.pop(run_id, ("tool", time.monotonic()))
        self._line(name, f"✗ {name} failed ({time.monotonic() - start:.1f}s): {_clip(error)}")


def progress_config(progress: bool) -> dict[str, Any] | None:
    """Run config that prints progress, or None to inherit the caller's callbacks."""
    return {"callbacks": [ProgressPrinter()]} if progress else None
