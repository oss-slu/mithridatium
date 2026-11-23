"""The benchmark model set: one poisoned ResNet-18 per attack (issue #190).

``models/benchmark.json`` lists each checkpoint with its attack, target class,
poison rate, trigger settings, clean accuracy and ASR. ``scripts/train_resnet18.py
--benchmark-name <name>`` writes the entry after training, so the numbers are
the ones the training run measured. Tuning scripts should read paths from here
instead of hardcoding them.

Checkpoints and trigger files are not committed; they live on Libra or Hugging
Face at the listed paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "models" / "benchmark.json"


def load_benchmark(path: str | Path | None = None) -> list[dict[str, Any]]:
    """All benchmark entries, or an empty list if the file doesn't exist yet."""
    target = Path(path) if path else BENCHMARK_PATH
    if not target.exists():
        return []
    return json.loads(target.read_text(encoding="utf-8"))["models"]


def get_model(name: str, path: str | Path | None = None) -> dict[str, Any]:
    """The entry called ``name``."""
    for entry in load_benchmark(path):
        if entry["name"] == name:
            return entry
    raise KeyError(f"no benchmark model named '{name}'")


def record_model(entry: dict[str, Any], path: str | Path | None = None) -> Path:
    """Add or replace the entry with the same name and write the file.

    Holds a lock while it reads and rewrites, since a Slurm array trains several
    models at once and they can finish together.
    """
    import fcntl

    if "name" not in entry:
        raise ValueError("benchmark entries need a name")
    target = Path(path) if path else BENCHMARK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        models = [m for m in load_benchmark(target) if m["name"] != entry["name"]]
        models.append(entry)
        models.sort(key=lambda m: m["name"])
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"models": models}, indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)
    return target
