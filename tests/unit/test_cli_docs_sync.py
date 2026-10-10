"""
Guardrail so CLI flags stay documented.

- Each `--aeva-*`, `--freeeagle-*`, `--mmbd-*`, and `--strip-*` option on the
  `audit` command must appear on that defense's page in `docs/defenses/`.
- Each `--*` option on `detect` must appear in `docs/detect.md`.
- Each `--*` option on `repair` must appear in `docs/repair.md`.

The flag lists are read from the live Typer/Click command objects, not
hard-coded snapshots.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer.main

from mithridatium.cli import app


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFENSES_DOCS_DIR = PROJECT_ROOT / "docs" / "defenses"

DEFENSE_PREFIXES = {
    "aeva": "--aeva-",
    "freeeagle": "--freeeagle-",
    "mmbd": "--mmbd-",
    "strip": "--strip-",
}

COMMAND_DOCS_MAP = {
    "detect": PROJECT_ROOT / "docs" / "detect.md",
    "repair": PROJECT_ROOT / "docs" / "repair.md",
}


def _command_option_names(command_name: str) -> list[str]:
    root = typer.main.get_command(app)
    subcommand = root.get_command(None, command_name)
    assert subcommand is not None, f"subcommand '{command_name}' not found on app"
    names: list[str] = []
    for param in subcommand.params:
        for opt in param.opts:
            if opt.startswith("--"):
                names.append(opt)
    return sorted(set(names))


def _audit_option_names() -> list[str]:
    return _command_option_names("audit")


def _flags_for_defense(prefix: str) -> list[str]:
    return sorted({name for name in _audit_option_names() if name.startswith(prefix)})


def _defense_doc_text(defense: str) -> str:
    path = DEFENSES_DOCS_DIR / f"{defense}.md"
    assert path.is_file(), f"missing defense docs page: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("defense,prefix", sorted(DEFENSE_PREFIXES.items()))
def test_audit_defense_flags_are_documented(defense: str, prefix: str) -> None:
    flags = _flags_for_defense(prefix)
    docs = _defense_doc_text(defense)
    missing = [flag for flag in flags if flag not in docs]
    assert missing == [], (
        f"{defense} CLI flags missing from docs/defenses/{defense}.md: "
        + ", ".join(missing)
    )


@pytest.mark.parametrize("command_name,doc_path", sorted(COMMAND_DOCS_MAP.items()))
def test_command_flags_are_documented(command_name: str, doc_path: Path) -> None:
    flags = _command_option_names(command_name)
    assert doc_path.is_file(), f"missing docs page for {command_name}: {doc_path}"
    docs = doc_path.read_text(encoding="utf-8")
    missing = [flag for flag in flags if flag not in docs]
    assert missing == [], (
        f"'{command_name}' CLI flags missing from {doc_path.relative_to(PROJECT_ROOT)}: "
        + ", ".join(missing)
    )
