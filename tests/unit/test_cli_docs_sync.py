"""
Guardrail so defense-prefixed `audit` flags stay documented.

Each `--aeva-*`, `--freeeagle-*`, `--mmbd-*`, and `--strip-*` option on the
`audit` command must appear on that defense's page in `docs/defenses/`.
The flag list is read from the live Typer/Click command object, not a
hard-coded snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer.main

from mithridatium.cli import app


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs" / "defenses"

DEFENSE_PREFIXES = {
    "aeva": "--aeva-",
    "freeeagle": "--freeeagle-",
    "mmbd": "--mmbd-",
    "strip": "--strip-",
}


def _audit_option_names() -> list[str]:
    command = typer.main.get_command(app)
    audit = command.get_command(None, "audit")
    names: list[str] = []
    for param in audit.params:
        for opt in param.opts:
            if opt.startswith("--"):
                names.append(opt)
    return names


def _flags_for_defense(prefix: str) -> list[str]:
    return sorted({name for name in _audit_option_names() if name.startswith(prefix)})


def _doc_text(defense: str) -> str:
    path = DOCS_DIR / f"{defense}.md"
    assert path.is_file(), f"missing defense docs page: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("defense,prefix", sorted(DEFENSE_PREFIXES.items()))
def test_audit_defense_flags_are_documented(defense: str, prefix: str) -> None:
    flags = _flags_for_defense(prefix)
    docs = _doc_text(defense)
    missing = [flag for flag in flags if flag not in docs]
    assert missing == [], (
        f"{defense} CLI flags missing from docs/defenses/{defense}.md: "
        + ", ".join(missing)
    )
