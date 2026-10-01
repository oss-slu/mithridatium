"""
Fail when a defense-prefixed audit CLI flag is missing from docs/defenses/.

The flag list is read from the live Typer app rather than a hard-coded list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer.main

from mithridatium.cli import app


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DEFENSES = REPO_ROOT / "docs" / "defenses"

DEFENSE_PREFIXES = ("aeva", "freeeagle", "mmbd", "strip")


def _audit_option_flags() -> list[str]:
    command = typer.main.get_command(app)
    audit = command.commands["audit"]
    flags: list[str] = []
    for param in audit.params:
        for opt in param.opts:
            if opt.startswith("--"):
                flags.append(opt)
    return flags


def _defense_flags_by_prefix() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {prefix: [] for prefix in DEFENSE_PREFIXES}
    for flag in _audit_option_flags():
        name = flag[2:]
        for prefix in DEFENSE_PREFIXES:
            if name == prefix or name.startswith(f"{prefix}-"):
                grouped[prefix].append(flag)
                break
    return grouped


def test_defense_docs_mention_all_prefixed_audit_flags() -> None:
    missing: dict[str, list[str]] = {}
    grouped = _defense_flags_by_prefix()

    for prefix, flags in grouped.items():
        page = DOCS_DEFENSES / f"{prefix}.md"
        assert page.is_file(), f"expected defense docs page at {page}"
        text = page.read_text(encoding="utf-8")
        absent = [flag for flag in flags if flag not in text]
        if absent:
            missing[prefix] = absent

    assert not missing, (
        "Defense docs are missing audit CLI flags:\n"
        + "\n".join(
            f"  {prefix}: {', '.join(flags)}" for prefix, flags in missing.items()
        )
    )
