"""
Smoke tests for core Mithridatium imports.

These tests are intentionally lightweight. They verify that the main package
modules import successfully after installation or code changes.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.smoke


def test_core_modules_import():
    import mithridatium
    import mithridatium.cli
    import mithridatium.loader
    import mithridatium.utils
    import mithridatium.evaluator
    import mithridatium.report
    import mithridatium.service

    assert mithridatium is not None


@pytest.mark.requires_hf
def test_huggingface_loader_imports_when_transformers_installed():
    # loader_hf depends on the optional `[hf]` extra. CI smoke+unit does not
    # install it; skip instead of treating Hugging Face as a required import.
    pytest.importorskip("transformers")
    import mithridatium.loader_hf

    assert mithridatium.loader_hf is not None


def test_defense_modules_import():
    import mithridatium.defenses.mmbd
    import mithridatium.defenses.strip
    import mithridatium.defenses.aeva
    import mithridatium.defenses.freeeagle
    import mithridatium.defenses._freeeagle_core

    assert mithridatium.defenses.mmbd is not None
    assert mithridatium.defenses.strip is not None
    assert mithridatium.defenses.aeva is not None
    assert mithridatium.defenses.freeeagle is not None
    assert mithridatium.defenses._freeeagle_core is not None


def test_attack_modules_import():
    import mithridatium.attacks.invisible
    import mithridatium.attacks.semantic

    assert mithridatium.attacks.invisible is not None
    assert mithridatium.attacks.semantic is not None