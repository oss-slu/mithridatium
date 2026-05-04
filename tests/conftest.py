"""
Shared pytest fixtures and helpers for Mithridatium tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"

CIFAR10_DIR = DATA_ROOT / "cifar-10-batches-py"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unit: fast tests for isolated functions/classes",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests that exercise multiple modules together",
    )
    config.addinivalue_line(
        "markers",
        "smoke: quick end-to-end sanity tests",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that may take longer to run",
    )
    config.addinivalue_line(
        "markers",
        "requires_data: tests that require local or downloadable datasets",
    )
    config.addinivalue_line(
        "markers",
        "requires_model: tests that require local model checkpoint files",
    )
    config.addinivalue_line(
        "markers",
        "requires_hf: tests that require Hugging Face downloads or cached HF models",
    )


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def data_root() -> Path:
    return DATA_ROOT


@pytest.fixture
def models_root() -> Path:
    return MODELS_ROOT


@pytest.fixture
def cifar10_dir() -> Path:
    return CIFAR10_DIR


@pytest.fixture
def require_cifar10(cifar10_dir: Path) -> Path:
    if not cifar10_dir.exists():
        pytest.skip(
            "CIFAR-10 data not found. Expected local dataset at "
            f"{cifar10_dir}. Run: python -m scripts.download_cifar10"
        )
    return cifar10_dir


@pytest.fixture
def require_models_dir(models_root: Path) -> Path:
    if not models_root.exists():
        pytest.skip(f"models directory not found at {models_root}")
    return models_root