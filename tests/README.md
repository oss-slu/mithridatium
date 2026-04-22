# Mithridatium Test Suite

This folder contains the automated tests for Mithridatium. The tests are organized by purpose so that developers can quickly run lightweight checks during development, while still having deeper integration tests available when needed.

## Test Folder Structure

```text
tests/
├── conftest.py
├── smoke/
├── unit/
└── integration/
```

## Test Categories

### Smoke Tests

Location:

```text
tests/smoke/
```

Smoke tests are the fastest high-level checks. They verify that the project can import successfully, the CLI starts, and the expected commands are available.

These tests should not require:

- real datasets
- real model checkpoints
- Hugging Face downloads
- long-running defenses

Run smoke tests with:

```bash
python -m pytest tests/smoke -q
```

Use smoke tests when you want to quickly confirm that the package is not obviously broken.

---

### Unit Tests

Location:

```text
tests/unit/
```

Unit tests check small pieces of functionality in isolation. These tests should use small synthetic inputs, mocks, or temporary files instead of real datasets and checkpoints.

Examples of unit-tested behavior:

- preprocessing config values
- STRIP entropy calculations
- STRIP score output contracts
- FreeEagle core output shape/contracts
- invisible trigger helper functions
- CLI routing with mocked model loading and mocked defense execution

Run unit tests with:

```bash
python -m pytest tests/unit -q
```

Unit tests should stay fast and deterministic.

---

### Integration Tests

Location:

```text
tests/integration/
```

Integration tests verify that multiple real components work together. These tests may use real CIFAR-10 data, real checkpoint loading, real CLI commands, or real defense execution.

Examples of integration-tested behavior:

- CIFAR-10 dataloader creation and normalization
- evaluator + model + DataLoader interaction
- local checkpoint save/load behavior
- Hugging Face wrapper contract
- CLI demo script execution
- optional slow FreeEagle detection against a real checkpoint

Run integration tests with:

```bash
python -m pytest tests/integration -q
```

Some integration tests are marked as slow or require local files.

---

## Shared Fixtures

Shared fixtures live in:

```text
tests/conftest.py
```

This file defines shared project paths and reusable pytest fixtures, such as:

```text
project_root
data_root
models_root
cifar10_dir
require_cifar10
require_models_dir
```

Use these fixtures instead of manually recalculating paths inside individual tests.

For example:

```python
def test_cifar10_dataloader_creation(require_cifar10):
    ...
```

The `require_cifar10` fixture skips the test if the local CIFAR-10 dataset is missing.

Expected CIFAR-10 location:

```text
data/cifar-10-batches-py/
```

---

## Pytest Markers

Markers are declared in `pyproject.toml`.

Current markers:

```text
unit
integration
smoke
slow
requires_data
requires_model
requires_hf
```

Marker meanings:

| Marker           | Meaning                                                       |
| ---------------- | ------------------------------------------------------------- |
| `unit`           | Fast tests for isolated functions/classes                     |
| `integration`    | Tests that exercise multiple modules together                 |
| `smoke`          | Quick end-to-end sanity checks                                |
| `slow`           | Tests that may take longer to run                             |
| `requires_data`  | Tests that require local or downloadable datasets             |
| `requires_model` | Tests that require local checkpoint files                     |
| `requires_hf`    | Tests that require Hugging Face downloads or cached HF models |

---

## Common Test Commands

Run all tests:

```bash
python -m pytest -q
```

Run only smoke tests:

```bash
python -m pytest tests/smoke -q
```

Run only unit tests:

```bash
python -m pytest tests/unit -q
```

Run only integration tests:

```bash
python -m pytest tests/integration -q
```

Run fast tests only:

```bash
python -m pytest tests/smoke tests/unit -q
```

Run everything except slow tests:

```bash
python -m pytest -m "not slow" -q
```

Run tests that require real data:

```bash
python -m pytest -m requires_data -q
```

Run tests that require real model checkpoints:

```bash
python -m pytest -m requires_model -q
```

---

## Recommended Development Workflow

After making a small code change, run:

```bash
python -m pytest tests/smoke tests/unit -q
```

After changing loaders, preprocessing, CLI routing, or report generation, run:

```bash
python -m pytest tests/integration -q
```

Before handing off or merging larger changes, run:

```bash
python -m pytest -q
```

If slow tests are too expensive, run:

```bash
python -m pytest -m "not slow" -q
```

---

## Dependency and Install Notes

Install the project in editable mode before running tests:

```bash
python -m pip install -e ".[dev]"
```

This makes the `mithridatium` package importable and exposes the CLI command:

```bash
mithridatium --version
mithridatium defenses
```

If using `requirements.txt`, install dependencies first:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Always prefer:

```bash
python -m pytest
```

instead of plain:

```bash
pytest
```

This ensures tests run using the active Python environment.

---

## Notes for Future Contributors

Keep the test categories separate:

- Put isolated logic tests in `tests/unit/`
- Put CLI import and basic command checks in `tests/smoke/`
- Put real dataloader, checkpoint, script, and defense pipeline tests in `tests/integration/`

Avoid adding real dataset downloads, real checkpoint requirements, or long defense runs to unit or smoke tests.

If a test depends on local files, mark it clearly with:

```python
pytestmark = [pytest.mark.integration, pytest.mark.requires_data]
```

or:

```python
pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.requires_model]
```

This keeps the test suite predictable and easy to run.
