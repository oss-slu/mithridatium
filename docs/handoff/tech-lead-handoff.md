# Tech Lead Handoff

## Project Status

Mithridatium is a research-oriented CLI and Python package for checking pretrained image classifiers for possible backdoors. It currently supports running one selected defense at a time:

- FreeEagle
- STRIP
- MMBD
- AEVA

The project has local PyTorch checkpoint support, initial Hugging Face image-classification support, JSON reporting, a Typer CLI, a Gradio entry point, and tests around key defense/report contracts.

## Current Architecture

The main flow is:

1. `mithridatium/cli.py` parses `mithridatium detect` options.
2. `mithridatium/loader.py` or `mithridatium/loader_hf.py` loads the model.
3. `mithridatium/utils.py` builds preprocessing config and dataloaders.
4. The selected defense runs from `mithridatium/defenses/`.
5. `mithridatium/report.py` builds and validates report JSON.

See [architecture overview](../architecture/overview.md) for diagrams.

## Major Completed Features

- Typer CLI with `detect`, `defenses`, and `ui` commands.
- Individual defense dispatch for `mmbd`, `strip`, `aeva`, and `freeeagle`.
- Local `.pt` and `.pth` checkpoint validation.
- ResNet-18 variant detection for standard and CIFAR-style checkpoints.
- Hugging Face image-classification wrapper.
- JSON report builder and schema validation.
- FreeEagle wrapper and core tests.
- STRIP entropy and threshold tests.
- Example docs for invisible and semantic backdoor demos.

## Important Files

| File                                 | Why it matters                                         |
| ------------------------------------ | ------------------------------------------------------ |
| `mithridatium/cli.py`                | Primary user entry point and defense dispatch          |
| `mithridatium/loader.py`             | Local checkpoint loading and compatibility checks      |
| `mithridatium/loader_hf.py`          | Hugging Face model wrapper                             |
| `mithridatium/utils.py`              | Dataset configs and dataloaders                        |
| `mithridatium/defenses/freeeagle.py` | FreeEagle integration wrapper                          |
| `mithridatium/defenses/strip.py`     | STRIP implementation and dynamic/static thresholds     |
| `mithridatium/defenses/mmbd.py`      | MMBD implementation                                    |
| `mithridatium/defenses/aeva.py`      | AEVA implementation and caching                        |
| `mithridatium/report.py`             | Report builder, summary renderer, JSON-safe conversion |
| `reports/report_schema.json`         | Report validation contract                             |
| `tests/`                             | Regression and contract tests                          |
| `docs/`                              | Reorganized project documentation                      |

## Common Commands

```bash
pip install -e .
pytest
mithridatium defenses
mithridatium detect --model models/resnet18_poison.pth --data cifar10 --defense mmbd --out reports/mmbd.json --force
mithridatium detect --provider huggingface --hf-model-id microsoft/resnet-50 --data cifar10_for_imagenet --defense strip --out reports/hf_strip.json --force
```

## Recommended Next Steps

- Calibrate defense thresholds on a shared benchmark set.
- Add CI coverage for fast tests.
- Decide how broad Hugging Face support should be and document supported model families.
- Add clearer benchmark model management, including expected metrics and download/training instructions.

## Handoff Checklist

- Read [known issues](known-issues.md).
- Read [future work](future-work.md).
- Run `pytest`.
- Run one local smoke test and one Hugging Face smoke test if network/model cache is available.
- Confirm docs match any new CLI flags before releasing.
