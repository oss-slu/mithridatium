# Testing Overview

Testing in Mithridatium has two layers:

- Unit tests for contracts, helper functions, and report structure.
- CLI/manual tests against local or Hugging Face models.

## Install for Testing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[hf]"      # Hugging Face model loading
pip install -e ".[ui]"      # Streamlit and Gradio UI dependencies
pip install -e ".[all]"     # Development, HF, UI, and demo dependencies
```

## Run the Test Suite

```bash
pytest
```

Run a focused subset:

```bash
pytest tests/test_freeeagle.py tests/test_freeeagle_wrapper.py
pytest tests/test_strip_scores.py tests/test_strip_entropy.py
pytest tests/tests_report.py
```

Some tests skip when local benchmark checkpoints are not available.

## Test Local Models

Train or provide a checkpoint, then run one defense at a time:

```bash
mithridatium detect \
  --model models/resnet18_poison.pth \
  --data cifar10 \
  --defense mmbd \
  --out reports/mmbd.json \
  --force
```

Use a known clean model and a known poisoned model when possible. A defense is easier to evaluate when you can compare both reports.

## Test Hugging Face Models

Use `--provider huggingface` and a model that loads through `AutoModelForImageClassification`:

```bash
mithridatium detect \
  --provider huggingface \
  --hf-model-id microsoft/resnet-50 \
  --data cifar10_for_imagenet \
  --defense strip \
  --out reports/hf_strip.json \
  --force
```

Hugging Face model support may depend on architecture compatibility and preprocessing. FreeEagle is not currently compatible with the Hugging Face wrapper because the wrapper does not expose stable internal ResNet stages.

## Interpret Outputs

The report JSON has top-level metadata and a defense-specific `results` object. Start with:

- `results.verdict`
- `results.thresholds`
- the defense's key score, such as `anomaly_metric`, `p_value`, `entropy_mean`, or `suspicion_score`
- `results.parameters`, to confirm what settings were used

Treat a single report as a signal, not a complete proof. Thresholds may need calibration for a new dataset, architecture, or preprocessing pipeline.

## Related Docs

- [FreeEagle testing](freeeagle-testing.md)
- [Hugging Face model testing](huggingface-model-testing.md)
- [Sample commands](sample-commands.md)
