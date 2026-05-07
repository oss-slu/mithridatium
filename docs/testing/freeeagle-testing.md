# FreeEagle Testing

FreeEagle is white-box and data-free. It analyzes internal ResNet behavior and does not add triggers to images.

## What to Test

FreeEagle tests should check:

- The wrapper returns the expected output contract.
- Clean random or known-clean ResNet models are not trivially flagged.
- Known poisoned benchmark checkpoints produce a suspicious `anomaly_metric`.
- Unsupported architectures fail clearly.

Relevant tests:

```bash
pytest tests/test_freeeagle.py tests/test_freeeagle_wrapper.py tests/test_freeeagle_core.py tests/test_freeeagle_reporting.py
```

## Local CLI Smoke Test

```bash
mithridatium detect \
  --model models/resnet18_poison.pth \
  --data cifar10 \
  --defense freeeagle \
  --freeeagle-optimize-steps 100 \
  --freeeagle-anomaly-threshold 2.0 \
  --out reports/freeeagle.json \
  --force
```

## Multiple Local Models

```bash
for model in models/*.pth; do
  name="$(basename "$model" .pth)"
  mithridatium detect \
    --model "$model" \
    --data cifar10 \
    --defense freeeagle \
    --freeeagle-optimize-steps 100 \
    --out "reports/${name}_freeeagle.json" \
    --force
done
```

## Output Checklist

In `results`, inspect:

- `verdict`: `likely clean` or `likely backdoored`
- `anomaly_metric`: IQR-style outlier score
- `thresholds.anomaly_metric_threshold`: default is `2.0`
- `tendency_per_target`: per-class routing tendency
- `anomaly_matrix`: class-to-class softmax score matrix
- `parameters.inspect_layer_position`: default is `2`

## Common Failure Modes

- Non-ResNet models raise `NotImplementedError`.
- Hugging Face wrapper models fail the compatibility check because feature extraction is not exposed.
- Very low `--freeeagle-optimize-steps` values are useful for smoke tests but may not be sensitive enough for real evaluation.
