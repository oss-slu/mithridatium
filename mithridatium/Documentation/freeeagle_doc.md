# FreeEagle Defense — Mithridatium

## Overview

FreeEagle detects backdoors by inspecting the internal embedding space of a model. It optimizes dummy embeddings to activate each target class via the model's classification head, then measures whether any class attracts an abnormally disproportionate share of softmax probability from all others. A backdoored model exhibits a single dominant target class in this embedding-space analysis, producing an elevated anomaly metric.

FreeEagle operates entirely in embedding space — it requires no input data and no dataset access.

## Method

FreeEagle proceeds as follows:

1. The model is split at a configurable ResNet stage (`inspect_layer_position`). The layers before the split point form the feature extractor; the layers after form the model head.
2. For each class, Adam optimization is run to find a dummy embedding that maximizes the head's softmax confidence for that class. Optimization is bounded (values clamped to `[0, 999]` by default).
3. Each dummy embedding is passed through the model head. The softmax scores over all classes are recorded, with the source class zeroed out, forming one row of the `anomaly_matrix` (shape: `[num_classes, num_classes]`).
4. The tendency per target class is computed as the column-wise average of the anomaly matrix, scaled by `num_classes / (num_classes - 1)`.
5. The anomaly metric is derived from the tendency array using an IQR-based outlier score:

   ```
   anomaly_metric = (max(tendency) - Q3) / IQR
   ```

6. If `anomaly_metric >= threshold` (default: 2.0), the model is flagged as likely backdoored.

Optionally, transpose correction can be applied to suppress symmetric mutual attraction between class pairs before computing tendency.

## Key Outputs

| Field                                 | Description                                                     |
| ------------------------------------- | --------------------------------------------------------------- |
| `anomaly_metric`                      | IQR-based outlier score over tendency values                    |
| `tendency_per_target`                 | Per-class tendency vector (length = num_classes)                |
| `anomaly_matrix`                      | num_classes × num_classes matrix of softmax scores              |
| `verdict`                             | `"likely clean"` or `"likely backdoored"`                       |
| `thresholds.anomaly_metric_threshold` | Decision threshold (default: 2.0)                               |
| `parameters`                          | Full record of all optimization and architectural settings used |

## CLI Usage

Base command:

```bash
mithridatium detect \
  --model models/resnet18.pth \
  --data cifar10 \
  --defense freeeagle \
  --out reports/freeeagle.json
```

FreeEagle exposes several tunable parameters via the CLI:

```bash
mithridatium detect \
  --model models/resnet18.pth \
  --data cifar10 \
  --defense freeeagle \
  --freeeagle-optimize-steps 500 \
  --freeeagle-anomaly-threshold 2.0 \
  --freeeagle-inspect-layer-position 2 \
  --out reports/freeeagle.json
```

## CLI Parameters

General parameters:

| Flag             | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| `--model, -m`    | Path to model checkpoint (.pth)                                |
| `--provider, -p` | `torchvision` or `huggingface`                                 |
| `--hf-model-id`  | Hugging Face model ID (required when `--provider huggingface`) |
| `--data, -d`     | Dataset name (e.g., `cifar10`)                                 |
| `--defense, -D`  | Must be `freeeagle`                                            |
| `--arch, -a`     | Architecture hint (used for model validation)                  |
| `--out, -o`      | Output JSON path; use `-` for stdout                           |
| `--force, -f`    | Overwrite existing output file                                 |

FreeEagle-specific parameters:

| Flag                                             | Default         | Description                                                 |
| ------------------------------------------------ | --------------- | ----------------------------------------------------------- |
| `--freeeagle-num-classes`                        | 0 (auto-infer)  | Override number of classes. Use 0 to infer from model head. |
| `--freeeagle-num-dummy`                          | 1               | Number of dummy embedding vectors optimized per class.      |
| `--freeeagle-num-important-neurons`              | 5               | Top neurons tracked during optimization (informational).    |
| `--freeeagle-metric`                             | `softmax_score` | Anomaly metric to use. Also supported: `logit`.             |
| `--freeeagle-use-transpose-correction`           | false           | Enable symmetric suppression correction.                    |
| `--freeeagle-bound-on / --freeeagle-no-bound-on` | true            | Clamp embeddings to `[0, 999]` during optimization.         |
| `--freeeagle-optimize-steps`                     | 300             | Adam optimization steps per class.                          |
| `--freeeagle-learning-rate`                      | 0.01            | Adam learning rate.                                         |
| `--freeeagle-weight-decay`                       | 0.005           | Adam weight decay.                                          |
| `--freeeagle-anomaly-threshold`                  | 2.0             | Threshold for verdict.                                      |
| `--freeeagle-inspect-layer-position`             | 2               | ResNet stage index to split at (0–4).                       |

The `inspect_layer_position` maps to ResNet stages as follows:

| Position | Stage                        |
| -------- | ---------------------------- |
| 0        | conv1 + bn1 + relu + maxpool |
| 1        | layer1                       |
| 2        | layer2 (default)             |
| 3        | layer3                       |
| 4        | layer4                       |

## Architecture Support

FreeEagle currently supports **ResNet-family models only** (`model.__class__.__name__ == "ResNet"`). Passing any other architecture raises `NotImplementedError`. This includes all standard torchvision ResNet variants (resnet18, resnet18_cifar, resnet34) loaded via Mithridatium's `loader`.

## Output Format

```json
{
  "defense": "freeeagle",
  "anomaly_metric": float,
  "anomaly_matrix": [[float, ...], ...],
  "tendency_per_target": [float, ...],
  "verdict": "likely clean | likely backdoored",
  "thresholds": {
    "anomaly_metric_threshold": float
  },
  "parameters": {
    "num_classes": int,
    "inspect_layer_position": int,
    "num_dummy": int,
    "num_important_neurons": int,
    "metric": "softmax_score | logit",
    "use_transpose_correction": bool,
    "bound_on": bool,
    "optimize_steps": int,
    "learning_rate": float,
    "weight_decay": float,
    "input_shape": [int, int, int]
  },
  "dataset": "cifar10"
}
```

## Interpretation

- `anomaly_metric >= 2.0` → `"likely backdoored"`. The default threshold of 2.0 is a starting point; it can be raised to reduce false positives.
- A dominant spike in `tendency_per_target` for a single class indicates that dummy embeddings optimized for all other classes still route substantial softmax mass toward that class — consistent with a backdoor target.
- Broadly distributed tendency values with no clear outlier indicate a clean model.
- A higher `--freeeagle-optimize-steps` value gives the optimizer more iterations to find the embedding that maximally activates each class, which can improve sensitivity. The example results below used 500 steps.

## Example Results

Clean model (`resnet18_cifar10.pt`):

```
anomaly_metric:  0.679   → below threshold 2.0
targets_scored:  10
inspect_layer:   2
optimize_steps:  500
verdict:         likely clean
```

Backdoored model (`resnet18_poison.pt`):

```
anomaly_metric:  5.884   → well above threshold 2.0
targets_scored:  10
inspect_layer:   2
optimize_steps:  500
verdict:         likely backdoored
```

## Limitations

- **ResNet only.** Other architectures are not currently supported.
- FreeEagle does not identify which class is the backdoor target — only whether a dominant class exists. Inspect `tendency_per_target` manually to identify the candidate target.
- Results are sensitive to `inspect_layer_position`. Layer 2 is the default and generally performs well for ResNet-18. Deeper layers (3–4) may produce different sensitivity characteristics.
- Because optimization is stochastic, results may vary slightly across runs unless a fixed seed is set in the optimizer (not currently exposed as a CLI flag).
- FreeEagle requires no dataset, but does require the model to be a correctly loaded and functional ResNet. Model validation (dry forward pass) is run by the CLI before FreeEagle is invoked.

## Summary

FreeEagle detects backdoors by analyzing the model's internal embedding space without requiring any input data. It is faster than AEVA and more architecturally interpretable than MMBD, but is currently limited to ResNet models. The `tendency_per_target` vector and `anomaly_matrix` provide fine-grained detail beyond the scalar verdict for further investigation.

## Graphs

**Tendency per target class** — the core FreeEagle signal. Each bar shows how much softmax mass from all other classes routes toward that class in embedding space. A clean model shows a flat, distributed tendency; a backdoored model has one dominant class. The anomaly metric (IQR-based outlier score) is annotated on each panel.

![FreeEagle tendency per class](graphs/freeeagle_tendency.png)

**Anomaly matrix (backdoored model)** — heatmap of the full num_classes × num_classes softmax score matrix. Each row is a source class; each column is a target class. An elevated column at the backdoor target indicates that all source embeddings route toward it.

![FreeEagle anomaly matrix](graphs/freeeagle_anomaly_matrix.png)
