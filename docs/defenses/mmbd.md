# MMBD

MMBD looks for abnormal class dominance or suspicious scoring behavior by optimizing synthetic inputs toward target classes and comparing class-wise scores.

## What It Does

The implementation creates random synthetic images for a subset of classes, optimizes them to maximize each target class response, then checks whether one class has an unusually dominant score. It reports raw per-class scores, MAD-normalized scores, and a gamma-distribution p-value.

```mermaid
flowchart TD
    A[Loaded model] --> B[Infer number of classes]
    B --> C[Choose classes to probe]
    C --> D[Optimize synthetic images per class]
    D --> E[Collect per-class scores]
    E --> F[Compute normalized scores and p-value]
    F --> G[Return verdict]
```

## Access Needed

- Model logits.
- Gradients with respect to synthetic inputs.
- No real dataset is required, but a preprocessing config is used for input shape and normalization.

## Inputs

- A compatible local or Hugging Face image classifier.
- Dataset/preprocessing selection.
- Model output class count must be inferable.

## Outputs

Important `results` fields:

| Field | Meaning |
| --- | --- |
| `per_class_scores` | Raw optimized scores for probed classes |
| `normalized_scores` | MAD-normalized deviation scores |
| `classes_probed` | List of class indices that were probed |
| `p_value` | Statistical anomaly score |
| `top_eigenvalue` | Maximum class score alias used by existing schema/reporting |
| `verdict` | `Likely clean` or `Likely backdoored` |
| `thresholds` | p-value and normalized-score interpretation bands |

## Example CLI Commands

Check default classes (first 5 classes):

```bash
mithridatium audit \
  --model models/resnet18_poison.pth \
  --data cifar10 \
  --defense mmbd \
  --out reports/mmbd.json \
  --force
```

Check specific target classes:

```bash
mithridatium audit \
  --model models/resnet18_poison.pth \
  --data cifar10 \
  --defense mmbd \
  --mmbd-classes 0,3,7 \
  --out reports/mmbd.json \
  --force
```

Check all classes:

```bash
mithridatium audit \
  --model models/resnet18_poison.pth \
  --data cifar10 \
  --defense mmbd \
  --mmbd-classes all \
  --out reports/mmbd.json \
  --force
```

Hugging Face example:

```bash
mithridatium audit \
  --provider huggingface \
  --hf-model-id microsoft/resnet-50 \
  --data cifar10_for_imagenet \
  --defense mmbd \
  --mmbd-classes 0,1,2 \
  --out reports/hf_mmbd.json \
  --force
```

## Probing Classes (`--mmbd-classes`)

By default, MMBD probes only the first `min(5, NC)` classes (`0` to `4` on CIFAR-10). If a backdoor attack targets a class outside this default range, MMBD can miss it.

The `--mmbd-classes` option allows users to configure exactly which classes are probed:

- **Comma-separated indices:** `--mmbd-classes 0,3,7` checks only classes 0, 3, and 7.
- **All classes:** `--mmbd-classes all` probes every class in the model.
- **Default (no flag):** probes the first `min(5, NC)` classes.

### Runtime Cost Considerations

MMBD optimizes synthetic images for each probed class across 75 optimization iterations (`NSTEP = 75`) with 30 synthetic images per class. Because optimization runs sequentially per class, execution time scales directly with the number of probed classes:

- On a 10-class dataset like CIFAR-10, probing all classes (`--mmbd-classes all`) doubles the runtime compared to the default 5 classes.
- On large datasets like ImageNet (1,000 classes), probing all classes requires 1,000 optimization passes and is computationally prohibitive.
- Using targeted probing (e.g. `--mmbd-classes 0,3,7`) allows efficient verification of suspected target classes without unnecessary computation.

## Current Internal Defaults

These settings configure MMBD behavior:

| Setting | Current value |
| --- | --- |
| `NSTEP` | `75` |
| `N_CLASSES_TO_PROBE` | `min(5, NC)` (default when `--mmbd-classes` is omitted) |
| `NUM_IMAGES` | `30` |
| optimizer | `SGD(lr=1e-2, momentum=0.9)` |
| p-value threshold | `0.05` |

## Strengths

- Does not require a real dataset.
- Looks for class dominance directly in model scoring behavior.
- Produces both raw scores and normalized scores.
- Can run on non-ResNet models if logits and gradients behave as expected.

## Limitations

- Probes only a subset of classes by default, so it can miss a target outside that subset.
- More computationally expensive than simple forward-pass methods.
- Requires gradients and a compatible PyTorch model path.
- The statistical fit can be unreliable with very few probed classes or unusual score distributions.
- Does not recover the trigger pattern.

## Interpretation

- `p_value < 0.05` means `Likely backdoored`.
- A `normalized_scores` value above `5.0` is very suspicious.
- A single dominant `per_class_scores` value is the main pattern to investigate.

## Graphs

![MMBD per-class activation scores](../assets/graphs/defenses/mmbd_per_class_scores.png)

![MMBD normalized deviation scores](../assets/graphs/defenses/mmbd_normalized_scores.png)

## Related Docs

- [Testing overview](../testing/overview.md)
- [Reporting pipeline](../architecture/reporting-pipeline.md)
