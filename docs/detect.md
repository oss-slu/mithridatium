# Detect (input-level methods)

## Commands at a glance

| Command | Purpose |
|---|---|
| `mithridatium audit` | **Model-level defenses** (FreeEagle, STRIP, MMBD, AEVA). One defense per run; JSON verdict report. |
| `mithridatium detect` | **Input-level detection** (e.g. SCALE-UP). Flags suspicious inputs; separate from audit defenses. |
| `mithridatium repair` | **Repair** a backdoored checkpoint after you know the model is compromised. |

## When to use detect vs audit

- Use **`audit`** when you want a defense-style verdict on whether the **model** looks backdoored (spectral / activation / runtime probes).
- Use **`detect`** when you want **per-input** signals (poisoned or triggered samples in a batch or dataset slice).
- Methods and rationale for detect are summarized in [Research: inference-time detection](research/detection-additions.md).

SCALE-UP scoring is not fully implemented yet; `--method scaleup` runs a stub that validates CLI options and writes a schema-checked JSON report.

## Example command

```bash
mithridatium detect \
  --model models/resnet18_poison.pth \
  --method scaleup \
  --data cifar10 \
  --scaleup-num-samples 1000 \
  --scaleup-threshold 1.0 \
  --out reports/detect_scaleup.json \
  --force
```

## Useful flags

| Flag | Role |
|---|---|
| `--model` / `-m` | Local checkpoint (`.pth` / `.pt`). |
| `--method` / `-M` | Detection method (`scaleup` today). |
| `--data` / `-d` | Dataset name for preprocessing (e.g. `cifar10`). |
| `--scaleup-num-samples` / `-n` | Number of test images to score (when implemented). |
| `--scaleup-threshold` | Fraction of scale factors an input must satisfy (0–1]. |
| `--scaleup-scales` | Tuple of pixel scale factors, e.g. `(3, 5, 7, 9, 11)`. |
| `--out` / `-o` | JSON report path, or `-` for stdout. |
| `--force` / `-f` | Overwrite an existing report file. |

More copy-paste examples: [Sample commands](testing/sample-commands.md).
