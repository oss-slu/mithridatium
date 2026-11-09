# Detect (input-level methods)

## Commands at a glance

| Command | Purpose |
|---|---|
| `mithridatium audit` | **Model-level defenses** (FreeEagle, STRIP, MMBD, AEVA). One defense per run; JSON verdict report. |
| `mithridatium detect` | **Input-level detection** (SCALE-UP, TED, IBD-PSC). Flags suspicious inputs; separate from audit defenses. |
| `mithridatium repair` | **Repair** a backdoored checkpoint after you know the model is compromised. |

## When to use detect vs audit

- Use **`audit`** when you want a defense-style verdict on whether the **model** looks backdoored (spectral / activation / runtime probes).
- Use **`detect`** when you want **per-input** signals (poisoned or triggered samples in a batch or dataset slice).
- Methods and rationale for detect are summarized in [Research: inference-time detection](research/detection-additions.md).

Three methods: [SCALE-UP](defenses/scaleup.md) (`--method scaleup`, black-box), [TED](defenses/ted.md) (`--method ted`) and [IBD-PSC](defenses/ibdpsc.md) (`--method ibdpsc`).

TED needs model internals and clean input data at inference time; SCALE-UP needs neither. IBD-PSC needs a model with BatchNorm layers and about 100 clean images.

## Example command

```bash
mithridatium detect \
  --model models/resnet18_poison.pth \
  --method scaleup \
  --data cifar10 \
  --scaleup-num-samples 1000 \
  --out reports/detect_scaleup.json \
  --force

mithridatium detect \
  --model models/resnet18_poison.pth \
  --method ted \
  --data cifar10 \
  --out reports/detect_ted.json

mithridatium detect \
  --model models/resnet18_poison.pth \
  --method ibdpsc \
  --data cifar10 \
  --out reports/detect_ibdpsc.json
```

## Useful flags

| Flag | Role |
|---|---|
| `--model` / `-m` | Local checkpoint (`.pth` / `.pt`). |
| `--method` / `-M` | Detection method: `scaleup`, `ted` or `ibdpsc`. |
| `--data` / `-d` | Dataset name for preprocessing (e.g. `cifar10`). |
| `--scaleup-num-samples` / `-n` | SCALE-UP: number of test images to score. |
| `--scaleup-threshold` | SCALE-UP: flag an input when its SPC exceeds this, within (0, 1]. |
| `--scaleup-scales` | SCALE-UP: tuple of pixel scale factors, e.g. `(3, 5, 7, 9, 11)`. |
| `--ted-num-samples` | TED: number of test images to score. |
| `--ted-reference-size` | TED: cap on clean reference images from the training split. |
| `--ted-contamination` | TED: fraction of clean reference inputs the threshold rejects, within (0, 0.5]. |
| `--ibdpsc-num-samples` | IBD-PSC: number of test images to score. |
| `--ibdpsc-clean-samples` | IBD-PSC: clean training images that choose which BN layers to amplify. |
| `--ibdpsc-scale` | IBD-PSC: factor on the amplified BN layers' gamma and beta, a finite number above 1. |
| `--ibdpsc-num-models` | IBD-PSC: number of amplified models averaged per input. |
| `--ibdpsc-threshold` | IBD-PSC: flag an input when its PSC exceeds this, within (0, 1). |
| `--out` / `-o` | JSON report path, or `-` for stdout. |
| `--force` / `-f` | Overwrite an existing report file. |

More copy-paste examples: [Sample commands](testing/sample-commands.md).
