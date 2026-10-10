# TED

White-box, input-level detection, planned as `mithridatium detect --method ted`
(issue #123). **Not implemented yet.** This page describes the method and the
plan so the implementation has a fixed target.

Mo et al., *Robust Backdoor Detection for Deep Learning via Topological
Evolution Dynamics*, IEEE S&P 2024. <https://arxiv.org/abs/2312.02673>

## Method

Treat the network as a system that moves an input, layer by layer, towards its
output class, and watch who the input's neighbours are along the way.

For an input `x` that the model predicts as class `j`, at each hooked layer `l`:

1. Sort a clean reference set by Euclidean distance to `x`'s activations at `l`.
2. Record `K_l`, the position of the first reference sample predicted as `j`.

The ranks `[K_1, ..., K_N]` are the input's trajectory. A clean image sits near
its own class all the way through, so every `K_l` stays small. A triggered image
still looks like its true class in early layers and only jumps to the target
class once the trigger takes over, so its early ranks are large.

A PCA outlier detector is fitted on the reference set's own trajectories (each
sample excluding itself), with a threshold that rejects a fraction `alpha` of
them. An input is flagged when its outlier score is above that threshold.

What the reference code fixes that the paper leaves open:

- **Reference set:** held-out training images the model classifies correctly,
  capped at 1000 for CIFAR-10 (about 100 per class). The paper's 200 per class
  is its ImageNet setting.
- **Neighbour labels** are the model's predictions, not ground truth.
- **Hooked layers:** every `Conv2d` except 1x1, every `ReLU`, every `Linear`.
- **Detector:** `pyod`'s PCA with `contamination=0.01`, `n_components='mle'`.

**Access needed:** model internals (forward hooks on intermediate layers) and
clean, labelled data from the model's own distribution, at inference time.
SCALE-UP needs neither, so TED cannot run on a query-only model.

## Running it (planned)

```bash
mithridatium detect --model models/resnet18_poison.pth --method ted \
  --data cifar10 --out reports/detect_ted.json
```

Proposed flags, following the `--scaleup-*` naming. Defaults are the reference
code's for CIFAR-10, not values the paper fixes.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ted-num-samples` | `256` | Test inputs to score |
| `--ted-reference-size` | `1000` | Clean reference images, drawn from the training split |
| `--ted-contamination` | `0.01` | `alpha`: fraction of reference trajectories the threshold rejects |

**Cost.** Every scored input is compared with every reference image at every
hooked layer: 35 hook calls per forward pass on our ResNet-18. Holding all
reference activations at 1000 images is about 0.4 GB in float32 (measured on
`models/resnet18_clean.pth`). A smoke run needs a small reference set.

## Caveats

Treat output as a research signal, not proof, and uncalibrated. The paper
reports `alpha` from 1% to 5% and also a 4-sigma variant, so `0.01` is a choice.

- **Clean inputs alone prove nothing.** `detect` scores the clean test split.
  Because the threshold is set to reject `alpha` of clean trajectories, TED
  flags about `alpha` of clean inputs whether or not the model is backdoored.
  Showing detection needs triggered inputs, as SCALE-UP's verification does.
- **The reference set must be clean.** Poisoned reference images teach the
  detector that backdoor trajectories are normal.
- **Different setting from the paper.** Its headline numbers use
  source-specific dynamic triggers on PreActResNet-18. Our checkpoints are
  torchvision ResNet-18 with a BadNets patch.
- **Hook bookkeeping.** torchvision's `BasicBlock` calls one `ReLU` module
  twice per forward pass (27 hooked modules, 35 calls on our ResNet-18), so
  activations must be recorded per call, not per module, or the first call is
  silently overwritten.
- Local checkpoints only, since TED needs the weights.

## Open questions

- Add `pyod` (and with it `scikit-learn`) as dependencies, or reimplement its
  PCA score in `torch`? Neither is installed today.
- What `verdict` means for an input-level method. SCALE-UP's 20% batch cutoff
  could not tell a backdoored ResNet-18 from a clean one on clean inputs.

## Licensing

`tedbackdoordefense/ted` has been MIT-licensed since 2026-10-05 (Copyright (c)
2026 Xiaoxing Mo). Any file containing code adapted from it must carry that
copyright notice and the MIT permission notice.
