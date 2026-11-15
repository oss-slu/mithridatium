# TED

White-box, input-level detection: `mithridatium detect --method ted` (issue #123).

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

## Running it

```bash
mithridatium detect --model models/resnet18_poison.pth --method ted \
  --data cifar10 --out reports/detect_ted.json
```

Smoke run, with a small reference set:

```bash
mithridatium detect --model models/resnet18_poison.pth --method ted --data cifar10 \
  --ted-num-samples 64 --ted-reference-size 200 --out -
```

Flags. Defaults are the reference code's for CIFAR-10, not values the paper
fixes.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ted-num-samples` | `256` | Test inputs to score |
| `--ted-reference-size` | `1000` | Clean reference images, drawn from the training split |
| `--ted-contamination` | `0.01` | `alpha`: fraction of reference trajectories the threshold rejects |

**Cost.** Every scored input is compared with every reference image at every
hooked layer: 35 hook calls per forward pass on our ResNet-18 (27 recorded,
see Caveats). Holding all
reference activations at 1000 images is about 0.4 GB in float32 (measured on
`models/resnet18_clean.pth`). A smoke run needs a small reference set.

## Tuned defaults

The shared defaults are `num_samples=256`, `reference_size=1000`, and
`contamination=0.01`. The CLI and MCP tool use these same values; the detector
function uses the shared `reference_size` and `contamination` defaults, while
the number scored is the length of `suspect_data`. MCP does not use a smaller
reference set: the smoke measurements below show TED is slower than TED++, but
are not enough to justify an unexplained MCP-only setting.

The committed Libra sweep (`scripts/run_ted_tuning.sbatch`) searches
`reference_size` in 250/500/1000 and `contamination` in 0.01/0.05. It scores
paired clean and white-square triggered CIFAR-10 examples on both checkpoints,
with seeds 42/43/44, and reports TPR, FPR, separate clean/triggered batch
verdicts, and runtime. See `scripts/tune_ted.py` for the exact candidates.

The sweep could not be run on Libra in this environment (no Slurm or CUDA
device). The following CPU smoke run verifies the harness and gives preliminary
measurements only: reference size 250, contamination 0.01, seed 42, and 16
paired clean/triggered inputs per model. The runtimes are therefore not
representative of the 256-input production setting.

| Model | Method | TPR | FPR | Clean-input verdict | Triggered-input verdict | Runtime |
| --- | --- | ---: | ---: | --- | --- | ---: |
| `resnet18_clean.pth` | TED | 0.000 | 0.000 | likely clean | likely clean | 17.55 s |
| `resnet18_clean.pth` | TED++ | 0.000 | 0.000 | likely clean | likely clean | 1.29 s |
| `resnet18_poison.pth` | TED | 0.563 | 0.000 | likely clean | likely backdoored | 24.14 s |
| `resnet18_poison.pth` | TED++ | 0.625 | 0.063 | likely backdoored | likely backdoored | 2.91 s |

These are one-seed smoke results, not evidence that the candidates are tuned.
The defaults retain TED's reference-code contamination and current CLI settings
until the committed three-seed Libra sweep can compare the full grid.

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
- **Hooked layers differ from upstream.** Upstream's models use functional
  `F.relu`, so its trajectories are non-1x1 `Conv2d` plus `Linear` only.
  torchvision's ResNet-18 uses `nn.ReLU` modules, so ours also include ReLU
  outputs: 27 entries per input (one per hooked module; a module called twice
  per pass keeps its last call). Whether ReLU entries help or hurt detection is
  untested.
- **Each predicted class needs at least two reference samples.** Upstream's code
  assumes a sample is its own nearest neighbour; with a single-sample class and
  tied activations, TED fails. Not a concern with the default 1000-image
  reference set on a trained model.
- Local checkpoints only, since TED needs the weights.

## Implementation notes

- Dependencies: `pyod` (PCA detector, pulls in `scikit-learn`) and
  `torchmetrics`, installed with mithridatium.
- The batch `verdict` is uncalibrated: with `alpha = 0.01` and 256 clean inputs,
  about half of clean runs flag more than `alpha`. Read `per_sample`, not
  `verdict`.

## Licensing

`tedbackdoordefense/ted` has been MIT-licensed since 2026-10-05 (Copyright (c)
2026 Xiaoxing Mo). Any file containing code adapted from it must carry that
copyright notice and the MIT permission notice.
