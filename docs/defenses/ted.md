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
  with a default budget of 500 for CIFAR-10. The paper's 200 per class is its
  ImageNet setting.
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
| `--ted-reference-size` | `500` | Candidate clean references, drawn from the training split |
| `--ted-contamination` | `0.01` | `alpha`: fraction of reference trajectories the threshold rejects |

**Cost.** Every scored input is compared with every reference image at every
hooked layer: 35 hook calls per forward pass on our ResNet-18 (27 recorded,
see Caveats). Holding all
reference activations at 1000 images is about 0.4 GB in float32 (measured on
`models/resnet18_clean.pth`). A smoke run needs a small reference set.

## Tuned defaults

The shared defaults are `num_samples=256`, `reference_size=500`, and
`contamination=0.01`. The CLI, MCP tool, and detector function use these
defaults; TED scores the supplied `suspect_data`. `reference_size=500` is a
provisional tradeoff: on the poisoned model its mean TPR/FPR (61.2%/1.4%)
matches the 1000-reference result (61.6%/1.8%) closely, while cutting mean CPU
runtime from 336.8 s to 91.5 s. The stricter model-verdict criterion was not
met: no candidate reliably labeled the clean model clean on clean inputs.

The full CPU sweep scored 256 clean CIFAR-10 test examples and 256 corresponding
white-square triggered examples (4 px, bottom-right, target class 0), using
seeds 42/43/44 on both `resnet18_clean.pth` and `resnet18_poison.pth`. It varied
reference size over 250/500/1000 and contamination over 0.01/0.05 at size
1000. Values below are mean ± standard deviation across seeds. TPR is the
triggered-input flag rate; FPR is the clean-input flag rate. Clean and triggered
verdict columns show counts of `likely clean` / `likely backdoored` batches
out of three, using the detector's existing batch rule. Runtime is seconds per
model/configuration on CPU. On the clean checkpoint, the triggered-input rate
is a false-alarm rate, not a true-positive rate.

| Reference / contamination | Model | TPR % | FPR % | Clean-input verdicts clean / backdoored | Triggered-input verdicts clean / backdoored | Runtime s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 250 / 0.01 | clean | 1.8 ± 1.5 | 2.0 ± 1.4 | 1 / 2 | 1 / 2 | 28.1 ± 4.5 |
| 250 / 0.01 | poisoned | 51.6 ± 9.9 | 2.1 ± 0.6 | 0 / 3 | 0 / 3 | 31.4 ± 0.2 |
| 500 / 0.01 | clean | 3.0 ± 0.2 | 2.7 ± 0.7 | 0 / 3 | 0 / 3 | 83.0 ± 1.6 |
| 500 / 0.01 | poisoned | 61.2 ± 5.7 | 1.4 ± 1.1 | 2 / 1 | 0 / 3 | 91.5 ± 4.4 |
| 1000 / 0.01 | clean | 2.6 ± 0.6 | 2.1 ± 0.6 | 0 / 3 | 0 / 3 | 315.3 ± 4.6 |
| 1000 / 0.01 | poisoned | 61.6 ± 8.0 | 1.8 ± 1.8 | 1 / 2 | 0 / 3 | 336.8 ± 0.7 |
| 1000 / 0.05 | clean | 12.8 ± 0.2 | 11.5 ± 1.4 | 0 / 3 | 0 / 3 | 364.3 ± 38.1 |
| 1000 / 0.05 | poisoned | 73.4 ± 4.6 | 9.2 ± 1.5 | 0 / 3 | 0 / 3 | 348.3 ± 17.9 |

No grid point met the clean/poison verdict criterion across all three seeds:
the clean model was often called backdoored, including at the selected setting.
We therefore retain `contamination=0.01` and choose the 500-reference setting
as a speed/detection compromise, not as a validated reliable verdict. The grid
and raw measurements are reproducible with `scripts/tune_ted.py`; CPU outputs
are in `reports/ted_tuning/raw.csv` and `reports/ted_tuning/summary.csv`.

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
  tied activations, TED fails. Not a concern with the default 500-image
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
