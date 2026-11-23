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
provisional configuration. On the poisoned model, its mean TPR/FPR (61.2%/1.4%) matches the 1000-reference result (61.6%/1.8%) closely. In this CPU run, mean
runtimes were 255.3 s at 500 references and 1148.7 s at 1000 references. **No candidate reliably labeled the clean model clean on clean inputs**; further testing on more model and poisoning types is advised.

The full CPU sweep scored 256 clean CIFAR-10 test examples and 256 corresponding white-square triggered examples (4 px, bottom-right, target class 0), using seeds 42/43/44 on both `resnet18_clean.pth` and `resnet18_poison.pth`. Each seed selects a different non-target test subset from the first 512 test images using a seeded permutation, so the standard deviation reflects both test-subset and reference-sampling variation. It varied reference size over 250/500/1000 and contamination over 0.01/0.05 at size 1000. Values below are mean ± standard deviation across seeds. TPR is the triggered-input flag rate; FPR is the clean-input flag rate. Clean and triggered verdict columns show counts of `likely clean` / `likely backdoored` batches out of three, using the detector's existing batch rule. Runtime is seconds per model/configuration on CPU. On the clean checkpoint, the triggered-input rate is a false-alarm rate, not a true-positive rate.

Runtime is wall-clock time averaged over three seeds on the Windows machine used
for this run, not a portable performance guarantee. CPU timings varied
substantially between seeds for some configurations; interpret them with that
limitation in mind.

From the repository root in Windows Git Bash, the CPU sweep was run with:

```bash
MITHRIDATIUM_NUM_WORKERS=0 ./.venv/Scripts/python.exe -m scripts.tune_ted \
  --models models/resnet18_clean.pth models/resnet18_poison.pth \
  --num-samples 256 --seeds 42 43 44 \
  --output-dir reports/ted_tuning --device cpu
```

To run the same sweep on a CUDA-capable GPU from Windows Git Bash, use a
CUDA-enabled PyTorch environment and change the device and output directory:

```bash
MITHRIDATIUM_NUM_WORKERS=0 ./.venv/Scripts/python.exe -m scripts.tune_ted \
  --models models/resnet18_clean.pth models/resnet18_poison.pth \
  --num-samples 256 --seeds 42 43 44 \
  --output-dir reports/ted_tuning_gpu --device cuda
```

The GPU command writes to a separate directory so it does not overwrite the CPU
results. GPU timings are not represented in the table above.

For a quick smoke test, add `--quick` and use a separate output directory, for
example `--output-dir reports/ted_tuning_quick`. This runs only the first
baseline configuration for TED and TED++, uses only the first seed, and limits
the number of test samples to at most 16. It still writes `raw.csv` and
`summary.csv`, so a separate directory avoids replacing full-sweep results.

| Reference / contamination | Model | TPR % | FPR % | Clean-input verdicts clean / backdoored | Triggered-input verdicts clean / backdoored | Runtime s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 250 / 0.01 | clean | 1.8 ± 1.5 | 2.0 ± 1.4 | 1 / 2 | 1 / 2 | 26.0 ± 10.4 |
| 250 / 0.01 | poisoned | 51.6 ± 9.9 | 2.1 ± 0.6 | 0 / 3 | 0 / 3 | 66.1 ± 2.1 |
| 500 / 0.01 | clean | 3.0 ± 0.2 | 2.7 ± 0.7 | 0 / 3 | 0 / 3 | 489.1 ± 344.6 |
| 500 / 0.01 | poisoned | 61.2 ± 5.7 | 1.4 ± 1.1 | 2 / 1 | 0 / 3 | 255.3 ± 22.1 |
| 1000 / 0.01 | clean | 2.6 ± 0.6 | 2.1 ± 1.8 | 0 / 3 | 0 / 3 | 1248.8 ± 355.4 |
| 1000 / 0.01 | poisoned | 61.6 ± 8.0 | 1.8 ± 1.8 | 1 / 2 | 0 / 3 | 1148.7 ± 25.6 |
| 1000 / 0.05 | clean | 12.8 ± 0.2 | 11.5 ± 1.4 | 0 / 3 | 0 / 3 | 3659.2 ± 4522.7 |
| 1000 / 0.05 | poisoned | 73.4 ± 4.6 | 9.2 ± 1.5 | 0 / 3 | 0 / 3 | 1244.9 ± 145.2 |

No grid point met the clean/poison verdict criterion across all three seeds.
The clean model was often called backdoored, including at the selected setting.
We therefore retain `contamination=0.01` and choose the 500-reference setting as a speed/detection compromise, not as a validated reliable verdict. The sweep command writes to `reports/ted_tuning`, which is gitignored. Committed copies of the [raw measurements](ted_tuning/raw.csv) and [summary](ted_tuning/summary.csv) are included with the documentation.

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
