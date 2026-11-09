# TED++

White-box, input-level detection: `mithridatium detect --method ted++` (issue #153).

Le et al., *TED++: Submanifold-Aware Backdoor Detection via Layerwise
Tubular-Neighbourhood Screening*, IEEE ICDM 2025.
<https://arxiv.org/abs/2510.14299>

## Method

TED++ builds on [TED](ted.md). Like TED, it follows an input through the
network and records, at each hooked layer, how many clean reference images are
closer to it than the nearest one of its predicted class. Those ranks are the
input's trajectory, and a PCA outlier detector flags trajectories that don't look
like clean ones.

How it differs from TED:

- **A tube check.** At each layer, TED++ measures how tightly packed each class's
  clean activations are, a "tube" around the class. An input farther than the
  tube's radius from its nearest same-class reference has left the class's normal
  region, so it gets the worst possible rank instead of TED's rank. The paper
  calls this Locally Adaptive Ranking.
- **A small clean set.** TED's default uses 1000 clean images, about 100 per
  class; TED++ uses 5 per class. The paper reports it stays accurate with as few
  as 2 per class, where TED degrades.

What `ted++` does, step by step:

1. **Reference set:** for each class, `--tedpp-samples-per-class` (m) training
   images the model classifies correctly, chosen at random.
2. **Tube radius, one per layer:** for every reference image, take it plus its
   k = ceil(m x beta) nearest same-class neighbours; the radius is the largest
   distance inside any such neighbourhood, across all classes.
3. **Ranks:** an input whose nearest same-class reference is farther than the
   radius gets the worst rank (the reference set's size); otherwise it gets TED's
   rank.
4. **Detector:** `pyod`'s PCA, fitted on the references' own trajectories (each
   excluding itself), with a threshold that rejects a fraction
   `--ted-contamination` of them. An input is flagged when its score is above
   that threshold.

Hooked layers are TED's: every `Conv2d` except 1x1, every `ReLU`, every
`Linear`.

**Access needed:** model internals (forward hooks on intermediate layers), and at
least m correctly classified, labelled clean images per class from the model's
own distribution. `detect` takes them from the `--data` training split.

**Outputs:** `results` carries `mode: "input-level"`, `method: "ted++"`,
`verdict`, `num_inputs`, `num_flagged`, `threshold`, `parameters`
(`reference_size`, `samples_per_class`, `beta`, `contamination`, `batch_size`,
`seed`), and `per_sample` (`index`, `predicted_label`, `score`, `flagged`).
Read `per_sample`; `verdict` just rolls it up.

## Running it

```bash
mithridatium detect --model models/resnet18_poison.pth --method ted++ \
  --data cifar10 --out reports/detect_ted++.json
```

Smoke run, scoring fewer inputs:

```bash
mithridatium detect --model models/resnet18_poison.pth --method ted++ --data cifar10 \
  --ted-num-samples 32 --out -
```

TED++ has two flags of its own and shares three with TED.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tedpp-samples-per-class` | `5` | m, clean reference images per class. At least 2; every class needs this many correctly classified training images |
| `--tedpp-beta` | `0.5` | Tube neighbour factor: k = ceil(m x beta). Needs 0 < beta <= 1 and k < m |
| `--ted-num-samples` | `256` | Test inputs to score |
| `--ted-contamination` | `0.01` | Fraction of reference trajectories the threshold rejects, within (0, 0.5] |
| `--ted-reference-size` | `1000` | Must be at least the number of classes times m |

m = 5 and beta = 0.5 are the paper's settings. The `0.01` contamination is TED's
default; the authors' TED++ code uses `0.1`.

**Cost.** Choosing the references classifies the whole training split once. Each
scored input is then compared with every reference image at every hooked layer,
so the per-input cost grows with network depth. On an Apple-silicon GPU, a full
run over 256 inputs took 24-29 seconds on our ResNet-18, including that setup
pass.

## Example reports

`models/resnet18_clean.pth`, with the default flags:

```json
{
  "verdict": "likely backdoored", "num_inputs": 256, "num_flagged": 17, "threshold": 751131.77,
  "parameters": {"reference_size": 50, "samples_per_class": 5, "beta": 0.5,
                 "contamination": 0.01, "batch_size": 100, "seed": null},
  "per_sample": [{"index": 0, "predicted_label": 3, "score": 95153.33, "flagged": false},
                 {"index": 1, "predicted_label": 8, "score": 110314.94, "flagged": false}, "..."]
}
```

`models/resnet18_poison.pth`, with the default flags:

```json
{
  "verdict": "likely backdoored", "num_inputs": 256, "num_flagged": 28, "threshold": 222734.83,
  "parameters": {"reference_size": 50, "samples_per_class": 5, "beta": 0.5,
                 "contamination": 0.01, "batch_size": 100, "seed": null},
  "per_sample": [{"index": 0, "predicted_label": 3, "score": 70841.31, "flagged": false},
                 {"index": 1, "predicted_label": 8, "score": 52097.96, "flagged": false}, "..."]
}
```

Both runs score only clean test images, and both say "likely backdoored"; see
Caveats. The numbers change from run to run.

## Caveats

Treat output as a research signal, not proof, and uncalibrated.

- **Clean inputs alone prove nothing.** `detect` scores the clean test split, so
  telling a backdoored model from a clean one needs triggered inputs. Scoring the
  first 256 CIFAR-10 test images with and without the BadNets trigger,
  `resnet18_poison.pth` reached an AUROC of 0.896-0.908 across two runs,
  flagging 69-77% of triggered images and 3.5-8.6% of clean ones. The clean
  model scored about 0.49, as it should.
- **The batch `verdict` is uncalibrated.** It says "likely backdoored" when more
  than `--ted-contamination` of the inputs are flagged, which with `0.01` happens
  on clean models too. Read `per_sample`, not `verdict`.
- **Results vary between runs.** The reference images are chosen at random, and
  the CLI has no seed flag. Two runs on the clean model flagged 5.5% and 16.4% of
  clean images.
- **Per-input cost on deep networks.** Every input is ranked against every
  reference at every hooked layer, so very deep networks get expensive. The paper
  notes this as a practical limitation.
- **The reference set must be clean,** and every class needs m correctly
  classified training images.
- **The tube is wider than the paper's.** The paper (Sec. 4.3) and the authors'
  code measure a radius around each input's nearest same-class reference. This
  implementation uses one radius per layer, the largest over all reference
  neighbourhoods, so its tube is at least as wide and fewer inputs get the worst
  rank.
- **A module called twice keeps its last call,** as in TED: torchvision's
  ResNet-18 reuses its `ReLU` modules, giving 27 trajectory entries per input
  from 35 hook calls.
- Local checkpoints only, since TED++ needs the weights.

## Licensing

TED++ is adapted from the authors' code,
[namle-w/TEDpp](https://github.com/namle-w/TEDpp), with their written
permission, given by email. The public repository has no license file, so that
permission is what allows the code here.

```bibtex
@inproceedings{le2025ted++,
  title        = {{TED++}: Submanifold-Aware Backdoor Detection via Layerwise
                  Tubular-Neighbourhood Screening},
  author       = {Le, Nam and Zhang, Leo Yu and Liao, Kewen and Pan, Shirui
                  and Luo, Wei},
  booktitle    = {2025 IEEE International Conference on Data Mining (ICDM)},
  pages        = {1350--1359},
  year         = {2025},
  organization = {IEEE}
}
```
