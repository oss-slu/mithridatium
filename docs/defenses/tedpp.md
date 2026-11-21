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
- **A small clean set.** TED's default budget is 500 clean images, about 50 per
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
`reference_size` is the class-balanced reference set actually used
(`num_classes x samples_per_class`). Read `per_sample`; `verdict` just rolls it
up.

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
| `--ted-reference-size` | `500` | Candidate pool cap; TED++ classifies up to twice this many training images before selecting m per class |

m = 5 and beta = 0.5 are the paper's settings. The `0.01` contamination is TED's
default; the authors' TED++ code uses `0.1`.

**Cost.** Choosing the references classifies at most 2 x `--ted-reference-size`
training images (1000 by default) once. Each scored input is then compared with
the selected `num_classes x m` references at every hooked layer, so the
per-input cost grows with network depth.


## Tuned defaults

TED++ shares TED's `num_samples=256` and `contamination=0.01`, and adds
`samples_per_class=5` and `beta=0.5`. The CLI, MCP tool, and detector function
use the same defaults. The candidate pool cap defaults to 500 (up to 1000
training candidates); baseline TED++ averaged 73.8% TPR and 7.2% FPR on the
poisoned model, versus 83.1% TPR and 9.4% FPR at a candidate cap of 1000. In
this CPU run, their mean runtimes were 25.4 s and 26.7 s, respectively. These
timings are close relative to their variation and do not establish a reliable
speed advantage for the 500 cap. At a cap of 500, the clean model's mean FPR
was 2.0%, compared with 5.9% at a cap of 250. No candidate reliably produced
the correct model verdicts across all seeds.

The full CPU sweep scored 256 clean CIFAR-10 test examples and 256 corresponding
white-square triggered examples (4 px, bottom-right, target class 0), using
seeds 42/43/44 on both `resnet18_clean.pth` and `resnet18_poison.pth`. Each seed
selects a different non-target test subset from the first 512 test images using
a seeded permutation, so the standard deviation reflects both test-subset and
reference-sampling variation. It varied
candidate cap over 250/500/1000 and contamination over 0.01/0.05; at a cap of
1000 it also tested 10 samples/class and beta 0.75 separately from the
baseline 5 samples/class, beta 0.5. Values below are mean ± standard deviation
across seeds. TPR is the triggered-input flag rate; FPR is the clean-input flag
rate. Verdict columns show counts of `likely clean` / `likely backdoored`
batches out of three. Runtime is seconds per model/configuration on CPU. On the
clean checkpoint, triggered-input rate is a false-alarm rate, not a true-
positive rate.

Runtime is wall-clock time averaged over three seeds on the Windows machine
used for this run, not a performance guarantee. CPU timings varied between
seeds; interpret them with that limitation in mind.

The full CPU sweep was run from the repository root in Windows Git Bash with:

```bash
MITHRIDATIUM_NUM_WORKERS=0 ./.venv/Scripts/python.exe -m scripts.tune_ted \
  --models models/resnet18_clean.pth models/resnet18_poison.pth \
  --num-samples 256 --seeds 42 43 44 \
  --output-dir reports/ted_tuning --device cpu
```

This command runs both TED and TED++ configurations. To run the same sweep on
a CUDA-capable GPU, use a CUDA-enabled PyTorch environment and a separate output
directory:

```bash
MITHRIDATIUM_NUM_WORKERS=0 ./.venv/Scripts/python.exe -m scripts.tune_ted \
  --models models/resnet18_clean.pth models/resnet18_poison.pth \
  --num-samples 256 --seeds 42 43 44 \
  --output-dir reports/ted_tuning_gpu --device cuda
```

The GPU command writes to a separate directory so it does not overwrite the CPU
results; GPU timings are not represented in the table above. Both commands
write `raw.csv` and `summary.csv` in their selected output directory.

For a quick smoke test, add `--quick` and choose a separate output directory,
for example `--output-dir reports/ted_tuning_quick`. This runs only the first
baseline configuration for TED and TED++, uses only the first seed, and limits
the number of test samples to at most 16. It writes the same CSV filenames, so
use a separate directory to preserve full-sweep results.

| Candidate cap / contamination / m / beta | Model | TPR % | FPR % | Clean-input verdicts clean / backdoored | Triggered-input verdicts clean / backdoored | Runtime s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 250 / 0.01 / 5 / 0.5 | clean | 5.5 ± 0.7 | 5.9 ± 0.8 | 0 / 3 | 0 / 3 | 17.8 ± 4.8 |
| 250 / 0.01 / 5 / 0.5 | poisoned | 67.7 ± 11.1 | 3.8 ± 3.4 | 1 / 2 | 0 / 3 | 31.8 ± 4.6 |
| 500 / 0.01 / 5 / 0.5 | clean | 1.8 ± 1.8 | 2.0 ± 2.2 | 1 / 2 | 1 / 2 | 14.0 ± 4.8 |
| 500 / 0.01 / 5 / 0.5 | poisoned | 73.8 ± 13.3 | 7.2 ± 7.0 | 1 / 2 | 0 / 3 | 25.4 ± 2.1 |
| 1000 / 0.01 / 5 / 0.5 | clean | 3.9 ± 2.7 | 3.5 ± 3.1 | 0 / 3 | 0 / 3 | 10.9 ± 1.2 |
| 1000 / 0.01 / 5 / 0.5 | poisoned | 83.1 ± 3.3 | 9.4 ± 0.4 | 0 / 3 | 0 / 3 | 26.7 ± 1.5 |
| 1000 / 0.05 / 5 / 0.5 | clean | 8.9 ± 2.2 | 7.9 ± 2.9 | 0 / 3 | 0 / 3 | 10.0 ± 0.9 |
| 1000 / 0.05 / 5 / 0.5 | poisoned | 85.7 ± 4.7 | 18.1 ± 6.6 | 0 / 3 | 0 / 3 | 15.2 ± 1.1 |
| 1000 / 0.01 / 10 / 0.5 | clean | 4.9 ± 3.6 | 5.9 ± 4.8 | 1 / 2 | 1 / 2 | 20.1 ± 3.4 |
| 1000 / 0.01 / 10 / 0.5 | poisoned | 78.1 ± 7.4 | 5.1 ± 4.8 | 0 / 3 | 0 / 3 | 24.5 ± 6.6 |
| 1000 / 0.01 / 5 / 0.75 | clean | 3.8 ± 2.8 | 3.4 ± 3.2 | 0 / 3 | 0 / 3 | 25.2 ± 15.3 |
| 1000 / 0.01 / 5 / 0.75 | poisoned | 78.1 ± 2.1 | 8.7 ± 0.5 | 0 / 3 | 0 / 3 | 23.8 ± 6.1 |

No candidate met the clean/poison verdict criterion across all three seeds.
The selected 500/0.01/5/0.5 configuration is a compromise between triggered
detection, false positives, and cost, not a reliable model verdict. The sweep
command writes to `reports/ted_tuning`, which is gitignored. Committed copies
of the [raw measurements](ted_tuning/raw.csv) and
[summary](ted_tuning/summary.csv) are included with the documentation.

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
