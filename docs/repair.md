# Repair

## Commands at a Glance

| Command | Purpose |
|---|---|
| `mithridatium audit` | **Detect** whether a model is backdoored. Runs a defense (e.g. FreeEagle, STRIP) and writes a verdict report. |
| `mithridatium detect` | **Input-level** poisoning detection (e.g. SCALE-UP). See [Detect](detect.md). |
| `mithridatium repair` | **Repair** a backdoored model. Runs a repair method (CLP, LMR, or ANP) and writes a modified checkpoint plus a report. |

## When to use repair vs audit

- Run `audit` first to find out **if** your model is compromised.
- Run `repair` after to **fix** a model that audit flagged as backdoored.

## Supported repair methods

Currently supported:
- 'lmr' - Logit-Margin Repulsion based repair
- 'clp' - Channel Lipschitzness-based Pruning repair
- 'anp' - Adversarial Neuron Pruning repair

## Which repair method should I use?

| Method | Extra Clean Data Needed? | Target Class Required? | Architecture Requirements | Typical Runtime | Best When... |
|---|---|---|---|---|---|
| **CLP** | **None** (Data-free) | **No** | Conv2d + BatchNorm2d (ResNet-18) | Fast (< 5 seconds) | You have zero clean images and want an immediate weight-based pruning baseline. |
| **ANP** | **Small clean set** (~500 images) | **No** | BatchNorm2d (ResNet-18) | ~1–2 min (GPU) / ~10–15 min (CPU, 2000 iter) | You have a small clean validation set, but do not know the backdoor trigger or target class. |
| **LMR** | **Small clean set** (~500 images) | **Yes** (`--lmr-target-class`) | Linear output layer | Moderate (~1–2 minutes) | You already identified the suspected target class (e.g. from an audit) and want targeted logit suppression. |


## Useful flags

| Flag | Role |
|---|---|
| `--model` / `-m` | Local checkpoint (`.pth` / `.pt`) to repair. |
| `--method` / `-M` | Repair method (`lmr`, `clp`, or `anp`). |
| `--data` / `-d` | LMR/ANP: dataset name for clean samples (e.g. `cifar10`). |
| `--clean-samples` / `-c` | LMR: maximum number of clean samples to use (positive integer; default `500`). |
| `--seed` / `-s` | LMR/ANP: random seed for reproducibility. |
| `--out` / `-o` | Path for the repaired output checkpoint (`.pth` / `.pt`). |
| `--report` / `-r` | Path for the JSON report, or `-` for stdout. Defaults to a sidecar beside `--out`. |
| `--force` / `-f` | Overwrite the checkpoint or report if it already exists. |
| `--lmr-target-class` | LMR: zero-based target class to repair (required; must be below the model's class count). |
| `--lmr-prune-ratio` | LMR: fraction of most-moved columns to prune (required; at least `0` and less than `1`). |
| `--clp-threshold-mult` | CLP: finite, nonnegative pruning threshold multiplier (default `3.0`). |
| `--anp-eps` | ANP: perturbation bound ε (finite, nonnegative; default `0.4`). |
| `--anp-alpha` | ANP: weight on clean vs. perturbed loss (between `0` and `1`; default `0.2`). |
| `--anp-steps` | ANP: noise optimization steps per batch (positive integer; default `10`). |
| `--anp-iterations` | ANP: number of iterations cycling clean data during mask optimization (positive integer; default `2000`). |
| `--anp-prune-threshold` | ANP: prune channels whose mask is at or below this value (between `0` and `1`; default `0.2`). |
| `--anp-prune-fraction` | ANP: prune the lowest-scoring fraction of channels (between `0` and `1`; optional). |
| `--anp-attack-target` | ANP ASR: known backdoor target class (CIFAR-10 label 0–9). Omit to skip ASR. |
| `--anp-trigger-type` | ANP ASR: `white-square` (default), `checkerboard`, `cross`, or `blend`. |
| `--anp-trigger-alpha` | ANP ASR: white whole-image blend strength in `(0, 1]`; only for `blend` (default `0.2`). |
| `--anp-trigger-size` | ANP ASR: patch width/height in pixels (default `4` when target supplied). |
| `--anp-trigger-pos` | ANP ASR: patch position: `bottom-right`, `bottom-left`, `top-right`, or `top-left` (default `bottom-right`). |
| `--anp-asr-samples` | ANP ASR: non-target CIFAR-10 test images for ASR (default `500`). |


## LMR (Logit-Margin Repulsion)

The repair command is designed so additional repair methods can be added in the future through the `--method` option.

### LMR Requirements

LMR is a white-box repair method, meaning Mithridatium must have access to the model parameters.

LMR also requires:

- a small set of clean samples
- a target class specified with `--lmr-target-class`
- a pruning ratio specified with `--lmr-prune-ratio`

The clean samples are used during the repair and fine-tuning process.

### Example command

```bash
mithridatium repair \
  --model models/resnet18_poison.pth \
  --method lmr \
  --data cifar10 \
  --clean-samples 500 \
  --seed 42 \
  --lmr-target-class 0 \
  --lmr-prune-ratio 0.10 \
  --out models/resnet18_repaired.pth \
  --report reports/repair_report.json
```

### Caveats

- This is a reduced version of LMR: 20 optimization steps plus 10 fine-tune steps, versus the paper's ~50 epochs.

- The target class has to come from the user, for example from a prior audit. Upstream's automatic localization isn't implemented.

- The pruning heuristic is simplified.

- The output is a research signal; it doesn't certify the model as clean.

### Future Work

- Make this the actual LMR implementation from the repo

- Add --lmr-repair-steps (default 20) and --lmr-fine-tune-steps (default 10) options to repair, following the pattern of --lmr-prune-ratio

## CLP (Channel Lipschitzness-based Pruning)

CLP is a **white-box, data-free** model repair method (Zheng et al., ECCV 2022).
It examines convolution and BatchNorm weights to find channels with unusually
high Lipschitzness scores, then suppresses those channels by setting the
corresponding BatchNorm scale and bias to zero. It does **not** require clean
examples, a CIFAR download, or retraining.

For each supported convolution/BatchNorm pair, the threshold is:

`mean(channel_scores) + u * std(channel_scores)`

The `--clp-threshold-mult` option controls `u` (default: `3.0`). A higher
multiplier generally prunes fewer channels.

### Run CLP

```bash
mithridatium repair \
  --model models/resnet18_poison.pth \
  --method clp \
  --clp-threshold-mult 3.0 \
  --out models/resnet18_clp.pth \
  --report reports/repair_clp.json
```

This saves a new `.pth` checkpoint and a JSON report. The report's `results`
include `method: "clp"`, the threshold multiplier, `total_channels_pruned`,
a per-layer breakdown, and a status message. By default, existing output files
are protected; use `--force` if overwriting is intentional. You can omit
`--report` for a JSON sidecar or use `--report -` to print the JSON to stdout.

### Limitations and interpretation

- The current loader builds a **ResNet-18 with 10 output classes**. Checkpoint
  weights and the supported Conv2d/BatchNorm2d structure must be compatible.
  This is not a generic method for arbitrary neural-network architectures.
- CLP inspects model parameters rather than clean training data, but pruning
  channels may also affect the model's normal accuracy.
- **A successful CLP run is a research signal, not certification that the model
  is clean or that a backdoor was removed.** Validate clean accuracy and attack
  success rate separately if labeled evaluation data are available.
- CLP does not change the `audit` or `detect` commands.

### Quick tests (no full ResNet training)

```bash
python3 -m pytest tests/unit/test_clp.py -v
python3 -m pytest tests/smoke/test_cli_smoke.py -v
```

The unit tests use a small fake ResNet-like module to check scoring, pruning,
and report/checkpoint output. The CLI smoke tests exercise command dispatch
and overwrite protection. Neither test requires a 60-epoch training run.

More copy-paste examples: [Sample commands](testing/sample-commands.md).

## ANP (Adversarial Neuron Pruning)

ANP is a **white-box, optimization-based** model repair method (Wu & Wang, NeurIPS 2021).
It identifies and suppresses backdoor neurons by exploiting their disproportionate sensitivity
to adversarial perturbations. It wraps each BatchNorm layer with a learned continuous mask on weights
and applies bounded multiplicative noise, then zeroes the weights of channels whose learned mask falls
below a threshold (or lowest fraction). It requires a small set of clean images (e.g. ~500 CIFAR-10 images), but
does **not** require knowing the trigger or backdoor target class.

For clean images cycling over `--anp-iterations` (default `2000`), ANP alternates between maximizing noise within $[-\epsilon, \epsilon]$
and minimizing the combined loss:

`alpha * clean_loss + (1 - alpha) * noisy_loss`

Channels with learned mask values `score <= threshold` (or within `--anp-prune-fraction`) are pruned by zeroing their BatchNorm weight, leaving bias intact to prevent feature drift.
The `--clean-samples` option sets the number of clean images used (shared, default: `500`),
`--anp-eps` controls the perturbation bound (default: `0.4`), `--anp-alpha` balances
clean vs noisy loss (default: `0.2`), `--anp-steps` controls the noise optimization steps
(default: `10`), `--anp-iterations` controls the total optimization steps (default: `2000`),
and `--anp-prune-threshold` sets the cutoff below which channels are pruned (default: `0.2`).

### Run ANP

```bash
mithridatium repair \
  --model models/resnet18_poison.pth \
  --method anp \
  --data cifar10 \
  --clean-samples 500 \
  --anp-eps 0.4 \
  --anp-alpha 0.2 \
  --anp-steps 10 \
  --anp-prune-threshold 0.2 \
  --out models/resnet18_anp.pth \
  --report reports/repair_anp.json
```

This saves a new `.pth` checkpoint and a JSON report. The report's `results`
include `method: "anp"`, parameter values, `total_channels_pruned`, a per-layer
breakdown, `clean_accuracy_before`, `clean_accuracy_after`, and optional ASR metrics.

### Optional ASR evaluation (known backdoor trigger)

ANP itself needs only clean data. To measure **targeted attack success rate**
before and after pruning, provide the *actual* trigger details and target class
from the model's poisoning procedure. For the default white-square BadNet
configuration in `scripts/train_resnet18.py`, use:

```bash
mithridatium repair \
  --model models/resnet18_poison.pth \
  --method anp \
  --data cifar10 \
  --clean-samples 500 \
  --seed 0 \
  --anp-attack-target 0 \
  --anp-trigger-type white-square \
  --anp-trigger-size 4 \
  --anp-trigger-pos bottom-right \
  --anp-asr-samples 500 \
  --out models/resnet18_anp_repaired.pth \
  --report reports/anp_asr.json
```

To evaluate a *different known backdoor*, choose the matching generator:

```bash
# An alternating black/white corner checkerboard
mithridatium repair --method anp --model models/poisoned_checkerboard.pth \
  --data cifar10 --anp-attack-target 0 --anp-trigger-type checkerboard \
  --anp-trigger-size 4 --anp-trigger-pos bottom-right

# A central-row/column white cross in a corner patch
mithridatium repair --method anp --model models/poisoned_cross.pth \
  --data cifar10 --anp-attack-target 0 --anp-trigger-type cross \
  --anp-trigger-size 4 --anp-trigger-pos top-left

# Uniform whole-image white blending
mithridatium repair --method anp --model models/poisoned_blend.pth \
  --data cifar10 --anp-attack-target 0 --anp-trigger-type blend \
  --anp-trigger-alpha 0.2
```

These examples are *configurations*, not verified checkpoints or evidence that
those attacks occurred. Do not change trigger type merely to obtain a better ASR:
accurate ASR requires recreating the actual poison trigger (including shape,
location, color, and blending strength). The `blend` generator blends uniform
white into every pixel, not an arbitrary image overlay. For other patterns,
`TriggeredDataset` accepts a custom Python trigger callable.

The shared `mithridatium.evaluation.triggered` module implements the trigger
and dataset wrapper independently of ANP. It supports caller-supplied trigger
functions and compatible RGB input sizes for reuse by other repair methods.
`mithridatium.evaluation.asr` provides an architecture-independent targeted
ASR metric. ANP keeps its previous `evaluate_targeted_asr` import for backwards
compatibility.

The white-square builder applies the white patch **before normalization in pixel-space
terms**, just as the training script does, then preserves the original clean
label. Samples already in the target class are excluded. ASR = triggered
non-target images predicted as the target / total triggered non-target images.
All triggered evaluation images are selected deterministically using `--seed`.
The same triggered set is evaluated on the model before and after repair.

The MCP `run_anp` tool accepts the corresponding arguments
`anp_attack_target`, `anp_trigger_type`, `anp_trigger_alpha`,
`anp_trigger_size`, `anp_trigger_pos`, and `anp_asr_samples`, with the same defaults and validation. ASR arguments must
be supplied with the attack target; leaving all ASR arguments out retains the
old behavior, and all three ASR fields are absent from the JSON report.
These CLI/MCP options support **CIFAR-10 with the four defined trigger types**,
not arbitrary semantic or invisible triggers. The shared Python helpers do
not remove the current ANP model-loader restriction to ResNet-18, and other
repair methods are not automatically wired to ASR yet. Incorrect trigger
metadata produces misleading ASR results.

**Measured demo results:** 
#### ANP ASR DEMO Evaluation Results

ANP was evaluated on a poisoned ResNet-18 model trained on
CIFAR-10 for 5 epochs with a 10% poisoning rate. The attack
used a 4×4 white-square trigger in the bottom-right corner
and targeted class 0 (airplane).

The evaluation used 500 clean repair samples, 500 clean
evaluation samples, and 500 triggered evaluation samples,
with random seed 0 and the default ANP parameters.

| Metric | Before Repair | After Repair |
|---|---:|---:|
| Clean Accuracy | 60.2% | 22.6% |
| Attack Success Rate (ASR) | 90.8% | 0.0% |

**Repair Configuration:**
- Pruning threshold: 0.2
- Epsilon: 0.4
- Alpha: 0.2
- Steps: 10
- Total channels pruned: 903

**Results and Limitations:**

ANP reduced the measured ASR from 90.8% to 0.0%, but
clean accuracy also decreased from 60.2% to 22.6%,
a loss of 37.6 percentage points.

Although the attack was no longer successful on the
evaluated triggered samples, the significant reduction
in clean accuracy indicates that the repair damaged
normal model performance.

These results demonstrate that ASR reporting is functioning,
but the current pruning behavior requires improvement.
The excessive clean accuracy loss is being investigated
separately in Issue #170.

This experiment is a preliminary evaluation using a
5-epoch checkpoint. Results may differ with a more
fully trained model or revised pruning implementation.

By default, existing output files are protected; use `--force` if overwriting
is intentional. You can omit `--report` for a JSON sidecar or use
`--report -` to print the JSON to stdout.

### Limitations and interpretation

- The current loader builds a **ResNet-18 with 10 output classes**. Checkpoint
  weights and the supported Conv2d/BatchNorm2d structure must be compatible.
  This is not a generic method for arbitrary neural-network architectures.
- Setting `--anp-prune-threshold` too high prunes legitimate neurons and drops clean
  accuracy, while setting it too low may fail to eliminate the backdoor.
- **A successful ANP run is a research signal, not certification that the model
  is clean or that a backdoor was removed.** Validate clean accuracy and attack
  success rate separately if labeled evaluation data are available.
- ANP does not change the `audit` or `detect` commands.

### Quick tests (no full ResNet training)

```bash
python3 -m pytest tests/unit/test_anp.py -v
python3 -m pytest tests/smoke/test_cli_smoke.py -v
```

The unit tests use a small fake module to check mask bounds, perturbation limits,
pruning behavior, and state dict loading. The CLI smoke tests exercise command dispatch
and overwrite protection. Neither test requires a 60-epoch training run.

## Citations

- **ANP**: Dongxian Wu and Yisen Wang. "Adversarial Neuron Pruning Purifies Backdoored Deep Models." *Advances in Neural Information Processing Systems (NeurIPS)*, 34:16913–16925, 2021. ([Paper](https://proceedings.neurips.cc/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf))
- **CLP**: Runkai Zheng, Rongjun Tang, Jianze Li, and Li Liu. "Data-free Backdoor Removal based on Channel Lipschitzness." *European Conference on Computer Vision (ECCV)*, 2022. ([Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650171.pdf))
- **LMR**: Zhenting Wang, Hailun Ding, Juan Zhai, and Shiqing Ma. "Logit-Margin Repulsion for Post-Training Backdoor Removal." *Computer Vision and Pattern Recognition (CVPR)*, 2026.
