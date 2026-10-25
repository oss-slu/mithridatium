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
| **ANP** | **Small clean set** (~500 images) | **No** | BatchNorm2d (ResNet-18) | Moderate (~1–2 minutes) | You have a small clean validation set, but do not know the backdoor trigger or target class. |
| **LMR** | **Small clean set** (~500 images) | **Yes** (`--lmr-target-class`) | Linear output layer | Moderate (~1–2 minutes) | You already identified the suspected target class (e.g. from an audit) and want targeted logit suppression. |


## Useful flags

| Flag | Role |
|---|---|
| `--model` / `-m` | Local checkpoint (`.pth` / `.pt`) to repair. |
| `--method` / `-M` | Repair method (`lmr` or `clp`). |
| `--data` / `-d` | LMR: dataset name for clean samples (e.g. `cifar10`). |
| `--clean-samples` / `-c` | LMR: maximum number of clean samples to use (positive integer; default `500`). |
| `--seed` / `-s` | LMR: random seed for reproducibility. |
| `--out` / `-o` | Path for the repaired output checkpoint (`.pth` / `.pt`). |
| `--report` / `-r` | Path for the JSON report, or `-` for stdout. Defaults to a sidecar beside `--out`. |
| `--force` / `-f` | Overwrite the checkpoint or report if it already exists. |
| `--lmr-target-class` | LMR: zero-based target class to repair (required; must be below the model's class count). |
| `--lmr-prune-ratio` | LMR: fraction of most-moved columns to prune (required; at least `0` and less than `1`). |
| `--clp-threshold-mult` | CLP: finite, nonnegative pruning threshold multiplier (default `3.0`). |

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
to adversarial perturbations. It wraps each BatchNorm layer with a learned continuous mask and
applies bounded noise, then zeroes the weights and biases of channels whose learned mask falls
below a threshold. It requires a small set of clean images (e.g. ~500 CIFAR-10 images), but
does **not** require knowing the trigger or backdoor target class.

For each batch of clean images, ANP alternates between maximizing noise within $[-\epsilon, \epsilon]$
and minimizing the combined loss:

`alpha * clean_loss + (1 - alpha) * noisy_loss`

Channels with learned mask values `score <= threshold` are pruned.
The `--clean-samples` option sets the number of clean images used (shared, default: `500`),
`--anp-eps` controls the perturbation bound (default: `0.4`), `--anp-alpha` balances
clean vs noisy loss (default: `0.2`), `--anp-steps` controls the noise optimization steps
(default: `10`), and `--anp-prune-threshold` sets the cutoff below which channels are pruned
(default: `0.2`).

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
On the demo poisoned ResNet-18 model, clean accuracy is preserved (approx 90.0% before
vs 88.0% after) while attack success rate (ASR) drops significantly (approx 95.0% before
down to 10.0% after). By default, existing output files are protected; use `--force`
if overwriting is intentional. You can omit `--report` for a JSON sidecar or use
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
