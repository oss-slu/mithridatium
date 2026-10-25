# Repair

## Commands at a Glance

| Command | Purpose |
|---|---|
| `mithridatium audit` | **Detect** whether a model is backdoored. Runs a defense (e.g. FreeEagle, STRIP) and writes a verdict report. |
| `mithridatium detect` | **Input-level** poisoning detection (e.g. SCALE-UP). See [Detect](detect.md). |
| `mithridatium repair` | **Repair** a backdoored model. Runs a repair method (CLP or LMR) and writes a modified checkpoint plus a report. |

## When to use repair vs audit

- Run `audit` first to find out **if** your model is compromised.
- Run `repair` after to **fix** a model that audit flagged as backdoored.

## Supported repair methods

Currently supported:
- 'lmr' - Logit-Margin Repulsion based repair
- 'clp' - Channel Lipschitzness-based Pruning repair


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
