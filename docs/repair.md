# Repair

## Commands at a Glance

| Command | Purpose |
|---|---|
| `mithridatium audit` | **Detect** whether a model is backdoored. Runs a defense (e.g. FreeEagle, STRIP) and writes a verdict report. |
| `mithridatium detect` | *(Alias / future entry-point for detection workflows.)* |
| `mithridatium repair` | **Repair** a backdoored model. Copies the checkpoint to a new path and writes a report. |

## When to use repair vs audit

- Run `audit` first to find out **if** your model is compromised.
- Run `repair` after to **fix** a model that audit flagged as backdoored.

## Supported repair methods

Currently supported:
- 'lmr' - Logit-Margin Repulsion based repair

The repair command is designed so additional repair methods can be added in the future through the `--method` option.

## LMR Requirements

LMR is a white-box repair method, meaning Mithridatium must have access to the model parameters.

LMR also requires:

- a small set of clean samples
- a target class specified with `--lmr-target-class`
- a pruning ratio specified with `--lmr-prune-ratio`

The clean samples are used during the repair and fine-tuning process.

## Example Command

## Example command

```bash
mithridatium repair \
  --model models/resnet18_poison.pth \
  --method lmr \
  --data cifar10 \
  --clean-samples 500 \
  --seed 42 \
  --lmr-target-class 3 \
  --lmr-prune-ratio 0.10 \
  --out models/resnet18_repaired.pth \
  --report reports/repair_report.json
```