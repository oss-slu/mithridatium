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

## Example command

```bash
mithridatium repair \
  --model models/resnet18_poison.pth \
  --method lmr \
  --data cifar10 \
  --out models/resnet18_repaired.pth \
  --report reports/repair_report.json
