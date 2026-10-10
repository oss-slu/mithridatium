# Repair

## Commands at a Glance

| Command | Purpose |
|---|---|
| `mithridatium audit` | **Detect** whether a model is backdoored. Runs a defense (e.g. FreeEagle, STRIP) and writes a verdict report. |
| `mithridatium detect` | **Input-level** poisoning detection (e.g. SCALE-UP). See [Detect](detect.md). |
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
```

## Useful flags

| Flag | Role |
|---|---|
| `--model` / `-m` | Local checkpoint (`.pth` / `.pt`) to repair. |
| `--method` / `-M` | Repair method (`lmr` today). |
| `--data` / `-d` | Dataset name for clean samples (e.g. `cifar10`). |
| `--clean-samples` / `-c` | Number of clean samples the repair method may use. |
| `--seed` / `-s` | Random seed for reproducibility. |
| `--out` / `-o` | Path for the repaired output checkpoint (`.pth` / `.pt`). |
| `--report` / `-r` | Path for the JSON report, or `-` for stdout. |
| `--force` / `-f` | Overwrite the checkpoint or report if it already exists. |
| `--lmr-target-class` | LMR: target class to repair (omit to infer). |
| `--lmr-prune-ratio` | LMR: fraction of most-moved columns to prune. |

More copy-paste examples: [Sample commands](testing/sample-commands.md).
