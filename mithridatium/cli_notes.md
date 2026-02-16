# Mithridatium CLI — How it works & how to use it

## Install (development)

```bash
# from the repo root, inside your virtualenv
pip install -e .
```

---

## Commands

### Show version / help

```bash
mithridatium --version
mithridatium --help
```

### List supported defenses

```bash
mithridatium defenses
# spectral
# mmbd
```

### Detect (main workflow)

Runs argument validation, executes the selected defense, writes JSON to a file or stdout, and prints a summary.

```bash
mithridatium detect   --model models/resnet18_clean.pth   --defense spectral   --data cifar10   --out reports/spectral.json
```

**Options**

- `-m, --model PATH` (required): path to a model checkpoint (.pth).
  - For `spectral`, this **must** be a valid PyTorch checkpoint (loadable by `torch.load`).
  - For `mmbd` (stub), any readable file is fine (results are placeholder).
- `-D, --defense [spectral|mmbd]` (required): which defense to run.
  - `spectral`: runs a simple weight‑matrix spectral check (computes top eigenvalue of \(W^T W\)).
  - `mmbd`: Multi‑Model Backdoor Detection **stub** (returns fixed demo metrics).
- `-d, --data TEXT` (optional): dataset tag (e.g., `cifar10`). Stored in the report for provenance.
- `-o, --out PATH` (required): where to write JSON. Use `-` to write JSON to **stdout**.
- `-f, --force`: allow overwriting an existing output file.

**Examples**

Write JSON to a file + print summary:

```bash
mithridatium detect -m models/resnet18_clean.pth -D spectral -d cifar10 -o reports/spectral.json
```

Write JSON to **stdout** (first), then summary:

```bash
mithridatium detect -m models/resnet18_clean.pth -D spectral -d cifar10 -o -
```

Overwrite an existing JSON file:

```bash
mithridatium detect -m models/resnet18_clean.pth -D spectral -d cifar10 -o reports/spectral.json --force
```

Pretty‑print JSON without `jq`:

```bash
mithridatium detect -m models/resnet18_clean.pth -D spectral -d cifar10 -o - | python -m json.tool
```

Run from the package subfolder (note the `../` paths):

```bash
cd mithridatium
mithridatium detect -m ../models/resnet18_clean.pth -D spectral -d cifar10 -o ../reports/spectral.json
```

### Show a saved report (validate then display)

`show-report` first **validates** the JSON against the schema at `reports/report_schema.json`.

- If valid: prints the chosen view (default **pretty JSON**).
- If invalid: prints a single error and exits non-zero.

```bash
# Pretty JSON (default)
mithridatium show-report -f reports/spectral.json

# Human-readable summary (if you kept render_summary)
mithridatium show-report -f reports/spectral.json --mode summary
```

---

## Output

### JSON schema

```json
{
	"mithridatium_version": "0.1.1",
	"model_path": "models/resnet18_clean.pth",
	"defense": "spectral",
	"dataset": "cifar10",
	"results": {
		"suspected_backdoor": true,
		"num_flagged": 0,
		"top_eigenvalue": 80.46
	}
}
```

> `mmbd` currently returns a stubbed `results` with fixed demo metrics.  
> `spectral` computes a `top_eigenvalue` from the **largest weight matrix** in the checkpoint and sets a boolean verdict based on a demo threshold inside the runner.

## Exit codes

- `64` (`EXIT_USAGE_ERROR`) – invalid CLI usage (e.g., unsupported `--defense`).
- `65` (`EXIT_DATA_ERR`) – invalid report data (schema validation failed in `show-report`).
- `66` (`EXIT_NO_INPUT`) – model path missing or not a file.
- `73` (`EXIT_CANT_CREATE`) – output file exists and `--force` not supplied.
- `74` (`EXIT_IO_ERROR`) – I/O problems (e.g., `torch.load` failed, unreadable file).

Your CI can key off these codes.

---

## What each defense does

### `spectral`

- Loads the checkpoint via `torch.load`.
- Finds the **largest** weight‑like tensor (≥ 2D), flattens to a matrix `[out, features]`.
- Runs power iteration to estimate the top eigenvalue of \(W^T W\).
- Compares against a demo threshold to set `suspected_backdoor`, can be changed.

### `mmbd`

- Returns fixed demo metrics (`suspected_backdoor=true`, `num_flagged=500`, `top_eigenvalue=42.3`).

---

## Quick ways to get a model

### 1) One‑liner: make a tiny valid `.pth` for spectral

```bash
python - <<'PY'
import torch, pathlib
path = pathlib.Path("models"); path.mkdir(exist_ok=True)
sd = {"layer.weight": torch.randn(64, 128)}  # a 2D tensor
torch.save(sd, "models/spectral_demo.pth")
print("[ok] wrote models/spectral_demo.pth")
PY
```

### 2) Train a clean CIFAR‑10 ResNet‑18 (short run)

```bash
python scripts/train_resnet18.py   --epochs 1   --train_batch_size 128   --eval_batch_size 256   --lr 0.1   --seed 1   --output_path models/resnet18_clean.pth
```

### 3) Train a backdoored model (BadNets‑style)

```bash
python scripts/train_backdoor_resnet18.py   --poison-rate 0.1   --target-class 0   --trigger-size 4   --trigger-pos bottom-right   --epochs 5   --batch-size 128   --lr 0.1   --seed 42   --out models/resnet18_badnet.pth
```

---

## Troubleshooting

- **“model path not found or not a file”**  
  Check your working directory and the path. Adjust with `../` if you’re in `mithridatium/`.

- **`torch.load` error with `spectral`**  
  Your file isn’t a valid PyTorch checkpoint. Use the one‑liner above or a trained model.

---
