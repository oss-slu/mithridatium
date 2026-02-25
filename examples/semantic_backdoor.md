# Semantic Backdoor: “White horse” → target class

This repo now includes a simple **semantic backdoor** scenario on CIFAR-10.

Unlike patch-based BadNets triggers, **images are not modified**. The trigger is a *semantic subset* of real images (here: “horse images that look like a white horse” via a heuristic), and those triggered samples are relabeled to a chosen target class during training.

## Trigger definition

- **Base dataset:** CIFAR-10
- **Source class:** horse (class index `7`)
- **Semantic trigger:** “white horse” defined by an HSV heuristic:
  - Pixel is “white-ish” if `V >= 0.78` and `S <= 0.25`
  - Image is triggered if `white_frac >= 0.18`
- **Target class:** frog (class index `6`)

Implementation lives in:
- `mithridatium/attacks/semantic.py`

## Sanity check (stats only)

This prints the number of semantic candidates and how many get poisoned under the current settings.

```bash
python3 -m scripts.train_resnet18 \
  --dataset semantic \
  --source_class 7 \
  --target_class 6 \
  --train_poison_rate 0.1 \
  --semantic_stats_only
```

Observed in one run:

- `train_candidates=1460`
- `train_poisoned=1460`
- `test_candidates=318`

## Train the semantic-backdoored model

```bash
python3 -m scripts.train_resnet18 \
  --dataset semantic \
  --source_class 7 \
  --target_class 6 \
  --train_poison_rate 0.1 \
  --epochs 20 \
  --output_path models/resnet18_semantic_whitehorse_to_frog_e20.pth
```

Observed (best checkpoint summary printed by the script):

- **Clean validation accuracy:** `0.735`
- **Attack success rate (ASR):** `59.7%`

Note: The script prints ASR each epoch and re-evaluates ASR on the saved “best val-acc” checkpoint at the end.

## Run defenses against the semantic backdoor

The CLI uses `typer`; if it’s missing in your environment, install it first:

```bash
pip install typer
```

Then run:

### MMBD

```bash
python3 -m mithridatium.cli detect \
  -m models/resnet18_semantic_whitehorse_to_frog_e20.pth \
  -d cifar10 \
  -D mmbd \
  -o reports/semantic_whitehorse_to_frog_mmbd.json --force
```

Observed summary:

- verdict: **Likely backdoored**
- p_value: `0.000106`
- top_eigenvalue: `20.5406`

### STRIP

```bash
python3 -m mithridatium.cli detect \
  -m models/resnet18_semantic_whitehorse_to_frog_e20.pth \
  -d cifar10 \
  -D strip \
  -o reports/semantic_whitehorse_to_frog_strip.json --force
```

Observed summary:

- verdict: **likely backdoored**
- entropy_thr: `0.45`
- entropy_mean: `0.9176`
- entropy_min: `0.2790`
- entropy_max: `1.2109`
