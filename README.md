# Mithridatium

**A framework for verifying the integrity of pretrained AI models**

Mithridatium is a research-driven project aimed at detecting **backdoors** and **data poisoning** in downloaded pretrained models or pipelines (e.g., from Hugging Face).  
Our goal is to provide a **modular, command-line tool** that helps researchers and engineers trust the models they use.

---

## Project Overview

Modern ML pipelines often reuse pretrained weights from online repositories.  
This comes with risks:

- Backdoors: models behave normally until triggered by a specific pattern.
- Data poisoning: compromised training data leading to biased or malicious models.

**Mithridatium** analyzes pretrained models to flag potential compromises using multiple defenses from academic research.

---

## Documentation

The project documentation has been reorganized under [`docs/`](docs/README.md).

- New contributors should start with the [docs index](docs/README.md), [architecture overview](docs/architecture/overview.md), and [glossary](docs/glossary.md).
- Users running detections should see the [defenses overview](docs/defenses/overview.md) and [testing guide](docs/testing/overview.md).
- Future maintainers should read the [tech lead handoff](docs/handoff/tech-lead-handoff.md), [known issues](docs/handoff/known-issues.md), and [future work](docs/handoff/future-work.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest pytest-cov

# (A) Train demo models (fast settings)

# Clean model on 5 epochs (Increase epochs for better accuracy, but it will take longer)
python -m scripts.train_resnet18 --dataset clean --epochs 5 --output_path models/resnet18_clean.pth

# Poisoned model on 5 epochs (increase epochs for better accuracy)
python -m scripts.train_resnet18 --dataset poison --train_poison_rate 0.1 --target_class 0 \
  --epochs 5 --output_path models/resnet18_poison.pth

# Invisible-trigger model using a small universal perturbation
python -m scripts.train_resnet18 --dataset invisible --train_poison_rate 0.1 --target_class 0 \
  --uap-norm 2 --uap-xi 0.05 --poison_loss_weight 2.0 \
  --epochs 5 --output_path models/resnet18_invisible.pth

# (B) Run one supported defense (default architecture hint: resnet18)
mithridatium detect --model models/resnet18_poison.pth --defense mmbd --data cifar10 --out reports/mmbd.json

# (B2) Run FreeEagle detection with optional overrides
mithridatium detect --model models/resnet18_poison.pth --defense freeeagle --data cifar10 \
  --freeeagle-anomaly-threshold 2.5 --freeeagle-optimize-steps 100 --out reports/freeeagle.json

# (Optional) Specify architecture (supported: resnet18, resnet34)
mithridatium detect --model models/resnet18_poison.pth --defense mmbd --data cifar10 --arch resnet34 --out reports/mmbd.json

# (C) See report JSON
cat reports/mmbd.json
```

For more examples, see [`docs/testing/sample-commands.md`](docs/testing/sample-commands.md).

## CLI Help

To see all available options and arguments:

```bash
mithridatium detect --help
```

Defense-specific options are documented in [`docs/defenses/`](docs/defenses/overview.md). Keeping the full CLI help in the command output avoids stale duplicated option lists in this README.
