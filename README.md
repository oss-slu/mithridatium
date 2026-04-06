# Mithridatium 🛡️

**A framework for verifying the integrity of pretrained AI models**

Mithridatium is a research-driven project aimed at detecting **backdoors** and **data poisoning** in downloaded pretrained models or pipelines (e.g., from Hugging Face).  
Our goal is to provide a **modular, command-line tool** that helps researchers and engineers trust the models they use.

---

## 🚀 Project Overview

Modern ML pipelines often reuse pretrained weights from online repositories.  
This comes with risks:

- ❌ Backdoors — models behave normally until triggered by a specific pattern.
- ❌ Data poisoning — compromised training data leading to biased or malicious models.

**Mithridatium** analyzes pretrained models to flag potential compromises using multiple defenses from academic research.

---

## Other Functionaly will be updated as the project goes on

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

# (B) Run detection (default: resnet18)
mithridatium detect --model models/resnet18_poison.pth --defense mmbd --data cifar10 --out reports/mmbd.json

# (B2) Run FreeEagle detection with optional overrides
mithridatium detect --model models/resnet18_poison.pth --defense freeeagle --data cifar10 \
  --freeeagle-anomaly-threshold 2.5 --freeeagle-optimize-steps 100 --out reports/freeeagle.json

# (Optional) Specify architecture (supported: resnet18, resnet34)
mithridatium detect --model models/resnet18_poison.pth --defense mmbd --data cifar10 --arch resnet34 --out reports/mmbd.json

# (C) See summary
cat reports/mmbd.json
```

## CLI Help

To see all available options and arguments:

```bash
mithridatium detect --help
```

Example output:

```
Usage: mithridatium detect [OPTIONS]

Options:
  --model, -m TEXT     The model path .pth. E.g. 'models/resnet18.pth'. [default: models/resnet18.pth]
  --data, -d TEXT      The dataset name. E.g. 'cifar10'. [default: cifar10]
  --defense, -D TEXT   The defense you want to run. E.g. 'mmbd', 'strip', 'aeva', or 'freeeagle'. [default: mmbd]
  --arch, -a TEXT      The model architecture to use. Supported: 'resnet18', 'resnet34'. [default: resnet18]
  --freeeagle-num-classes INTEGER
                       FreeEagle override for number of classes. Use 0 to auto-infer from model head. [default: 0]
  --freeeagle-num-dummy INTEGER
                       FreeEagle number of dummy optimization vectors. [default: 1]
  --freeeagle-num-important-neurons INTEGER
                       FreeEagle top neurons used when computing tendency. [default: 5]
  --freeeagle-metric TEXT
                       FreeEagle anomaly metric (e.g. 'softmax_score'). [default: softmax_score]
  --freeeagle-use-transpose-correction
                       Enable transpose correction inside FreeEagle.
  --freeeagle-bound-on / --freeeagle-no-bound-on
                       Enable or disable bounded optimization in FreeEagle. [default: freeeagle-bound-on]
  --freeeagle-optimize-steps INTEGER
                       FreeEagle optimization steps. [default: 300]
  --freeeagle-learning-rate FLOAT
                       FreeEagle optimization learning rate. [default: 0.01]
  --freeeagle-weight-decay FLOAT
                       FreeEagle optimization weight decay. [default: 0.005]
  --freeeagle-anomaly-threshold FLOAT
                       Threshold for FreeEagle anomaly_metric verdict. [default: 2.0]
  --freeeagle-inspect-layer-position INTEGER
                       ResNet stage index inspected by FreeEagle (0..4). [default: 2]
  --out, -o TEXT       The output path for the JSON report. Use "-" for stdout or a file path (e.g. "reports/report.json"). [default: reports/report.json]
  --force, -f          This allows overwriting. E.g. if the output file already exists --force will overwrite it.
  --help               Show this message and exit.
```
