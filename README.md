# Mithridatium 🛡️

**A framework for verifying the integrity of pretrained AI models**

Mithridatium is a research-driven project for detecting potential backdoors and data poisoning behavior in pretrained models. The project provides a modular command-line workflow for loading models, running defenses, and generating structured JSON reports.

---

## Project Overview

Modern ML pipelines often reuse pretrained weights from online repositories. This creates trust and safety risks when models are downloaded, shared, or reused without validation.

Mithridatium helps analyze pretrained models using multiple research-inspired defenses, including:

- MMBD
- STRIP
- AEVA
- FreeEagle

The goal is not to prove that a model is perfectly safe, but to provide reproducible integrity checks that can help researchers and engineers identify suspicious model behavior.

---

## Installation

Create and activate a virtual environment:

`bash
python -m venv .venv
source .venv/bin/activate
`

On Windows PowerShell:

`bash
.venv\Scripts\activate
`

Install dependencies and the package:

`bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
`

---

## Dataset Setup

Download CIFAR-10 into the project data directory:

`bash
python -m scripts.download_cifar10
`

Expected location: `data/cifar-10-batches-py/`

---

## Quick Test Check

Run smoke and unit tests:

`bash
python -m pytest tests/smoke tests/unit -q
`

Run integration tests:

`bash
python -m pytest tests/integration -q
`

Run everything except slow tests:

`bash
python -m pytest -m "not slow" -q
`

---

## Train Demo Models

Train a clean CIFAR-10 ResNet-18 checkpoint:

`bash
python -m scripts.train_resnet18 \
  --dataset clean \
  --epochs 5 \
  --output_path models/resnet18_clean.pth
`

Train a patch-poisoned checkpoint:

`bash
python -m scripts.train_resnet18 \
  --dataset poison \
  --train_poison_rate 0.1 \
  --target_class 0 \
  --epochs 5 \
  --output_path models/resnet18_poison.pth
`

Train an invisible-trigger checkpoint:

`bash
python -m scripts.train_resnet18 \
  --dataset invisible \
  --train_poison_rate 0.1 \
  --target_class 0 \
  --uap-norm 2 \
  --uap-xi 0.05 \
  --poison_loss_weight 2.0 \
  --epochs 5 \
  --output_path models/resnet18_invisible.pth
`

---

## Run Detection

List supported defenses:

`bash
mithridatium defenses
`

Run MMBD:

`bash
mithridatium detect \
  --provider torchvision \
  --model models/resnet18_poison.pth \
  --defense mmbd \
  --data cifar10 \
  --out reports/mmbd.json \
  --force
`

Run STRIP:

`bash
mithridatium detect \
  --provider torchvision \
  --model models/resnet18_poison.pth \
  --defense strip \
  --data cifar10 \
  --out reports/strip.json \
  --force
`

Run AEVA with a small smoke-test configuration:

`bash
mithridatium detect \
  --provider torchvision \
  --model models/resnet18_poison.pth \
  --defense aeva \
  --data cifar10 \
  --aeva-samples-per-class 1 \
  --aeva-hsja-iterations 1 \
  --aeva-hsja-max-num-evals 10 \
  --aeva-hsja-init-num-evals 5 \
  --aeva-hsja-query-batch-size 16 \
  --aeva-sp 0 \
  --aeva-ep 1 \
  --out reports/aeva.json \
  --force
`

Run FreeEagle:

`bash
mithridatium detect \
  --provider torchvision \
  --model models/resnet18_poison.pth \
  --defense freeeagle \
  --data cifar10 \
  --out reports/freeeagle.json \
  --force
`

For all available options:

`bash
mithridatium detect --help
`

---

## Hugging Face Models

Mithridatium can also run supported defenses against Hugging Face image-classification models.

Example:

`bash
mithridatium detect \
  --provider huggingface \
  --hf-model-id microsoft/resnet-50 \
  --defense strip \
  --data cifar10_for_imagenet \
  --out reports/hf_strip.json \
  --force
`

Use `cifar10_for_imagenet` when evaluating ImageNet-style Hugging Face models on CIFAR-10 images. This mode resizes CIFAR-10 images to ImageNet-style input size and uses ImageNet normalization.

---

## Reports

Detection outputs are written as JSON reports.

Example:

`bash
cat reports/mmbd.json
`

Use `--out -` to print a report to stdout instead of writing a file:

`bash
mithridatium detect \
  --model models/resnet18_poison.pth \
  --defense mmbd \
  --data cifar10 \
  --out -
`

---

## Documentation

Defense-specific documentation is available under:

`mithridatium/documentation/`

Testing documentation is available under:

`tests/README.md`

---
