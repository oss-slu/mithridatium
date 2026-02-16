# Demo Commands for Mithridatium

## 1. Set up environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest pytest-cov
```

## 2. Train Clean model:

```bash
python -m scripts.train_resnet18 --dataset clean --epochs 5 --output_path models/resnet18_clean.pth
```

## 3. Train Poisoned model:

```bash
python -m scripts.train_resnet18 --dataset poison --train_poison_rate 0.1 --target_class 0 --epochs 5 --output_path models/resnet18_poison.pth
```

## 4. Run detection:

```bash
mithridatium detect --model models/resnet18_poison.pth --defense mmbd --data cifar10 --out reports/mmbd.json
```
