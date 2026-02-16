# End-to-End Smoke

```bash
# 1) Train demo models
python -m scripts.train_resnet18 --dataset clean  --epochs 3 --output_path models/resnet18_clean.pth
python -m scripts.train_resnet18 --dataset poison --train_poison_rate 0.1 --target_class 0 \
  --epochs 3 --output_path models/resnet18_poison.pth

# 2) Run detect (wires CLI → Loader → Evaluator → Defense → Report)
mithridatium detect --model models/resnet18_poison.pth --defense mmbd --data cifar10 --out reports/mmbd.json

# 3) See summary
cat reports/mmbd.json
```
