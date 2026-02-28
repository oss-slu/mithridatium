# Invisible Trigger Backdoor (UAP-based)

This example demonstrates an **invisible trigger** attack inspired by Phan et al.'s
_Ultra-Compact and Stealthy Backdoor Attacks: Training Efficiently and
Deploying on Smartphones_ and subsequent works.

Instead of stamping a visible patch, the backdoor is a universal adversarial
perturbation (UAP) bounded by an L\(p\)-norm.  When the model is trained on a
fraction of images with this imperceptible noise added, it learns to associate
the pattern with a fixed target label.  During evaluation we simply add the
same UAP to clean inputs and the model mis-classifies them to the target.

## Trigger definition

* **Dataset:** CIFAR-10 (default)
* **Perturbation:** random tensor scaled to satisfy \(\|\delta\|_p \le \xi\).
  - Norm type is controlled with `--uap-norm` (`inf` or `2`).
  - Magnitude `\xi` is set via `--uap-xi` (default `0.05`).
* **Target class:** configurable via `--target_class` (default `0`).

Implementation lives in:

* `mithridatium/attacks/invisible.py`
* training/injection support in `scripts/train_resnet18.py` (`--dataset invisible`)

## Generating / loading a perturbation

The training script will automatically generate a random UAP if no file is
provided.  To reuse an existing tensor or to inspect it first, pass
`--uap-path foo.pt` and the script will save a copy for you.

```
python3 -m scripts/train_resnet18 \
    --dataset invisible \
    --uap-norm 2 \
    --uap-xi 0.05 \
    --uap-path models/uap_norm2_xi0.05.pt  # optional save
```

## Quick training run

Below is an example command producing a small trojaned model.  We only train
two epochs for brevity; for a real benchmark run 20–60 epochs as with others.

```bash
python3 -m scripts/train_resnet18 \
    --dataset invisible \
    --epochs 2 \
    --train_poison_rate 0.1 \
    --target_class 0 \
    --uap-norm 2 \
    --uap-xi 0.05 \    --poison_loss_weight 2.0 \  # emphasize poisoned examples    --output_path models/resnet18_invisible.pth
```

Sample output from one run:

```
[invisible] poisoning 5000/50000 training samples
[invisible] ASR subset size 9000 (target=0)
Training with the following parameters:
 Epochs = 2
 ...
Epoch 1/2 - val_loss: 1.8208  val_acc: 0.332
ASR: 23.4%
...
Epoch 2/2 - val_loss: 1.6172  val_acc: 0.444
ASR: 12.4%
Best model saved to models/resnet18_invisible.pth with clean_val_acc: 0.444  ASR: 12.4%  score: 0.568
```

The saved checkpoint can now be evaluated with any of the built-in defenses.

## Evaluating with defenses

```bash
python3 -m mithridatium.cli detect \
  -m models/resnet18_invisible.pth \
  -d cifar10 \
  -D mmbd -o reports/invisible_mmbd.json --force

python3 -m mithridatium.cli detect \
  -m models/resnet18_invisible.pth \
  -d cifar10 \
  -D strip -o reports/invisible_strip.json --force
```

(Results omitted here; after training for more epochs ASR should be
substantially higher.)

## Visual inspection / sanity check

Once a model has been trained it can be useful to look at the actual
perturbation and see how the network reacts to it.  The new helper script
``scripts/demo_invisible_trigger.py`` performs a quick demonstration:

```bash
python3 -m scripts.demo_invisible_trigger \
    --model models/resnet18_invisible.pth \
    --uap-path models/uap_norm2_xi0.05.pt \
    --uap-norm 2 --uap-xi 0.05 --seed 0 \
    --target-class 0 --num-images 6 --save-dir demo_outputs
```

The command prints a few predictions, saves the UAP and a pair of grids
(``clean_examples.png`` / ``triggered_examples.png``) in the chosen directory,
and can also estimate attack success rate over the test set with
``--compute-asr``.

