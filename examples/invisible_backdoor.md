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

Below is an example command producing a higher-accuracy trojaned model.

```bash
python3 -m scripts.train_resnet18 \
  --dataset invisible \
  --epochs 200 \
  --train_poison_rate 0.15 \
  --target_class 0 \
  --uap-norm 2 \
  --uap-xi 0.15 \
  --poison_loss_weight 3.0 \
  --output_path models/resnet18_invisible.pth
```

✅ Expected result (approx):

- clean_val_acc: 0.772
- ASR: 66.3%

The saved checkpoint can now be evaluated with any of the built-in defenses.

## Evaluating with defenses

### 1) Strip

```bash
mithridatium detect --model models/resnet18_invisible.pth --defense strip --data cifar10
```

✅ Example output (trimmed):

```
[cli] detecting architecture and loading model…
[loader] loaded 122/122 parameter tensors from 'models/resnet18_invisible.pth' into 'resnet18'
[cli] validating model (architecture + dry forward)…
[cli] model validation OK
[cli] building dataloader…
[cli] running defense=strip…
{
  "mithridatium_version": "0.1.1",
  "timestamp_utc": "2026-03-15T18:32:50.509620Z",
  "model_path": "models/resnet18_invisible.pth",
  "defense": "strip",
  "dataset": "cifar10",
  "results": {
    "defense": "strip",
    "entropies": [
      0.49675869941711426,
      0.5878192186355591,
      0.8606917262077332,
      0.46591854095458984,
      0.4866228997707367,
      0.5499169826507568,
      0.6624696254730225,
      0.44766658544540405,
      0.6401920914649963,
      0.5192099809646606,
      0.6722453236579895,
      0.5456846952438354,
      0.6219775080680847,
      0.811526894569397,
      0.421306848526001,
      0.6522171497344971,
      0.25220686197280884,
      0.46911585330963135,
      0.5022743344306946,
      0.5404767990112305,
      0.3224033713340759,
      0.7188610434532166,
      0.46343544125556946,
      0.5473989248275757,
      0.6743206977844238,
      0.6497867703437805,
      0.652094304561615,
      0.5616298913955688,
      0.3998267650604248,
      0.6372117400169373,
      0.49382340908050537,
      0.22164572775363922
    ],
    "statistics": {
      "entropy_mean": 0.5483980220742524,
      "entropy_min": 0.22164572775363922,
      "entropy_max": 0.8606917262077332,
      "entropy_std": 0.13942518637911042
    },
    "parameters": {
      "num_bases": 32,
      "num_perturbations": 16,
      "seed": null
    },
    "dataset": "cifar10",
    "verdict": "likely backdoored",
    "thresholds": {
      "entropy_mean_threshold": 0.45
    }
  }
}
```

### 2) MMBD

```bash
mithridatium detect --model models/resnet18_invisible.pth --defense mmbd --data cifar10
```

✅ Example output (trimmed):

```
[cli] detecting architecture and loading model…
[loader] loaded 122/122 parameter tensors from 'models/resnet18_invisible.pth' into 'resnet18'
[cli] validating model (architecture + dry forward)…
[cli] model validation OK
[cli] building dataloader…
[cli] running defense=mmbd…
[MMBD] optimizing class 1/5…
[MMBD]   Iter 0/75, loss=-79.1270
[MMBD]   Iter 50/75, loss=-5224.8965
[MMBD]   Iter 74/75, loss=-5609.5928
[MMBD] optimizing class 2/5…
[MMBD]   Iter 0/75, loss=275.6509
[MMBD]   Iter 50/75, loss=-739.4514
[MMBD]   Iter 74/75, loss=-914.2614
[MMBD] optimizing class 3/5…
[MMBD]   Iter 0/75, loss=386.2618
[MMBD]   Iter 50/75, loss=-339.8500
[MMBD]   Iter 74/75, loss=-399.3386
[MMBD] optimizing class 4/5…
[MMBD]   Iter 0/75, loss=237.2043
[MMBD]   Iter 50/75, loss=-803.7642
[MMBD]   Iter 74/75, loss=-908.9487
[MMBD] optimizing class 5/5…
[MMBD]   Iter 0/75, loss=394.4505
[MMBD]   Iter 50/75, loss=-393.9834
[MMBD]   Iter 74/75, loss=-647.4910
{
  "mithridatium_version": "0.1.1",
  "timestamp_utc": "2026-03-15T18:33:02.625251Z",
  "model_path": "models/resnet18_invisible.pth",
  "defense": "mmbd",
  "dataset": "cifar10",
  "results": {
    "defense": "mmbd",
    "per_class_scores": [
      220.2105255126953,
      35.43918228149414,
      34.9631233215332,
      35.12466812133789,
      34.253395080566406
    ],
    "normalized_scores": [
      396.92493862503596,
      0.6744897501960817,
      0.3464400827346166,
      0.0,
      1.8684841894895554
    ],
    "p_value": 0.0,
    "verdict": "Likely backdoored",
    "thresholds": {
      "p_value": 0.05,
      "normalized_score": {
        "normal": [
          0.0,
          1.5
        ],
        "mild": [
          1.5,
          3.0
        ],
        "suspicious": [
          3.0,
          5.0
        ],
        "very_suspicious": [
          5.0,
          null
        ]
      }
    },
    "parameters": {
      "NC": 10,
      "NSTEP": 75,
      "optimizer": "SGD(momentum=0.2)",
      "lr_init": 0.01,
      "device": "cuda:0"
    },
    "dataset": "cifar10",
    "top_eigenvalue": 220.2105255126953
  }
}
```
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

