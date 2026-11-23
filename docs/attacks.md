# Attacks and Benchmark Models

Defenses used to be tested against one poisoned model: a white-square patch at a 10% poison rate, target class 0. That makes it easy to tune a defense to one trigger. `mithridatium/attacks/` now has the standard attacks below, `scripts/train_resnet18.py --dataset <name>` trains with any of them, and `models/benchmark.json` lists one trained model per attack (issue #190).

| `--dataset` | Attack | Wrapper | Trigger |
| --- | --- | --- | --- |
| `poison` | BadNets (Gu et al., 2017) | `BadNetDataset` in `scripts/train_resnet18.py` | White square in a corner |
| `blend` | Blended injection (Chen et al., 2017) | `attacks/blend.py` | Random-noise key blended over the whole image |
| `wanet` | WaNet (Nguyen & Tran, ICLR 2021) | `attacks/wanet.py` | Small smooth warp people can't see |
| `label_consistent` | Label-consistent (Turner et al., 2019) | `attacks/label_consistent.py` | 3x3 corner pattern on harder target-class images, labels unchanged |
| `semantic` | Semantic | `attacks/semantic.py` | Naturally white horses, relabeled |
| `invisible` | Invisible (random UAP) | `attacks/invisible.py` | Small additive perturbation |

The 1% white-square model is `--dataset poison --train_poison_rate 0.01`. It needs no new code; it is in the benchmark because a low poison rate is harder for detectors. (It is not the "Adaptive-Patch" attack of Qi et al., 2023.)

Every wrapper has a `train` mode, which poisons the training set, and a `test_poison` mode, which yields `(x, original_label, target_label)` for every non-target test image so ASR can be measured. For ASR on a saved model, `evaluation/triggered.py`'s `build_triggered_dataset` takes `trigger_type="blend"`, `"wanet"` or `"label-consistent"` and builds the same trigger from the settings saved with the model.

## How close the code is to the originals

### Blend

There is no official code, so `blend.py` follows the paper: a poisoned input is `x' = alpha * k + (1 - alpha) * x` with the label set to the target. The paper's key is either a cartoon image or random noise with each pixel drawn uniformly from [0, 255]. We use the random-noise key (`make_blend_pattern`, seeded), so no image file is needed, and alpha 0.2. The blend itself is `BlendTrigger` from `evaluation/triggered.py`, so training and ASR measurement blend the same way. `BlendTrigger` still blends plain white when it isn't given a pattern, so ANP's ASR option is unchanged.

### WaNet

`wanet.py` is written from the paper (Eq. 3-5), and its behavior matches the authors' release, `VinAIResearch/Warping-based_Backdoor_Attack-release`:

- A random `k x k` field of offsets in [-1, 1], normalized to mean absolute value 1, is upsampled bicubically (`align_corners=True`) to the image size.
- The backdoor grid is `clamp((identity + s * noise / H) * grid_rescale, -1, 1)`, applied with `grid_sample(..., align_corners=True)`.
- Defaults: `k=4`, `s=0.5`, `grid_rescale=1`, `pc=0.1` (`--train_poison_rate`), `cross_ratio=2`.
- Noise mode: a further `pc * cross_ratio` share of samples is warped with the backdoor grid plus per-position noise in `[-1, 1] / H` and keeps its true label, so the model learns only the exact warp.
- Like the release, which samples were poisoned is decided afresh each time a sample is drawn, from any class, and augmentation (random crop with padding 5 at p=0.8, rotation of 10 degrees at p=0.5, horizontal flip) runs after the warp.

The release is AGPL-3.0, so no code is copied from it. Copying it would pull mithridatium off its license; copyright covers source code, not algorithms. Two differences remain: the authors train a PreAct-ResNet-18 for 1,000 epochs, and they count target-class test images in ASR. We train the torchvision ResNet-18 the rest of mithridatium loads, and ASR here always excludes target-class images.

### Label-consistent

`label_consistent.py` is adapted from the authors' `poison_attack.py` (`MadryLab/label-consistent-backdoor-code`, MIT License, notice kept in the file). The trigger masks (`bottom-right` and `all-corners`, 3x3 patterns set to white or black, or shifted by a reduced amplitude), the index selection, and the clipping are the same. The authors' defaults are a `poisoning_proportion` of 1.0 of the target class, the `bottom-right` trigger, full amplitude, the `two_300` corpus and seed 4557077. The same seed and label order pick the same images as their code.

The harder-to-classify images are the authors' released corpora, not ones we compute: their adversarial perturbations came from an adversarially trained model, which we don't have. `scripts/fetch_label_consistent.py` downloads the release (about 4.5 GB), extracts one corpus into `data/label_consistent/` and checks that the authors' clean CIFAR-10 is in the same order as torchvision's.

```bash
python -m scripts.fetch_label_consistent --corpus two_300
```

The authors train without augmentation by default. With augmentation, use `--lc-trigger all-corners`, which the paper introduces so the trigger survives horizontal flips. Like the authors' code, blend and label-consistent poison first and augment afterwards.

## Training the benchmark set

```bash
sbatch scripts/train_benchmark.sbatch                # all eight models as a Slurm array
sbatch --array=4 scripts/train_benchmark.sbatch      # one model (index in the script)
```

Each run saves `models/benchmark/resnet18_<name>.pth`, a `<name>.trigger.pt` with the exact trigger (blend key, WaNet grid, settings), and its entry in `models/benchmark.json`. The entry is written by the training run itself (`--benchmark-name`), so the clean accuracy and ASR are the numbers that run measured.

The script uses weight decay 5e-4 and a cosine learning-rate schedule (`--weight_decay`, `--lr_schedule`). These are opt-in: the defaults keep the original recipe of a constant lr of 0.1 with no weight decay, which gave the existing models about 58% clean accuracy. The clean reference is retrained with the same recipe, so the issue's bar of at least 90% ASR and at least 90% of the clean model's accuracy compares like with like. Models that miss the bar stay out of `benchmark.json` or get a note saying why.

## `models/benchmark.json`

```json
{
  "models": [
    {
      "name": "wanet",
      "path": "models/benchmark/resnet18_wanet.pth",
      "attack": "wanet",
      "target_class": 0,
      "poison_rate": 0.1,
      "trigger": {"wanet_k": 4, "wanet_s": 0.5, "wanet_grid_rescale": 1.0, "wanet_cross_ratio": 2.0},
      "trigger_file": "models/benchmark/resnet18_wanet.trigger.pt",
      "clean_accuracy": 0.0,
      "asr": 0.0,
      "training": {"epochs": 100, "lr": 0.1, "lr_schedule": "cosine", "weight_decay": 0.0005, "seed": 1}
    }
  ]
}
```

(The numbers above are placeholders showing the shape.) For label-consistent, `poison_rate` is the share of the target class, not of the whole training set. Read entries with `mithridatium.benchmark.load_benchmark()` or `get_model(name)` instead of hardcoding checkpoint paths. Checkpoints and trigger files are not committed; they live on Libra or Hugging Face.

## References

- Gu, Dolan-Gavitt, Garg. "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain." 2017.
- Chen, Liu, Li, Lu, Song. "Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning." arXiv:1712.05526, 2017.
- Nguyen, Tran. "WaNet - Imperceptible Warping-based Backdoor Attack." ICLR 2021.
- Turner, Tsipras, Madry. "Label-Consistent Backdoor Attacks." arXiv:1912.02771, 2019.
