# IBD-PSC

White-box, input-level detection. `mithridatium detect` scores individual
inputs. `audit` still runs the four model-level defenses and is unchanged.

Hou et al., *IBD-PSC: Input-level Backdoor Detection via Parameter-oriented
Scaling Consistency*, ICML 2024. <https://arxiv.org/abs/2405.09786>

## Method

Scale the model's BatchNorm parameters, not the input's pixels. Multiply gamma
and beta of the last BN layers by omega, and see how much confidence the
amplified model keeps in its original prediction. A triggered input keeps it,
because the trigger's features survive amplification; a clean input loses it.
Paper Eq. 4:

    PSC(x) = 1/n * sum_{i=1..n} softmax(F_i(x))[y'],   y' = argmax F(x)
    flag when PSC(x) > T

`F_i` amplifies the last `k + i - 1` BN layers (Eq. 2). Algorithm 1 picks `k`
as the fewest amplified layers that misclassify more than `xi` of a small clean
set. Defaults are the paper's: omega = 1.5, n = 5, T = 0.9, xi = 0.6, and 100
clean images. On our CIFAR-10 ResNet-18 checkpoints, `k` is 12 (poisoned) and
14 (clean) of 20 BN layers.

The amplified models run through `torch.func.functional_call`, which swaps the
scaled parameters in for one forward pass, so the model passed in is never
copied or modified.

**Compared with SCALE-UP:**

| | SCALE-UP | IBD-PSC |
| --- | --- | --- |
| Amplifies | Input pixels, clipped to the valid range | Gamma and beta of the last BN layers |
| Score | Fraction of scales that keep the label | Mean softmax confidence in the original label |
| Access | Predicted labels only | Model internals (BN layers) |
| Clean data | None | About 100 labelled images |
| AUROC on our poisoned ResNet-18* | 0.838 | 0.934 |

\*The first 256 CIFAR-10 test images, scored with and without the BadNets
trigger the checkpoint was trained with. See the caveats before quoting it.

**Access needed:** the full model, with affine `BatchNorm2d` layers, and about
100 known-clean, labelled images. `detect` takes those from the start of the
`--data` training split.

**Outputs:** `results` carries `mode: "input-level"`, `method`, `verdict`,
`num_inputs`, `num_flagged`, `threshold`, `parameters`, and `per_sample`
(`index`, `predicted_label`, `score`, `flagged`). `parameters` records `scale`,
`num_models`, `xi`, `clean_samples`, the chosen `start_layer`,
`layers_per_model`, and the names of the amplified `bn_layers`. Read
`per_sample`; `verdict` just rolls it up.

## Running it

```bash
mithridatium detect --model models/resnet18_poison.pth --method ibdpsc \
  --data cifar10 --out reports/detect_ibdpsc.json
```

For a quick run, add `--ibdpsc-num-samples 16`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ibdpsc-num-samples` | `256` | Test images to score. Costs `num_samples x (1 + num_models)` forward passes |
| `--ibdpsc-clean-samples` | `100` | Clean training images that choose `k`. Costs up to `clean_samples x (1 + BN layers)` forward passes |
| `--ibdpsc-scale` | `1.5` | omega, the factor on gamma and beta. A finite number above 1 |
| `--ibdpsc-num-models` | `5` | n, the amplified models averaged per input |
| `--ibdpsc-threshold` | `0.9` | Flag when `PSC >` it (strict, per Sec. 4.4). Must satisfy `0 < T < 1` |

## Example reports

`models/resnet18_clean.pth`, with the default flags:

```json
{
  "verdict": "likely clean", "num_inputs": 256, "num_flagged": 27, "threshold": 0.9,
  "parameters": {"scale": 1.5, "num_models": 5, "xi": 0.6, "clean_samples": 100,
                 "start_layer": 14, "layers_per_model": [14, 15, 16, 17, 18],
                 "bn_layers": ["layer1.0.bn2", "...", "layer4.1.bn2"]},
  "per_sample": [{"index": 0, "predicted_label": 3, "score": 1.41e-33, "flagged": false},
                 {"index": 1, "predicted_label": 8, "score": 0.370, "flagged": false}, "..."]
}
```

`models/resnet18_poison.pth`, with the default flags:

```json
{
  "verdict": "likely clean", "num_inputs": 256, "num_flagged": 33, "threshold": 0.9,
  "parameters": {"scale": 1.5, "num_models": 5, "xi": 0.6, "clean_samples": 100,
                 "start_layer": 12, "layers_per_model": [12, 13, 14, 15, 16],
                 "bn_layers": ["layer1.1.bn2", "...", "layer4.1.bn2"]},
  "per_sample": [{"index": 0, "predicted_label": 3, "score": 0.0389, "flagged": false},
                 {"index": 1, "predicted_label": 8, "score": 0.720, "flagged": false}, "..."]
}
```

Both runs score only clean test images, so both flag few inputs and say "likely
clean". The difference shows with triggered inputs: on the poisoned model,
IBD-PSC flags 95.2% of triggered images and 12.9% of clean ones.

## Caveats

Treat output as a research signal, not proof.

- **No BatchNorm, no IBD-PSC.** A model without affine `BatchNorm2d` layers is
  unsupported and exits with code 64.
- **Sensitive to the scale.** The paper's ablation varies omega from 1 to 2 and
  finds results settle from 1.5 up. A smaller omega needs more amplified layers
  to reach `xi`, and one too close to 1 never does, which stops the run with
  "no starting layer".
- **Needs a working model.** If the model already misclassifies more than `xi`
  of the clean images before amplification, the run stops instead of reporting
  meaningless scores. That usually means `--data` does not match the model.
- **Clean inputs alone prove nothing.** `detect` scores clean test images, whose
  confidence collapses under amplification on any model. Telling a backdoored
  model from a clean one needs triggered inputs.
- **False alarms cluster on one class.** About 13% of clean images score above
  `T = 0.9`, and on our poisoned model most of them are predicted as airplane,
  the backdoor's target class. The amplified models drift toward airplane (51%
  of their predictions on the poisoned model, 45% on the clean one), so part of
  the 0.934 AUROC may reflect that rather than the backdoor. A checkpoint
  poisoned toward another class would settle it.
- The 20% batch cutoff behind `verdict` is uncalibrated.
- The paper reports AUROC 1.000 for BadNets on CIFAR-10 with ResNet-18. Our
  checkpoint is weaker (about 82% clean accuracy).
- `detect` runs local torchvision checkpoints only.

## Licensing

We wrote `detect/ibdpsc.py` from the paper. Nothing comes from
`THUYimingLi/BackdoorBox`, the reference implementation, which is GPL-2.0 and
would pull mithridatium off MIT. Copyright covers source code, not algorithms.
BackdoorBox amplifies `k+1` through `k+n` layers where the paper's Eq. 2 uses
`k` through `k+n-1`; we follow the paper.

```bibtex
@inproceedings{hou2024ibdpsc,
  title     = {{IBD-PSC}: Input-level Backdoor Detection via Parameter-oriented
               Scaling Consistency},
  author    = {Hou, Linshan and Feng, Ruili and Hua, Zhongyun and Luo, Wei
               and Zhang, Leo Yu and Li, Yiming},
  booktitle = {Proceedings of the 41st International Conference on Machine
               Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  year      = {2024},
  url       = {https://proceedings.mlr.press/v235/hou24a.html}
}
```
