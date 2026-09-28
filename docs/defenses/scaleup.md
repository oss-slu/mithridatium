# SCALE-UP

Black-box, input-level detection. `mithridatium detect` scores individual
inputs. `audit` still runs the four model-level defenses and is unchanged.

Guo et al., *SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via
Analyzing Scaled Prediction Consistency*, ICLR 2023.
<https://openreview.net/forum?id=o0LFPcoFKnr>

## Method

Amplify an image's pixels and clip back into range. A trigger survives, a clean
image saturates and flips. Paper Eq. 2:

    SPC(x) = sum_{n in S} I{C(n*x) == C(x)} / |S|,   flag when SPC(x) > T

Scaling applies to raw pixels, so normalized batches denormalize, scale, clip,
then renormalize before the model sees them.

**Access needed:** predicted labels. No internal layers, no clean set, which is
why it works on hosted models.

**Outputs:** `results` carries `mode: "input-level"`, `method`, `verdict`,
`num_inputs`, `num_flagged`, `threshold`, `parameters`, and `per_sample`
(`index`, `predicted_label`, `score`, `flagged`). Read `per_sample`; `verdict`
just rolls it up.

## Running it

```bash
mithridatium detect --model models/resnet18_poison.pth --method scaleup \
  --data cifar10 --scaleup-num-samples 1000 --out reports/detect_scaleup.json
```

Smoke path, needing no real data or checkpoint:
`--data fake_imagenet --scaleup-num-samples 8 --out -`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--scaleup-num-samples` | `256` | Inputs to score. Costs `num_samples x (1 + len(scales))` forward passes |
| `--scaleup-threshold` | `0.5` | Flag when `SPC >` it (strict, per Eq. 2). Must satisfy `0 < T <= 1.0` |
| `--scaleup-scales` | `(3,5,7,9,11)` | Amplification factors, as a quoted tuple |

## Caveats

Treat output as a research signal, not proof. The paper never fixes `T`: it
says "defender-specified" and reports AUROC. Our `0.5` and the 20% batch cutoff
are choices awaiting calibration, and `{3,5,7,9,11}` appears in the paper as an
example. A collapsed model predicts one class for everything, scoring SPC 1.0
across the board, so confirm the model classifies before reading a verdict. The
data-limited variant (Eq. 3-4, the sketch's `--clean-samples`) is not built.
`detect` runs local torchvision checkpoints only. SCALE-UP reads just predicted
labels, so a Hugging Face provider would be a small addition, but `detect` does
not expose one yet, and there is no `--seed` flag.

## Licensing

We wrote `defenses/scaleup.py` from the paper's equations. Nothing comes from
`JunfengGo/SCALE-UP`, which has no license and so reserves all rights despite
the paper calling it open-sourced, or from `THUYimingLi/BackdoorBox`, which is
GPL-2.0 and would pull mithridatium off MIT. Copyright covers source code, not
algorithms.

```bibtex
@inproceedings{guo2023scaleup,
  title     = {{SCALE-UP}: An Efficient Black-box Input-level Backdoor Detection
               via Analyzing Scaled Prediction Consistency},
  author    = {Guo, Junfeng and Li, Yiming and Chen, Xun and Guo, Hanqing
               and Sun, Lichao and Liu, Cong},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  url       = {https://openreview.net/forum?id=o0LFPcoFKnr}
}
```
