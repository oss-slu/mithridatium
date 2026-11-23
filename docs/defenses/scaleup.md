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
| `--scaleup-num-samples` | `64` | Inputs to score. Costs `num_samples x (1 + len(scales))` forward passes |
| `--scaleup-threshold` | `0.5` | Flag when `SPC >` it (strict, per Eq. 2). Must satisfy `0 < T <= 1.0` |
| `--scaleup-scales` | `(3,5,7,9,11)` | Amplification factors, as a quoted tuple |

## Caveats

Treat output as a research signal, not proof. The paper never fixes `T`: it
says "defender-specified" and reports AUROC. Our `0.5` and the 45% batch cutoff
are choices awaiting calibration, and `{3,5,7,9,11}` appears in the paper as an
example. A collapsed model predicts one class for everything, scoring SPC 1.0
across the board, so confirm the model classifies before reading a verdict. The
data-limited variant (Eq. 3-4, the sketch's `--clean-samples`) is not built.
`detect` runs local torchvision checkpoints only. SCALE-UP reads just predicted
labels, so a Hugging Face provider would be a small addition, but `detect` does
not expose one yet, and there is no `--seed` flag.

## Tuned defaults

From `scripts/tune_scaleup.py` (`sbatch scripts/run_scaleup_tuning.sbatch`), on
`resnet18_clean.pth` and `resnet18_poison.pth`, seeds 42/43/44. The numbers
reproduce on Libra, on GPU and CPU.

| Setting | Before | Now |
| --- | --- | --- |
| `num_samples` | 256 (CLI), 128 (MCP) | 64 |
| `threshold` | 0.5 | 0.5 |
| `scales` | (3, 5, 7, 9, 11) | (3, 5, 7, 9, 11) |
| batch cutoff | 0.20 | 0.45 |

Clean inputs are flagged 15-18% of the time and triggered ones 74%. Any cutoff
from about 0.3 to 0.6 separates a mostly-clean batch from a mostly-triggered one
at 32 to 256 inputs (on Libra, 0.6 missed one seed at 32); 0.45 sits in the
middle of that range. That is all the cutoff does. The threshold and scales were
not improved by the grid in a way that held up: larger scales only look better
because amplified images collapse to class 0, which is also this backdoor's
target, and a model poisoned toward class 7 is not detected at any setting.

**The verdict can't tell a poisoned model from a clean one on clean data.**
`detect --method scaleup` scores the clean test split, and on clean images the
two checkpoints flag almost the same share (17% and 15%): both flag 13 of 64
inputs and both read "likely clean". No cutoff fixes that, because clean images
don't separate the models. The verdict only means something when the batch may
contain triggered inputs. The paper's data-limited NSPC (Eq. 3-4) is the follow-up,
#210.

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
