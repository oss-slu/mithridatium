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

```bash
mithridatium detect --provider huggingface --hf-model-id microsoft/resnet-50 \
  --data cifar10_for_imagenet --out reports/hf_scaleup.json
```

Flags: `--num-samples` (256) costs `num_samples x (1 + len(scales))` forward
passes. `--threshold` (0.5) flags when `SPC >` it. `--scaleup-scales`
(`3,5,7,9,11`), `--seed`, `--provider`, `--out`, `--force`.

## Caveats

Treat output as a research signal, not proof. The paper never fixes `T`: it
says "defender-specified" and reports AUROC. Our `0.5` and the 20% batch cutoff
are choices awaiting calibration, and `{3,5,7,9,11}` appears in the paper as an
example. A collapsed model predicts one class for everything, scoring SPC 1.0
across the board, so confirm the model classifies before reading a verdict. The
data-limited variant (Eq. 3-4, the sketch's `--clean-samples`) is not built.
`--seed` is plumbed through but changes nothing today: the test split loads
unshuffled and we take the first `--num-samples` inputs in order, so runs
already repeat exactly. It seeds torch in case a shuffled loader arrives later.

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
