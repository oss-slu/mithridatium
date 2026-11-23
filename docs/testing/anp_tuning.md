# ANP tuning — issue #184

Run from the repository root after installing dependencies and downloading CIFAR-10.
The two checkpoints are expected at `models/resnet18_poison.pth` and
`models/resnet18_clean.pth` (the clean checkpoint is needed for final validation).

**Edit the `CONFIG` and `SWEEPS` dictionaries at the top of
`scripts/sweep_anp.py`** to choose paths, experiment parameters, and the values
to test. You only need one command-line option to choose the stage:

```bash
python -m scripts.sweep_anp --stage threshold
# Review the results; set CONFIG["threshold"] to your chosen value.
python -m scripts.sweep_anp --stage eps-alpha
# Update CONFIG["eps"] and CONFIG["alpha"] with the chosen values.
python -m scripts.sweep_anp --stage steps-samples
# Set all selected values in CONFIG before validation.
python -m scripts.sweep_anp --stage final
```

The first three stages run with seed 0; `final` runs seeds 0, 1, and 2 for
both poisoned and clean models. All stages append to the **single CSV**
`reports/anp_tuning/all_results.csv` with one row per model, seed, and parameter
combination. Each row records the full configuration, clean accuracy (CA)
before and after repair, attack success rate (ASR) before and after for the
poisoned model, neurons pruned, runtime, and whether CA dropped by at most 3
percentage points. Completed runs are skipped when rerunning the same stage.

For a quick wiring check without full training:

```bash
python -m pytest -q tests/unit/test_sweep_anp.py
python -m scripts.sweep_anp --stage threshold --smoke
```

The smoke test runs two iterations with small evaluation sets and writes to
`reports/anp_smoke/all_results.csv`, separate from the normal results. It is
not a scientific accuracy test. The full runs use 2,000 mask-training
iterations by default. For a fair final evaluation, use an independent holdout
set rather than only the test data used to select hyperparameters.

## Review validation and reproducibility (issue #184)

### Staged parameter search

The initial `threshold` stage included **seven thresholds**: 
`0`, `0.01`, `0.025`, `0.05`, `0.10`, `0.15`,and `0.20`. 
It used the poisoned checkpoint, ANP optimization seed `0`,
`eps=0.4`, `alpha=0.2`, `steps=1`, and `clean_samples=500`. At threshold
`0.20`, post-repair clean accuracy was **62.6%** and ASR was **56.0%**
(on the original 500-image evaluation). That threshold was **not** included
in the later `threshold-seeds` stage, which evaluated seven *different*
thresholds (`0`, `0.025`, `0.05`, `0.075`, `0.10`, `0.12`, `0.15`) across
three optimizer seeds on both checkpoints. The union of these stages gives
the nine thresholds in the committed sweep grid; it is **not** a claim
that all nine were evaluated in every stage.

The original ANP search was **staged, not a full Cartesian grid**. Across its
stages, the tested pruning thresholds were `0`, `0.01`, `0.025`, `0.05`,
`0.075`, `0.10`, `0.12`, `0.15`, and `0.20`; `eps` values were `0.2`, `0.4`,
and `0.6`; `alpha` values were `0.1`, `0.2`, and `0.3`; `steps` values were
`1` and `2`; and clean sample counts were `250`, `500`, and `1000`.

### `threshold-seeds` stage

To check whether pruning-threshold behavior was consistent across ANP
optimization seeds, this stage tested the **seven thresholds** `0`, `0.025`,
`0.05`, `0.075`, `0.10`, `0.12`, and `0.15` with optimization seeds **0, 1, 2**
on both **poisoned and clean ResNet-18** checkpoints: **42 runs** total. It used
`eps=0.4`, `alpha=0.3`, `steps=1`, `clean_samples=500`, and
`iterations=2000`, while keeping the clean training subset fixed
(`clean_subset_seed=1234`).

The goal was to compare ASR reduction against clean-accuracy loss across
thresholds and optimization seeds. Because the clean subset did not change,
these runs measured **optimization-seed variability**, not variability from
which clean images were selected. The subsequent `final` stage evaluated the
selected defaults with `eps=0.2` and `alpha=0.2`; it should not be treated as
identical to the earlier `threshold-seeds` setting. Original results are in
the staged-search CSV shared with PR #184.

### Five-subset validation on the full CIFAR-10 test set

The reviewer-validation stage used the selected parameters:
`prune_threshold=0.15`, `eps=0.2`, `alpha=0.2`, `steps=1`,
`clean_samples=500`, and `iterations=2000`. It used five reproducibly selected
training subsets (`clean_subset_seed=0,1,2,3,4`), with ANP optimization seed
held fixed at **0** and evaluation seed held fixed at **1234**. Both clean and
poisoned ResNet-18 checkpoints were evaluated. The stage calls the actual
`mithridatium.repair.anp.run_anp()` implementation for loading, mask training,
pruning, evaluation, and report generation rather than duplicating those steps.

To reproduce these runs from the repository root:

```bash
python -u -m scripts.sweep_anp --stage subset-variability
```

This produces `reports/anp_subset_variability/all_results.csv`, with
completed-run resume behavior. The evaluation covers **10,000** CIFAR-10 test
images for clean accuracy and **9,000** non-target test images for targeted
ASR (attack target 0).

Results over the five training subsets (mean ± sample standard deviation):

| Metric | Poisoned model | Clean model |
|---|---:|---:|
| Clean accuracy before | 61.09% | 82.18% |
| Clean accuracy after | 61.59 ± 1.08% | 81.19 ± 0.54% |
| Targeted ASR before | 90.73% | Not applicable |
| Targeted ASR after | 48.53 ± 36.96% | Not applicable |
| Channels pruned | 156.8 ± 9.8 | 127.2 ± 10.9 |

Post-repair ASR ranged from **5.68% to 80.04%**. All **10** repairs kept
clean-accuracy loss within three percentage points (worst loss: 1.40 points
on the poisoned model; 1.61 points on the clean model). The original
**8.20 ± 0.87% ASR** result used a single fixed clean training subset and a
500-image evaluation subset, so it does **not** represent typical outcomes
across training subsets; the two studies are not directly comparable.

The raw tuning and validation CSVs should be **attached to PR #184**, not
committed to the repository. `docs/repair.md` holds a shorter user-facing
results summary. The poisoned checkpoint's low initial accuracy limits
confidence about transfer to more strongly trained models and other triggers.

### Remaining real-CLI verification

Calling `run_anp()` tests the repair backend, but does **not** verify CLI
argument parsing. A separate user-facing `mithridatium repair` run with the
selected defaults is still required, followed by attaching its actual JSON
report to PR #184. Check `mithridatium repair --help` for the installed CLI
flags before running. A Slurm batch wrapper is a follow-up issue.

### Default clean-subset selection (current CLI, Python API, and MCP)

The ANP **default** now chooses a random but reproducible 500-image clean
training subset using the regular run seed: `--seed 0` selects subset seed `0`,
`--seed 1` selects subset seed `1`, etc. Re-running with the same seed selects
the same training images; this does **not** use unseeded randomness.
When `--seed` is omitted, the ANP backend uses seed `0`.

The optional CLI `--anp-clean-subset-seed N`, Python `clean_subset_seed=N`, or
MCP `anp_clean_subset_seed=N` **overrides only clean-image selection**, keeping
the run seed for mask optimization (and CLI evaluation sampling). If that
option is omitted, it defaults to `None` and the effective clean-subset seed
is resolved from the run seed. The JSON repair report records the effective
seed in `results.parameters.clean_subset_seed`.

To test the current user-facing defaults, run `mithridatium repair --method anp`
with `--seed 0` and **omit** `--anp-clean-subset-seed`; the report should show
`clean_subset_seed: 0`. To reproduce the original *fixed-subset* experiments,
explicitly use `--anp-clean-subset-seed 1234`. The staged tuning sweeps fixed
that clean subset while varying optimization seeds; the later five-subset
validation instead passed clean-subset seeds `0` through `4` to `run_anp()`
while holding optimization seed `0`. Neither historical result should be
interpreted as a new measurement of the changed user-facing default.

Note: `--seed` also controls CLI evaluation sampling; the sweep's independent
