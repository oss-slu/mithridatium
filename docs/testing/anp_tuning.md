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
