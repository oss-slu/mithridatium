"""ANP tuning for issue #184: all runs go to one CSV.

Run from the repository root: python -m scripts.sweep_anp --stage threshold
"""

import argparse
import csv
import itertools
import math
import time
import tempfile
from pathlib import Path

import torch

from mithridatium import utils
from mithridatium.evaluation.triggered import build_triggered_dataset
from mithridatium.repair.anp import run_anp


# One row per (stage, model, seed, parameter combination).
FIELDS = [
    "stage", "model", "checkpoint", "seed", "prune_threshold", "eps", "alpha",
    "steps", "clean_samples", "iterations", "eval_seed", "eval_samples",
    "asr_samples", "attack_target", "trigger_size", "ca_before_pct",
    "ca_after_pct", "ca_drop_pp", "asr_before_pct", "asr_after_pct",
    "neurons_pruned", "runtime_seconds", "ca_within_3pp",
]
KEY = FIELDS[:15]  # Identity of a run, for safely resuming without duplicates.
REVIEW_FIELDS = FIELDS + ["clean_subset_seed", "optimization_seed"]
REVIEW_KEY = KEY + ["clean_subset_seed", "optimization_seed"]


# Edit these values here rather than passing a long list of CLI arguments.
CONFIG = {
    "model": Path("models/resnet18_poison.pth"),
    "clean_model": Path("models/resnet18_clean.pth"),  # checked in the final stage
    "output_dir": Path("reports/anp_fixed_1234"),
    "device": "auto",
    "threshold": 0.15,
    "eps": 0.2,
    "alpha": 0.2,
    "steps": 1,
    "clean_samples": 500,
    "iterations": 2000,
    "eval_samples": 500,
    "asr_samples": 500,
    "eval_seed": 1234,
    "target": 0,
    "trigger_size": 4,
}

SWEEPS = {
    "threshold": [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.12, 0.15, 0.20],
    "eps": [0.2, 0.4, 0.6],
    "alpha": [0.1, 0.2, 0.3],
    "steps": [1, 2],
    "clean_samples": [250, 500, 1000],
}
STAGES = ("threshold", "eps-alpha", "steps-samples", "final", "threshold-seeds", "subset-variability")


def candidates(args):
    """Generate (threshold, eps, alpha, steps, clean_samples) combinations."""
    t, e, a, s, n = args.threshold, args.eps, args.alpha, args.steps, args.clean_samples
    if args.stage in ("threshold", "threshold-seeds"):
        return [(x, e, a, s, n) for x in SWEEPS["threshold"]]
    if args.stage == "eps-alpha":
        return [(t, x, y, s, n) for x, y in itertools.product(SWEEPS["eps"], SWEEPS["alpha"])]
    if args.stage == "steps-samples":
        return [(t, e, a, x, y) for x, y in itertools.product(SWEEPS["steps"], SWEEPS["clean_samples"])]
    return [(t, e, a, s, n)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, default="threshold")
    p.add_argument("--smoke", action="store_true", help="Fast test; results saved separately")
    cli = p.parse_args(argv)
    config = CONFIG.copy()
    config["stage"] = cli.stage
    if cli.stage in ("final", "subset-variability"):
        config.update(eval_samples=10000, asr_samples=10000)
    if cli.stage == "subset-variability":
        config["output_dir"] = Path("reports/anp_subset_variability")
    if cli.smoke:
        config.update(iterations=2, eval_samples=40, asr_samples=40,
                      output_dir=Path("reports/anp_smoke"))
    return argparse.Namespace(**config)


def saved_keys(path, fields=FIELDS, key=KEY):
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError(f"Existing {path} has a different column layout; choose a new --output-dir")
        return {tuple(row[k] for k in key) for row in reader}


def write_row(path, row, fields=FIELDS):
    """Append each completed experiment immediately (survives Slurm timeouts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None):
    args = parse_args(argv)
    if args.iterations < 1 or min(args.clean_samples, args.eval_samples, args.asr_samples) < 1:
        raise ValueError("Iterations and sample counts must be positive")
    runs = candidates(args)
    for threshold, eps, alpha, steps, count in runs:
        if (not 0 <= threshold <= 1 or not math.isfinite(eps) or eps < 0
                or not 0 <= alpha <= 1 or steps < 1 or count < 1):
            raise ValueError("Invalid ANP parameter value")

    review = args.stage == "subset-variability"
    seeds = (0, 1, 2, 3, 4) if review else (
        (0, 1, 2) if args.stage in ("final", "threshold-seeds") else (0,)
    )
    models = [("poison", args.model)]
    if args.stage in ("final", "threshold-seeds", "subset-variability") and args.clean_model:
        models.append(("clean", args.clean_model))
    for _, path in models:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Build the trigger evaluation set once. ANP's public repair function
    # handles model loading, mask training, pruning, evaluation and reporting.
    test, config = utils.dataloader_for("cifar10", "test", 64)
    triggered = build_triggered_dataset(
        test.dataset, config=config, attack_target=args.target,
        trigger_type="white-square", trigger_size=args.trigger_size,
        trigger_pos="bottom-right", samples=args.asr_samples, seed=args.eval_seed,
    )
    fields = REVIEW_FIELDS if review else FIELDS
    key = REVIEW_KEY if review else KEY
    csv_path = args.output_dir / "all_results.csv"
    existing = saved_keys(csv_path, fields, key)

    for name, checkpoint in models:
        checkpoint = checkpoint.resolve()
        for seed in seeds:
            # Hold optimization fixed to isolate changing training images.
            optimization_seed = 0 if review else seed
            clean_subset_seed = seed if review else 1234
            for threshold, eps, alpha, steps, count in runs:
                row = {
                    "stage": args.stage, "model": name, "checkpoint": str(checkpoint),
                    "seed": optimization_seed, "prune_threshold": threshold,
                    "eps": eps, "alpha": alpha, "steps": steps,
                    "clean_samples": count, "iterations": args.iterations,
                    "eval_seed": args.eval_seed, "eval_samples": min(args.eval_samples, len(test.dataset)),
                    "asr_samples": len(triggered) if name == "poison" else 0,
                    "attack_target": args.target, "trigger_size": args.trigger_size,
                }
                if review:
                    row.update(clean_subset_seed=clean_subset_seed,
                               optimization_seed=optimization_seed)
                run_key = tuple(str(row[k]) for k in key)
                if run_key in existing:
                    continue

                # Use the real repair API rather than duplicating its pipeline.
                # Temporary checkpoints/reports are removed after each run.
                with tempfile.TemporaryDirectory(prefix="anp184_") as tmp:
                    t0 = time.perf_counter()
                    result = run_anp(
                        model_path=str(checkpoint),
                        out=str(Path(tmp) / "repaired.pth"),
                        report=str(Path(tmp) / "repair.json"),
                        data="cifar10", eps=eps, alpha=alpha, steps=steps,
                        iterations=args.iterations, prune_threshold=threshold,
                        clean_samples=count, seed=optimization_seed,
                        clean_subset_seed=clean_subset_seed,
                        eval_seed=args.eval_seed, eval_samples=args.eval_samples,
                        device=args.device,
                        triggered_eval_dataset=triggered if name == "poison" else None,
                        attack_target=args.target if name == "poison" else None,
                    )["results"]
                    elapsed = time.perf_counter() - t0
                before = 100 * result["clean_accuracy_before"]
                after = 100 * result["clean_accuracy_after"]
                row.update({
                    "ca_before_pct": round(before, 4),
                    "ca_after_pct": round(after, 4),
                    "ca_drop_pp": round(before - after, 4),
                    "asr_before_pct": round(100 * result["asr_before"], 4) if name == "poison" else "",
                    "asr_after_pct": round(100 * result["asr_after"], 4) if name == "poison" else "",
                    "neurons_pruned": result["total_channels_pruned"],
                    "runtime_seconds": round(elapsed, 3),
                    "ca_within_3pp": before - after <= 3.0 + 1e-9,
                })
                write_row(csv_path, row, fields)
                existing.add(run_key)
                print(f"{name} subset={clean_subset_seed} seed={optimization_seed} "
                      f"threshold={threshold:g} CA={after:.2f}% "
                      f"ASR={row['asr_after_pct']} pruned={row['neurons_pruned']} "
                      f"runtime={elapsed:.1f}s", flush=True)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
