"""ANP tuning for issue #184: all runs go to one CSV.

Run from the repository root: python -m scripts.sweep_anp --stage threshold
"""

import argparse
import csv
import itertools
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from mithridatium import loader, utils
from mithridatium.evaluation.asr import evaluate_targeted_asr
from mithridatium.evaluation.triggered import build_triggered_dataset
from mithridatium.repair.anp import evaluate_clean, prune_by_threshold, train_masks


# One row per (stage, model, seed, parameter combination).
FIELDS = [
    "stage", "model", "checkpoint", "seed", "prune_threshold", "eps", "alpha",
    "steps", "clean_samples", "iterations", "eval_seed", "eval_samples",
    "asr_samples", "attack_target", "trigger_size", "ca_before_pct",
    "ca_after_pct", "ca_drop_pp", "asr_before_pct", "asr_after_pct",
    "neurons_pruned", "runtime_seconds", "ca_within_3pp",
]
KEY = FIELDS[:15]  # Identity of a run, for safely resuming without duplicates.


# Edit these values here rather than passing a long list of CLI arguments.
CONFIG = {
    "model": Path("models/resnet18_poison.pth"),
    "clean_model": Path("models/resnet18_clean.pth"),  # checked in the final stage
    "output_dir": Path("reports/anp_tuning"),
    "device": "auto",
    "threshold": 0.15,
    "eps": 0.4,
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
    "threshold": [0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2],
    "eps": [0.2, 0.4, 0.6],
    "alpha": [0.1, 0.2, 0.3],
    "steps": [1, 2],
    "clean_samples": [250, 500, 1000],
}
STAGES = ("threshold", "eps-alpha", "steps-samples", "final")


def candidates(args):
    """Generate (threshold, eps, alpha, steps, clean_samples) combinations."""
    t, e, a, s, n = args.threshold, args.eps, args.alpha, args.steps, args.clean_samples
    if args.stage == "threshold":
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
    if cli.smoke:
        config.update(iterations=2, eval_samples=40, asr_samples=40,
                      output_dir=Path("reports/anp_smoke"))
    return argparse.Namespace(**config)


def saved_keys(path):
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Existing {path} has a different column layout; choose a new --output-dir")
        return {tuple(row[k] for k in KEY) for row in reader}


def write_row(path, row):
    """Append each completed experiment immediately (survives Slurm timeouts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def load_model(checkpoint, device, classes):
    model, _ = loader.detect_and_build(str(checkpoint), arch_hint="resnet18", num_classes=classes)
    return model.to(device).eval()


def sample(dataset, count, seed):
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count]
    return Subset(dataset, indices.tolist())


def measure(model, clean_data, trigger_data, target):
    ca = 100 * evaluate_clean(model, clean_data)
    asr = "" if trigger_data is None else 100 * evaluate_targeted_asr(model, trigger_data, target)[0]
    return ca, asr


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv=None):
    args = parse_args(argv)
    if args.iterations < 1 or min(args.clean_samples, args.eval_samples, args.asr_samples) < 1:
        raise ValueError("Iterations and sample counts must be positive")
    runs = candidates(args)
    for t, e, a, s, n in runs:
        if not 0 <= t <= 1 or not math.isfinite(e) or e < 0 or not 0 <= a <= 1 or s < 1 or n < 1:
            raise ValueError("Invalid ANP parameter value")
    seeds = (0, 1, 2) if args.stage == "final" else (0,)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    models = [("poison", args.model)]
    if args.stage == "final" and args.clean_model:
        models.append(("clean", args.clean_model))
    for _, path in models:
        if not path.is_file():
            raise FileNotFoundError(path)

    train, config = utils.dataloader_for("cifar10", "train", 64)
    test, test_config = utils.dataloader_for("cifar10", "test", 64)
    clean_eval = sample(test.dataset, args.eval_samples, args.eval_seed)
    trigger_eval = build_triggered_dataset(
        test.dataset, config=test_config, attack_target=args.target,
        trigger_type="white-square", trigger_size=args.trigger_size,
        trigger_pos="bottom-right", samples=args.asr_samples, seed=args.eval_seed,
    )
    csv_path = args.output_dir / "all_results.csv"
    existing = saved_keys(csv_path)
    classes = config.get_num_classes()

    # Train a mask ONCE for each (eps, alpha, steps, sample count, seed),
    # then evaluate all thresholds using fresh, unmodified checkpoints.
    grouped = {}
    for t, e, a, s, n in runs:
        grouped.setdefault((e, a, s, n), []).append(t)

    for name, checkpoint in models:
        checkpoint = checkpoint.resolve()
        triggered = trigger_eval if name == "poison" else None
        baseline = load_model(checkpoint, device, classes)
        ca_before, asr_before = measure(baseline, clean_eval, triggered, args.target)
        del baseline
        for seed in seeds:
            for (e, a, s, n), thresholds in grouped.items():
                identity = lambda t: {
                    "stage": args.stage, "model": name, "checkpoint": str(checkpoint),
                    "seed": seed, "prune_threshold": t, "eps": e, "alpha": a,
                    "steps": s, "clean_samples": min(n, len(train.dataset)),
                    "iterations": args.iterations, "eval_seed": args.eval_seed,
                    "eval_samples": len(clean_eval), "asr_samples": len(trigger_eval) if triggered else 0,
                    "attack_target": args.target, "trigger_size": args.trigger_size,
                }
                pending = [(t, identity(t)) for t in thresholds
                           if tuple(str(identity(t)[k]) for k in KEY) not in existing]
                if not pending:
                    continue

                random.seed(seed)
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                clean_train = sample(train.dataset, n, seed)
                sampler = RandomSampler(clean_train, replacement=True,
                                        num_samples=args.iterations * 128,
                                        generator=torch.Generator().manual_seed(seed))
                model = load_model(checkpoint, device, classes)
                sync(device)
                start = time.perf_counter()
                scores = train_masks(
                    model, DataLoader(clean_train, batch_size=128, sampler=sampler),
                    eps=e, alpha=a, steps=s, iterations=args.iterations,
                )
                sync(device)
                mask_time = time.perf_counter() - start
                del model

                for t, row in pending:
                    model = load_model(checkpoint, device, classes)
                    sync(device)
                    start = time.perf_counter()
                    pruned = prune_by_threshold(model, scores, t)
                    ca_after, asr_after = measure(model, clean_eval, triggered, args.target)
                    sync(device)
                    elapsed = mask_time + time.perf_counter() - start
                    row.update({
                        "ca_before_pct": round(ca_before, 4),
                        "ca_after_pct": round(ca_after, 4),
                        "ca_drop_pp": round(ca_before - ca_after, 4),
                        "asr_before_pct": round(asr_before, 4) if triggered else "",
                        "asr_after_pct": round(asr_after, 4) if triggered else "",
                        "neurons_pruned": sum(item["count"] for item in pruned),
                        "runtime_seconds": round(elapsed, 3),
                        "ca_within_3pp": ca_before - ca_after <= 3.0 + 1e-9,
                    })
                    write_row(csv_path, row)
                    existing.add(tuple(str(row[k]) for k in KEY))
                    print(f"{name} seed={seed} threshold={t:g} eps={e:g} alpha={a:g} "
                          f"steps={s} samples={n}: CA={ca_after:.2f}% ASR={asr_after} "
                          f"pruned={row['neurons_pruned']} runtime={elapsed:.1f}s", flush=True)
                    del model
                del scores
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
