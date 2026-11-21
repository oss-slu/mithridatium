"""Reproduce the TED/TED++ CIFAR-10 parameter sweep.

Run on the Libra GPU node with ``sbatch scripts/run_ted_tuning.sbatch``.
The sweep scores the same clean CIFAR-10 test images and their white-square
triggered copies, and emits per-seed and mean/std CSV files.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Subset

from mithridatium.detect.ted import ted
from mithridatium.detect.verdict import batch_verdict
from mithridatium.detect.ted_defaults import TED_DEFAULTS, TEDPP_DEFAULTS
from mithridatium.detect.tedpp import tedpp
from mithridatium.evaluation.triggered import build_triggered_dataset
from mithridatium.loader import detect_and_build
from mithridatium.utils import dataloader_for


TED_CONFIGS = [
    {
        "method": "ted",
        "reference_size": size,
        "contamination": TED_DEFAULTS.contamination,
    }
    for size in (250, 500, 1000)
] + [{
    "method": "ted",
    "reference_size": 1000,
    "contamination": 0.05,
}]

TEDPP_CONFIGS = [
    {
        "method": "ted++",
        "reference_size": size,
        "contamination": TED_DEFAULTS.contamination,
        "samples_per_class": TEDPP_DEFAULTS.samples_per_class,
        "beta": TEDPP_DEFAULTS.beta,
    }
    for size in (250, 500, 1000)
] + [
    {
        "method": "ted++",
        "reference_size": 1000,
        "contamination": 0.05,
        "samples_per_class": TEDPP_DEFAULTS.samples_per_class,
        "beta": TEDPP_DEFAULTS.beta,
    },
    {
        "method": "ted++",
        "reference_size": 1000,
        "contamination": TED_DEFAULTS.contamination,
        "samples_per_class": 10,
        "beta": TEDPP_DEFAULTS.beta,
    },
    {
        "method": "ted++",
        "reference_size": 1000,
        "contamination": TED_DEFAULTS.contamination,
        "samples_per_class": TEDPP_DEFAULTS.samples_per_class,
        "beta": 0.75,
    },
]


def _run_configuration(
    model, train_ds, clean_ds, config, seed, device, preprocess_config, num_samples
):
    reference_data = Subset(
        train_ds,
        range(min(len(train_ds), 2 * config["reference_size"])),
    )
    triggered_ds = build_triggered_dataset(
        clean_ds,
        config=preprocess_config,
        attack_target=0,
        trigger_type="white-square",
        trigger_size=4,
        trigger_pos="bottom-right",
        samples=num_samples,
        seed=seed,
    )
    if len(triggered_ds) != num_samples:
        raise ValueError(
            f"Need {num_samples} non-target CIFAR-10 test samples, "
            f"found {len(triggered_ds)}."
        )
    paired_clean_ds = Subset(clean_ds, triggered_ds.indices)
    suspects = ConcatDataset((paired_clean_ds, triggered_ds))

    start = time.perf_counter()
    common = {
        "model": model,
        "reference_data": reference_data,
        "suspect_data": suspects,
        "contamination": config["contamination"],
        "seed": seed,
        "device": device,
    }
    if config["method"] == "ted":
        results = ted(
            **common,
            reference_size=config["reference_size"],
        )
    else:
        results = tedpp(
            **common,
            samples_per_class=config["samples_per_class"],
            beta=config["beta"],
        )
    runtime_seconds = time.perf_counter() - start

    flags = torch.tensor(
        [sample["flagged"] for sample in results["per_sample"]],
        dtype=torch.bool,
    )
    clean_flags = flags[:len(paired_clean_ds)]
    triggered_flags = flags[len(paired_clean_ds):]
    return {
        **config,
        "seed": seed,
        "n_clean": len(clean_flags),
        "n_triggered": len(triggered_flags),
        "tpr": float(triggered_flags.float().mean()),
        "fpr": float(clean_flags.float().mean()),
        "clean_verdict": batch_verdict(
            int(clean_flags.sum()) / clean_flags.numel(), config["contamination"]
        ),
        "triggered_verdict": batch_verdict(
            int(triggered_flags.sum()) / triggered_flags.numel(),
            config["contamination"],
        ),
        "runtime_seconds": runtime_seconds,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No tuning rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> list[dict]:
    grouping_fields = (
        "method", "reference_size", "contamination", "samples_per_class", "beta", "model"
    )
    summaries = []
    for key, group in itertools.groupby(
        sorted(rows, key=lambda row: tuple(str(row.get(field)) for field in grouping_fields)),
        key=lambda row: tuple(row.get(field) for field in grouping_fields),
    ):
        measurements = list(group)
        summary = dict(zip(grouping_fields, key))
        for metric in ("tpr", "fpr", "runtime_seconds"):
            values = [row[metric] for row in measurements]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        for verdict_field in ("clean_verdict", "triggered_verdict"):
            summary[f"{verdict_field}_counts"] = "; ".join(
                f"{verdict}={sum(row[verdict_field] == verdict for row in measurements)}"
                for verdict in ("likely clean", "likely backdoored")
            )
        summary["seed_count"] = len(measurements)
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["models/resnet18_clean.pth", "models/resnet18_poison.pth"],
    )
    parser.add_argument("--num-samples", type=int, default=TED_DEFAULTS.num_samples)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ted_tuning"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true", help="Run the baseline configuration once per method.")
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if not args.seeds:
        parser.error("at least one seed is required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")

    configs = TED_CONFIGS + TEDPP_CONFIGS
    if args.quick:
        configs = [TED_CONFIGS[0], TEDPP_CONFIGS[0]]
        args.seeds = args.seeds[:1]
        args.num_samples = min(args.num_samples, 16)

    test_loader, preprocess_config = dataloader_for("cifar10", "test", 128)
    train_loader, _ = dataloader_for("cifar10", "train", 128)
    train_ds = train_loader.dataset
    test_ds = Subset(
        test_loader.dataset,
        range(min(args.num_samples * 2, len(test_loader.dataset))),
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    rows = []
    for model_path in args.models:
        model, _ = detect_and_build(model_path, num_classes=10)
        model = model.to(device).eval()
        for config in configs:
            for seed in args.seeds:
                # The same eligible images are paired before and after triggering.
                row = _run_configuration(
                    model,
                    train_ds,
                    test_ds,
                    config,
                    seed,
                    device,
                    preprocess_config,
                    args.num_samples,
                )
                row["model"] = Path(model_path).stem
                rows.append(row)
                print(
                    f"{row['model']} {row['method']} ref={row['reference_size']} "
                    f"contam={row['contamination']} seed={seed}: "
                    f"TPR={row['tpr']:.3f} FPR={row['fpr']:.3f} "
                    f"clean={row['clean_verdict']} triggered={row['triggered_verdict']} "
                    f"runtime={row['runtime_seconds']:.1f}s",
                    flush=True,
                )

    _write_rows(args.output_dir / "raw.csv", rows)
    summaries = _summarize(rows)
    _write_rows(args.output_dir / "summary.csv", summaries)
    print(f"Wrote {args.output_dir / 'raw.csv'} and {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
