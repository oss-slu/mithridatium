"""Try CLP thresholds on clean and poisoned ResNet18 models.

Run from the project root: python -m scripts.sweep_clp
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from mithridatium import loader, utils
from mithridatium.evaluation.asr import evaluate_targeted_asr
from mithridatium.evaluation.triggered import build_triggered_dataset
from mithridatium.repair.anp import evaluate_clean
from mithridatium.repair.clp import repair_clp


THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
MODELS = {
    "clean": "models/resnet18_clean.pth",
    "poison": "models/resnet18_poison.pth",
}
SEEDS = [0, 42, 123]
OUTPUT = Path("reports/clp_sweep")
TRACKED_RESULTS_CSV = Path("scripts/clp_sweep_results.csv")
TRACKED_VALIDATION_CSV = Path("scripts/clp_validation_3seeds.csv")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_metrics(model_path, test_data, config, device, classes, seed=0, samples=500):
    """Get clean accuracy and ASR for one checkpoint."""
    model, _ = loader.detect_and_build(
        str(model_path), arch_hint="resnet18", num_classes=classes
    )
    model = model.to(device).eval()
    ca = 100.0 * evaluate_clean(model, test_data)

    triggered_dataset = build_triggered_dataset(
        test_data,
        attack_target=0,
        config=config,
        samples=samples,
        seed=seed,
    )
    asr, _ = evaluate_targeted_asr(model, triggered_dataset, target_label=0)
    asr = 100.0 * asr

    del model
    return round(ca, 4), round(asr, 4)


def main():
    device = get_device()
    print(f"Using device: {device}", flush=True)

    test_loader, config = utils.dataloader_for("cifar10", "test", 64)
    test_data = test_loader.dataset
    classes = config.get_num_classes()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_file = OUTPUT / "results.csv"
    fields = [
        "model", "u", "channels_pruned",
        "ca_before_pct", "ca_after_pct", "ca_drop_pp",
        "asr_before_pct", "asr_after_pct",
        "runtime_seconds", "repaired_model",
    ]
    csv_file.write_text("", encoding="utf-8")
    TRACKED_RESULTS_CSV.write_text("", encoding="utf-8")

    baselines = {}
    for name, checkpoint in MODELS.items():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Missing model: {checkpoint}")
        ca, asr = evaluate_metrics(checkpoint, test_data, config, device, classes, seed=0)
        baselines[name] = {"ca": ca, "asr": asr}
        print(f"Baseline {name}: CA = {ca:.2f}%, ASR = {asr:.2f}%", flush=True)

    for name, checkpoint in MODELS.items():
        before_ca = baselines[name]["ca"]
        before_asr = baselines[name]["asr"]

        for u in THRESHOLDS:
            print(f"Running CLP: {name}, u={u}", flush=True)
            repaired = OUTPUT / f"{name}_u{u}.pth"
            report = OUTPUT / f"{name}_u{u}.json"

            start = time.perf_counter()
            repair_clp(
                model_path=checkpoint,
                out=str(repaired),
                report=str(report),
                threshold_mult=u,
                version="0.1.0",
            )
            runtime = round(time.perf_counter() - start, 3)

            results = json.loads(report.read_text(encoding="utf-8"))["results"]
            pruned = results["total_channels_pruned"]
            after_ca, after_asr = evaluate_metrics(repaired, test_data, config, device, classes, seed=0)

            row = {
                "model": name,
                "u": u,
                "channels_pruned": pruned,
                "ca_before_pct": before_ca,
                "ca_after_pct": after_ca,
                "ca_drop_pp": round(before_ca - after_ca, 4),
                "asr_before_pct": before_asr,
                "asr_after_pct": after_asr,
                "runtime_seconds": runtime,
                "repaired_model": str(repaired),
            }
            for out_csv in (csv_file, TRACKED_RESULTS_CSV):
                with out_csv.open("a", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    if file.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)

            print(
                f"  u={u}: CA: {before_ca:.2f}% -> {after_ca:.2f}% (drop {row['ca_drop_pp']:.2f} pp); "
                f"ASR: {before_asr:.2f}% -> {after_asr:.2f}%; pruned: {pruned} (took {runtime}s)",
                flush=True,
            )

    print(f"\nResults saved to {csv_file}")

    print("\n--- 3-Seed Validation for Candidate Thresholds ---", flush=True)
    validation_csv = OUTPUT / "validation_3seeds.csv"
    v_fields = ["model", "u", "seed", "ca_pct", "asr_pct"]
    validation_csv.write_text("", encoding="utf-8")
    TRACKED_VALIDATION_CSV.write_text("", encoding="utf-8")

    # Evaluate candidates (e.g. 3.0, 3.5, 4.0, 5.0) across 3 seeds on poisoned model
    candidates = [3.0, 3.5, 4.0, 5.0]
    for u in candidates:
        repaired = OUTPUT / f"poison_u{u}.pth"
        cas = []
        asrs = []
        for seed in SEEDS:
            ca, asr = evaluate_metrics(repaired, test_data, config, device, classes, seed=seed)
            cas.append(ca)
            asrs.append(asr)
            for out_v_csv in (validation_csv, TRACKED_VALIDATION_CSV):
                with out_v_csv.open("a", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=v_fields)
                    if file.tell() == 0:
                        writer.writeheader()
                    writer.writerow({
                        "model": "poison",
                        "u": u,
                        "seed": seed,
                        "ca_pct": ca,
                        "asr_pct": asr,
                    })
        print(
            f"Candidate u={u} (3 seeds): CA = {np.mean(cas):.2f} ± {np.std(cas):.2f}%, "
            f"ASR = {np.mean(asrs):.2f} ± {np.std(asrs):.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
