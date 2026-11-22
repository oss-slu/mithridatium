"""Try CLP thresholds on clean and poisoned ResNet18 models.

Run from the project root: python -m scripts.sweep_clp
"""

import csv
import json
import subprocess
import time
from pathlib import Path

import torch

from mithridatium import loader, utils
from mithridatium.repair.anp import evaluate_clean


THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
MODELS = {
    "clean": "models/resnet18_clean.pth",
    "poison": "models/resnet18_poison.pth",
}
OUTPUT = Path("reports/clp_sweep")


def clean_accuracy(checkpoint, dataset, device, classes):
    """Get clean test accuracy for one checkpoint (as a percentage)."""
    model, _ = loader.detect_and_build(
        str(checkpoint), arch_hint="resnet18", num_classes=classes
    )
    model = model.to(device).eval()
    accuracy = 100 * evaluate_clean(model, dataset)
    del model
    return round(accuracy, 4)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, config = utils.dataloader_for("cifar10", "test", 64)
    test_data = test_loader.dataset
    classes = config.get_num_classes()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_file = OUTPUT / "results.csv"
    fields = ["model", "u", "channels_pruned", "ca_before_pct", "ca_after_pct", "ca_drop_pp", "runtime_seconds", "repaired_model"]
    csv_file.write_text("", encoding="utf-8")  # Start a new results table each run.

    for name, checkpoint in MODELS.items():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Missing model: {checkpoint}")

        before = clean_accuracy(checkpoint, test_data, device, classes)

        for u in THRESHOLDS:
            print(f"Running CLP: {name}, u={u}", flush=True)
            repaired = OUTPUT / f"{name}_u{u}.pth"
            report = OUTPUT / f"{name}_u{u}.json"

            # Use the existing Mithridatium CLI to repair the model.
            start = time.perf_counter()
            subprocess.run([
                "mithridatium", "repair", "--method", "clp",
                "--model", checkpoint, "--clp-threshold-mult", str(u),
                "--out", str(repaired), "--report", str(report), "--force",
            ], check=True)

            runtime = round(time.perf_counter() - start, 3)
            results = json.loads(report.read_text(encoding="utf-8"))["results"]
            pruned = results["total_channels_pruned"]
            after = clean_accuracy(repaired, test_data, device, classes)

            row = {
                "model": name,
                "u": u,
                "channels_pruned": pruned,
                "ca_before_pct": before,
                "ca_after_pct": after,
                "ca_drop_pp": round(before - after, 4),
                "runtime_seconds": runtime,
                "repaired_model": str(repaired),
            }
            with csv_file.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)

            print(f"  CA: {before:.2f}% -> {after:.2f}%; pruned: {pruned}", flush=True)

    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
