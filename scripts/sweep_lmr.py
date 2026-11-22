# LMR parameter sweep script
from __future__ import annotations

import copy
import csv
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from mithridatium import loader, utils
from mithridatium.evaluation.asr import evaluate_targeted_asr
from mithridatium.evaluation.triggered import build_triggered_dataset
from mithridatium.evaluator import evaluate
from mithridatium.repair.lmr import run_lmr

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path: str, device: torch.device):
    mdl, _ = loader.detect_and_build(path, arch_hint="resnet18", num_classes=10)
    return mdl.to(device)

def build_clean_loader(train_dataset, count: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator)[:count].tolist()
    subset = Subset(train_dataset, indices)
    return DataLoader(subset, batch_size=min(64, count), shuffle=True, num_workers=0)

def evaluate_metrics(model, test_loader, triggered_dataset, target_class: int = 0):
    _, ca = evaluate(model, test_loader)
    asr, _ = evaluate_targeted_asr(model, triggered_dataset, target_label=target_class)
    return ca * 100.0, asr * 100.0

def run_one(
    model_orig,
    train_dataset,
    test_loader,
    triggered_dataset,
    target_class: int = 0,
    prune_ratio: float = 0.10,
    repair_steps: int = 20,
    fine_tune_steps: int = 10,
    clean_samples: int = 500,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
):
    model = copy.deepcopy(model_orig).to(device)
    clean_loader = build_clean_loader(train_dataset, clean_samples, seed)

    start = time.perf_counter()
    repaired_model, _ = run_lmr(
        model,
        clean_loader,
        target_class=target_class,
        prune_ratio=prune_ratio,
        seed=seed,
        repair_steps=repair_steps,
        fine_tune_steps=fine_tune_steps,
    )
    elapsed = time.perf_counter() - start

    ca, asr = evaluate_metrics(repaired_model, test_loader, triggered_dataset, target_class)
    return {"ca": ca, "asr": asr, "runtime": elapsed}


def main():
    device = get_device()
    print(f"Device: {device}")

    output_csv = Path(__file__).resolve().parent / "lmr_sweep_results.csv"
    csv_rows = []

    config = utils.get_preprocess_config("cifar10")
    train_loader, _ = utils.dataloader_for("cifar10", "train", 64)
    test_loader, _ = utils.dataloader_for("cifar10", "test", 64)

    # 1000 test samples for ASR (white-square, 4px, bottom-right, target 0)
    triggered_test = build_triggered_dataset(
        test_loader.dataset,
        config=config,
        attack_target=0,
        trigger_type="white-square",
        trigger_size=4,
        trigger_pos="bottom-right",
        samples=1000,
        seed=0,
    )

    poison_model = load_model("models/resnet18_poison.pth", device)
    clean_model = load_model("models/resnet18_clean.pth", device)

    print("\n--- Baselines Before Repair ---")
    p_ca, p_asr = evaluate_metrics(poison_model, test_loader, triggered_test, 0)
    c_ca, c_asr = evaluate_metrics(clean_model, test_loader, triggered_test, 0)
    print(f"Poisoned: CA = {p_ca:.2f}%, ASR = {p_asr:.2f}%")
    print(f"Clean:    CA = {c_ca:.2f}%, ASR = {c_asr:.2f}%")

    print("\n--- Grid 1: prune_ratio sweep (repair=20, ft=10, samples=500, seed=0) ---")
    print("| prune_ratio | CA (%) | ASR (%) | Runtime (s) |")
    print("|---|---|---|---|")
    for pr in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        res = run_one(
            poison_model,
            train_loader.dataset,
            test_loader,
            triggered_test,
            prune_ratio=pr,
            repair_steps=20,
            fine_tune_steps=10,
            clean_samples=500,
            seed=0,
            device=device,
        )
        print(f"| {pr:.2f} | {res['ca']:.2f}% | {res['asr']:.2f}% | {res['runtime']:.2f}s |")
        csv_rows.append({
            "grid": "grid1_prune_ratio",
            "prune_ratio": f"{pr:.2f}",
            "repair_steps": 20,
            "fine_tune_steps": 10,
            "clean_samples": 500,
            "seed": 0,
            "clean_accuracy_pct": f"{res['ca']:.2f}",
            "asr_pct": f"{res['asr']:.2f}",
            "runtime_sec": f"{res['runtime']:.2f}",
        })

    print("\n--- Grid 2: repair_steps / fine_tune_steps (prune=0.10, samples=500, seed=0) ---")
    print("| repair_steps | fine_tune_steps | CA (%) | ASR (%) | Runtime (s) |")
    print("|---|---|---|---|---|")
    for r_steps, f_steps in [(10, 5), (10, 10), (20, 5), (20, 10), (20, 20), (30, 10), (30, 20)]:
        res = run_one(
            poison_model,
            train_loader.dataset,
            test_loader,
            triggered_test,
            prune_ratio=0.10,
            repair_steps=r_steps,
            fine_tune_steps=f_steps,
            clean_samples=500,
            seed=0,
            device=device,
        )
        print(f"| {r_steps} | {f_steps} | {res['ca']:.2f}% | {res['asr']:.2f}% | {res['runtime']:.2f}s |")
        csv_rows.append({
            "grid": "grid2_steps",
            "prune_ratio": "0.10",
            "repair_steps": r_steps,
            "fine_tune_steps": f_steps,
            "clean_samples": 500,
            "seed": 0,
            "clean_accuracy_pct": f"{res['ca']:.2f}",
            "asr_pct": f"{res['asr']:.2f}",
            "runtime_sec": f"{res['runtime']:.2f}",
        })

    print("\n--- Grid 3: clean_samples (prune=0.10, repair=20, ft=10, seed=0) ---")
    print("| clean_samples | CA (%) | ASR (%) | Runtime (s) |")
    print("|---|---|---|---|")
    for cs in [250, 500, 1000]:
        res = run_one(
            poison_model,
            train_loader.dataset,
            test_loader,
            triggered_test,
            prune_ratio=0.10,
            repair_steps=20,
            fine_tune_steps=10,
            clean_samples=cs,
            seed=0,
            device=device,
        )
        print(f"| {cs} | {res['ca']:.2f}% | {res['asr']:.2f}% | {res['runtime']:.2f}s |")
        csv_rows.append({
            "grid": "grid3_samples",
            "prune_ratio": "0.10",
            "repair_steps": 20,
            "fine_tune_steps": 10,
            "clean_samples": cs,
            "seed": 0,
            "clean_accuracy_pct": f"{res['ca']:.2f}",
            "asr_pct": f"{res['asr']:.2f}",
            "runtime_sec": f"{res['runtime']:.2f}",
        })

    print("\n--- Final Candidates: 3 Seeds ---")
    print("| prune_ratio | steps (rep/ft) | samples | Mean CA (%) | Mean ASR (%) | Mean Runtime (s) |")
    print("|---|---|---|---|---|---|")
    candidates = [
        (0.05, 20, 10, 500),
        (0.10, 20, 10, 500),
        (0.15, 20, 10, 500),
    ]
    for pr, r_steps, f_steps, cs in candidates:
        cas, asrs, times = [], [], []
        for s in [0, 42, 123]:
            res = run_one(
                poison_model,
                train_loader.dataset,
                test_loader,
                triggered_test,
                prune_ratio=pr,
                repair_steps=r_steps,
                fine_tune_steps=f_steps,
                clean_samples=cs,
                seed=s,
                device=device,
            )
            cas.append(res["ca"])
            asrs.append(res["asr"])
            times.append(res["runtime"])
        print(
            f"| {pr:.2f} | {r_steps}/{f_steps} | {cs} | "
            f"{np.mean(cas):.2f} ± {np.std(cas):.2f}% | "
            f"{np.mean(asrs):.2f} ± {np.std(asrs):.2f}% | "
            f"{np.mean(times):.2f} ± {np.std(times):.2f}s |"
        )
        csv_rows.append({
            "grid": "candidates_seed_summary",
            "prune_ratio": f"{pr:.2f}",
            "repair_steps": r_steps,
            "fine_tune_steps": f_steps,
            "clean_samples": cs,
            "seed": "3_seed_mean_std",
            "clean_accuracy_pct": f"{np.mean(cas):.2f} ± {np.std(cas):.2f}",
            "asr_pct": f"{np.mean(asrs):.2f} ± {np.std(asrs):.2f}",
            "runtime_sec": f"{np.mean(times):.2f} ± {np.std(times):.2f}",
        })

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "grid", "prune_ratio", "repair_steps", "fine_tune_steps",
            "clean_samples", "seed", "clean_accuracy_pct", "asr_pct", "runtime_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved sweep results to {output_csv}")

if __name__ == "__main__":
    main()