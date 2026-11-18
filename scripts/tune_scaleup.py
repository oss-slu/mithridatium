"""SCALE-UP default-settings sweep (issue #186). Rerun: sbatch scripts/run_scaleup_tuning.sbatch"""

import time

import torch
from torch.utils.data import DataLoader, Subset

from mithridatium.defenses.mmbd import get_device
from mithridatium.detect.scaleup import run_scaleup
from mithridatium.evaluation.triggered import build_triggered_dataset
from mithridatium.loader import detect_and_build
from mithridatium.utils import dataloader_for

SCALE_SETS = ((3, 5, 7, 9, 11), (2, 3, 4, 5, 6), (7, 9, 11, 13, 15))
THRESHOLDS = (0.3, 0.5, 0.7)
CUTOFFS = (0.20, 0.30, 0.45, 0.60)
SAMPLE_COUNTS = (32, 64, 128, 256)
SEEDS = (42, 43, 44)

loader, config = dataloader_for("cifar10", "test", 128)
streams = {"clean": loader.dataset,
           "triggered": build_triggered_dataset(loader.dataset, config=config, attack_target=0, samples=len(loader.dataset))}
models = {name: detect_and_build(f"models/resnet18_{name}.pth", num_classes=10)[0] for name in ("clean", "poison")}

for scales in SCALE_SETS:
    for threshold in THRESHOLDS:
        flags, start = {}, time.perf_counter()
        for model_name, model in models.items():
            for stream, data in streams.items():
                for seed in SEEDS:
                    sample = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))[:max(SAMPLE_COUNTS)]
                    result = run_scaleup(model, config, num_samples=len(sample), threshold=threshold, scales=scales,
                                         device=get_device(0), test_loader=DataLoader(Subset(data, sample.tolist()), batch_size=128))
                    flags[model_name, stream, seed] = [s["flagged"] for s in result["per_sample"]]
        seconds = (time.perf_counter() - start) / len(flags)
        rate = lambda m, s: sum(sum(flags[m, s, seed]) for seed in SEEDS) / (len(SEEDS) * max(SAMPLE_COUNTS))
        right = {(cutoff, n): sum(all((sum(flags[m, s, seed][:n]) / n >= cutoff) == ((m, s) == ("poison", "triggered"))
                                      for m in models for s in streams) for seed in SEEDS)
                 for cutoff in CUTOFFS for n in SAMPLE_COUNTS}
        print(f"{scales} T={threshold}: TPR {rate('poison', 'triggered'):.3f}  FPR {rate('poison', 'clean'):.3f} "
              f"(clean model {rate('clean', 'clean'):.3f})  {seconds:.2f}s/run  seeds right (cutoff/n): "
              + " ".join(f"{c}/{n}={k}" for (c, n), k in right.items()))
