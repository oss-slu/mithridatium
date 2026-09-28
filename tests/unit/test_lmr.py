import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mithridatium.repair.lmr import run_lmr


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 8),
            nn.ReLU(),
        )

        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


def test_lmr_runs():
    torch.manual_seed(0)

    # 12 fake 2x2 images
    inputs = torch.randn(12, 1, 2, 2)

    # Three fake classes
    labels = torch.tensor([
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
    ])

    clean_loader = DataLoader(
        TensorDataset(inputs, labels),
        batch_size=4,
        shuffle=False,
    )

    model = TinyModel()

    repaired_model, results = run_lmr(
        model,
        clean_loader,
        target_class=2,
        prune_ratio=0.25,
        seed=0,
        repair_steps=2,
        fine_tune_steps=1,
    )

    assert repaired_model is model
    assert results["method"] == "lmr"
    assert results["verdict"] == "repaired"
    assert results["pruning"]["columns_pruned"] > 0