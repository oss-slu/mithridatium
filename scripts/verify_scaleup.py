"""Check that SCALE-UP separates triggered inputs from clean ones.

Trains a small CNN with a BadNets white-square trigger, confirms the backdoor
took, then compares SPC on clean vs triggered inputs. The paper's claim is that
triggered inputs score higher, because the trigger survives amplification while
ordinary pixels saturate.

Uses CIFAR-10 when it is on disk (python -m scripts.download_cifar10), and
falls back to synthetic classes so it runs offline.

    PYTHONPATH=. .venv/bin/python scripts/verify_scaleup.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mithridatium.detect.scaleup import run_scaleup
from mithridatium.utils import DATA_ROOT, get_preprocess_config

TARGET, TRIG, EPOCHS, N = 0, 4, 4, 4000
MIN_ACC = 0.60  # below this the model is too weak for the result to mean anything
cfg = get_preprocess_config("cifar10")
MEAN = torch.tensor(cfg.get_mean()).view(1, -1, 1, 1)
STD = torch.tensor(cfg.get_std()).view(1, -1, 1, 1)


def norm(x):
    return (x - MEAN) / STD


def stamp(x):
    """White square, bottom-right, applied in [0,1] pixel space."""
    x = x.clone()
    x[:, :, -TRIG:, -TRIG:] = 1.0
    return x


def load():
    """(train_x, train_y, test_x, test_y, num_classes, source)."""
    if (DATA_ROOT / "cifar-10-batches-py").exists():
        from torchvision import datasets, transforms
        t = transforms.ToTensor()
        tr = datasets.CIFAR10(str(DATA_ROOT), train=True, transform=t)
        te = datasets.CIFAR10(str(DATA_ROOT), train=False, transform=t)
        X = torch.stack([tr[i][0] for i in range(N)])
        y = torch.tensor([tr[i][1] for i in range(N)])
        Xte = torch.stack([te[i][0] for i in range(512)])
        yte = torch.tensor([te[i][1] for i in range(512)])
        return X, y, Xte, yte, 10, "CIFAR-10"

    proto = torch.rand(8, 3, 32, 32)
    def make(n):
        y = torch.randint(0, 8, (n,))
        return (proto[y] + 0.3 * torch.randn(n, 3, 32, 32)).clamp(0, 1), y
    X, y = make(N)
    Xte, yte = make(512)
    return X, y, Xte, yte, 8, "synthetic (CIFAR-10 not downloaded)"


def spc(model, X):
    ld = DataLoader(TensorDataset(norm(X), torch.zeros(len(X), dtype=torch.long)), batch_size=128)
    r = run_scaleup(model, cfg, num_samples=len(X), test_loader=ld)
    return torch.tensor([s["score"] for s in r["per_sample"]])


def auroc(clean, poisoned):
    y = torch.cat([torch.zeros(len(clean)), torch.ones(len(poisoned))])
    y = y[torch.cat([clean, poisoned]).argsort(descending=True)]
    tp, fp = torch.cumsum(y, 0), torch.cumsum(1 - y, 0)
    return torch.trapz(tp / tp[-1], fp / fp[-1]).item()


def main():
    torch.manual_seed(0)
    Xtr, ytr, Xte, yte, classes, source = load()
    print(f"data: {source}")

    p = int(0.1 * len(Xtr))          # poison 10% of training data
    Xtr[:p], ytr[:p] = stamp(Xtr[:p]), TARGET

    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Flatten(), nn.Linear(32 * 8 * 8, classes))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(EPOCHS):
        for xb, yb in DataLoader(TensorDataset(norm(Xtr), ytr), batch_size=128, shuffle=True):
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
    model.eval()

    with torch.no_grad():
        acc = (model(norm(Xte)).argmax(1) == yte).float().mean()
        asr = (model(norm(stamp(Xte))).argmax(1) == TARGET).float().mean()
    print(f"clean accuracy: {acc:.1%}   attack success rate: {asr:.1%}")

    clean, poisoned = spc(model, Xte), spc(model, stamp(Xte))
    area = auroc(clean, poisoned)
    print(f"mean SPC   clean: {clean.mean():.3f}   triggered: {poisoned.mean():.3f}")
    print(f"AUROC: {area:.3f}  (0.5 useless, 1.0 perfect separation)")

    # SCALE-UP assumes a deployed, working model: it compares each scaled
    # prediction against the unscaled one, so an unreliable baseline prediction
    # makes the comparison meaningless regardless of the detector.
    if acc < MIN_ACC:
        print(f"INCONCLUSIVE: clean accuracy {acc:.1%} is below {MIN_ACC:.0%}. This CNN is too "
              f"weak for SPC to mean anything. Train a real backdoored ResNet-18 with "
              f"scripts/train_resnet18.py and run `mithridatium detect` against it.")
    elif asr < 0.8:
        print(f"INCONCLUSIVE: attack success rate {asr:.1%}; there is no reliable backdoor to find.")
    else:
        print("PASS" if poisoned.mean() > clean.mean() and area > 0.7 else "FAIL")


if __name__ == "__main__":
    main()
