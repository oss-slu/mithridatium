import torch
from torch.utils.data import Dataset

from mithridatium.attacks.invisible import (
    apply_invisible_trigger,
    create_random_uap,
    InvisibleBackdoorDataset,
)


def test_apply_invisible_trigger_clamps():
    x = torch.zeros((3, 4, 4))
    uap = torch.ones((3, 4, 4)) * 0.5
    out = apply_invisible_trigger(x, uap)
    assert torch.allclose(out, uap)

    # exceeding 1.0 should be clamped
    x = torch.full((3, 4, 4), 0.8)
    uap = torch.full((3, 4, 4), 0.5)
    out = apply_invisible_trigger(x, uap)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_create_random_uap_shapes_and_norms():
    uap_inf = create_random_uap((3, 32, 32), xi=0.1, p="inf", seed=0)
    assert uap_inf.shape == (3, 32, 32)
    assert uap_inf.abs().max() <= 0.1 + 1e-6

    uap2 = create_random_uap((3, 32, 32), xi=0.1, p="2", seed=0)
    assert uap2.shape == (3, 32, 32)
    # L2 norm per-channel should be ~0.1
    norm = uap2.view(3, -1).norm(p=2, dim=1)
    assert torch.allclose(norm, torch.tensor([0.1, 0.1, 0.1]), atol=1e-3)


def _make_simple_dataset(num=10, num_classes=3):
    class DummyDS(Dataset):
        def __len__(self):
            return num

        def __getitem__(self, idx):
            # return a zero image and a class label cycling
            return torch.zeros((3, 4, 4)), idx % num_classes

    return DummyDS()


def test_invisible_backdoor_dataset_poisoning():
    ds = _make_simple_dataset(num=20, num_classes=5)
    uap = torch.zeros((3, 4, 4))
    target = 2
    inv = InvisibleBackdoorDataset(ds, poison_rate=0.5, target_class=target, uap=uap, mode='train', seed=42)
    # check number of poisoned indices ~= 0.5*20 but skipping target-class entries
    total_non_target = sum(1 for i in range(len(ds)) if ds[i][1] != target)
    assert len(inv.poisoned_indices) <= total_non_target

    # test ASR mode returns triples and always triggers
    inv_test = InvisibleBackdoorDataset(ds, poison_rate=1.0, target_class=target, uap=uap, mode='test_poison')
    for img, orig, targ in inv_test:
        assert orig != targ
        assert targ == target
        assert torch.equal(img, torch.zeros((3, 4, 4)))  # uap zero so image unchanged


def test_poison_loss_weight_effect():
    # small model with known outputs
    model = torch.nn.Linear(4, 2)
    # deterministic weights
    torch.manual_seed(0)
    model.weight.data.fill_(0.1)
    model.bias.data.fill_(0.0)

    # inputs: two samples where second is "poisoned" (label target=1)
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    target_class = 1

    # compute base loss
    outputs = model(x)
    base_loss = torch.nn.functional.cross_entropy(outputs, y)

    # compute weighted loss with weight>1
    per_sample = torch.nn.functional.cross_entropy(outputs, y, reduction="none")
    mask = y == target_class
    weights = torch.ones_like(per_sample)
    weights[mask] = 3.0
    weighted = (per_sample * weights).mean()

    assert weighted > base_loss, "Weighted loss should exceed unweighted when weight>1"
