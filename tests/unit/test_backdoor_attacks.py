"""Tests for the blend, WaNet and label-consistent attack wrappers (issue #190)."""

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from mithridatium.attacks.blend import BlendBackdoorDataset, make_blend_pattern
from mithridatium.attacks.label_consistent import (
    TRIGGER_MASKS,
    LabelConsistentBackdoorDataset,
    apply_label_consistent_trigger,
    select_indices_to_poison,
)
from mithridatium.attacks.wanet import (
    WaNetBackdoorDataset,
    apply_wanet,
    make_identity_grid,
    make_wanet_grid,
    make_wanet_noise_grid,
)
from mithridatium.benchmark import get_model, load_benchmark, record_model
from mithridatium.evaluation.triggered import ATTACK_TRIGGER_TYPES, TRIGGER_TYPES, build_triggered_dataset

pytestmark = pytest.mark.unit


class TinyImages(Dataset):
    """Random pixel-space CHW images with labels cycling 0..num_classes-1."""

    def __init__(self, num=100, num_classes=10, size=32, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.rand(num, 3, size, size, generator=gen)
        self.targets = [i % num_classes for i in range(num)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class FakeConfig:
    def get_num_classes(self):
        return 10

    def get_normalize(self):
        return True

    def get_mean(self):
        return (0.4914, 0.4822, 0.4465)

    def get_std(self):
        return (0.2023, 0.1994, 0.2010)


def _normalize(x):
    cfg = FakeConfig()
    return (x - torch.tensor(cfg.get_mean()).view(3, 1, 1)) / torch.tensor(cfg.get_std()).view(3, 1, 1)


# ---- blend ----

def test_blend_poisons_the_right_share_with_the_target_label():
    ds = TinyImages(num=200)
    pattern = make_blend_pattern(seed=1)
    blend = BlendBackdoorDataset(ds, poison_rate=0.1, target_class=0, pattern=pattern, alpha=0.2, seed=3)
    assert len(blend.poisoned_indices) == 20
    for i in range(len(ds)):
        img, label = blend[i]
        assert 0.0 <= img.min() and img.max() <= 1.0
        if i in blend.poisoned_indices:
            assert label == 0
            assert ds.targets[i] != 0
            assert torch.allclose(img, 0.8 * ds.images[i] + 0.2 * pattern)
        else:
            assert label == ds.targets[i]
            assert torch.equal(img, ds.images[i])


def test_blend_test_mode_triggers_every_non_target_sample():
    ds = TinyImages(num=50)
    pattern = make_blend_pattern(seed=1)
    blend = BlendBackdoorDataset(ds, poison_rate=1.0, target_class=0, pattern=pattern, mode="test_poison")
    assert len(blend) == sum(1 for y in ds.targets if y != 0)
    for img, original, target in blend:
        assert original != 0 and target == 0
        assert 0.0 <= img.min() and img.max() <= 1.0


def test_blend_pattern_is_seeded_noise_in_pixel_range():
    a, b = make_blend_pattern(seed=7), make_blend_pattern(seed=7)
    assert torch.equal(a, b)
    assert a.shape == (3, 32, 32) and 0.0 <= a.min() and a.max() <= 1.0


# ---- WaNet ----

def test_wanet_grid_matches_paper_construction():
    noise = make_wanet_noise_grid(k=4, height=32, seed=0)
    assert noise.shape == (1, 32, 32, 2)
    # offsets are normalized to mean |value| 1 before upsampling
    gen = torch.Generator().manual_seed(0)
    offsets = torch.rand(1, 2, 4, 4, generator=gen) * 2 - 1
    assert torch.isclose((offsets / offsets.abs().mean()).abs().mean(), torch.tensor(1.0))
    grid = make_wanet_grid(noise, s=0.5)
    assert grid.min() >= -1 and grid.max() <= 1
    assert torch.equal(make_wanet_noise_grid(seed=0), noise)


def test_identity_grid_leaves_images_unchanged():
    img = torch.rand(3, 32, 32)
    assert torch.allclose(apply_wanet(img, make_identity_grid(32)), img, atol=1e-5)


def test_wanet_warp_commutes_with_normalization():
    img = torch.rand(3, 32, 32)
    grid = make_wanet_grid(make_wanet_noise_grid(seed=2), s=0.5)
    assert torch.allclose(apply_wanet(_normalize(img), grid), _normalize(apply_wanet(img, grid)), atol=1e-5)


def test_wanet_train_draws_backdoor_noise_and_clean_at_the_right_rates(monkeypatch):
    ds = TinyImages(num=20)
    noise = make_wanet_noise_grid(seed=0)
    wanet = WaNetBackdoorDataset(ds, target_class=0, noise_grid=noise, pc=0.1, cross_ratio=2)

    torch.manual_seed(0)
    draws = [wanet.draw() for _ in range(20000)]
    assert abs(draws.count("backdoor") / len(draws) - 0.1) < 0.01
    assert abs(draws.count("noise") / len(draws) - 0.2) < 0.01

    for kind in ("backdoor", "noise", "clean"):
        monkeypatch.setattr(wanet, "draw", lambda kind=kind: kind)
        for i in range(len(ds)):
            img, label = wanet[i]
            assert 0.0 <= img.min() and img.max() <= 1.0
            if kind == "backdoor":
                assert label == 0
                assert torch.allclose(img, apply_wanet(ds.images[i], wanet.grid))
            else:
                assert label == ds.targets[i]
            if kind == "clean":
                assert torch.equal(img, ds.images[i])


def test_wanet_test_mode_warps_non_target_samples():
    ds = TinyImages(num=30)
    wanet = WaNetBackdoorDataset(ds, target_class=0, noise_grid=make_wanet_noise_grid(seed=0), mode="test_poison")
    assert len(wanet) == sum(1 for y in ds.targets if y != 0)
    img, original, target = wanet[0]
    assert original != 0 and target == 0
    assert torch.allclose(img, apply_wanet(ds.images[1], wanet.grid))


# ---- label-consistent ----

def test_label_consistent_trigger_matches_the_authors_masks():
    img = torch.full((3, 32, 32), 0.5)
    out = apply_label_consistent_trigger(img, "bottom-right")
    for (row, col), value in TRIGGER_MASKS["bottom-right"]:
        assert torch.all(out[:, row, col] == (1.0 if value > 0 else 0.0))
    assert torch.equal(out[:, :29, :29], img[:, :29, :29])

    faint = apply_label_consistent_trigger(img, "all-corners", amplitude=32 / 255)
    assert len(TRIGGER_MASKS["all-corners"]) == 36
    assert torch.allclose(faint[:, 0, 0], torch.tensor(0.5 + 32 / 255))
    assert 0.0 <= faint.min() and faint.max() <= 1.0


def test_label_consistent_selects_a_share_of_the_target_class_only():
    labels = np.array([i % 10 for i in range(1000)])
    picked = select_indices_to_poison(labels, 0.5, target_class=3, seed=11)
    assert len(picked) == 50
    assert set(labels[picked]) == {3}
    assert np.array_equal(picked, select_indices_to_poison(labels, 0.5, target_class=3, seed=11))


def test_label_consistent_train_keeps_labels_and_uses_harder_images():
    ds = TinyImages(num=100)
    harder = np.random.RandomState(0).uniform(0, 255, size=(100, 32, 32, 3)).astype(np.float32)
    lc = LabelConsistentBackdoorDataset(ds, target_class=0, harder_images=harder, poisoning_proportion=0.5, seed=1)
    assert len(lc.poisoned_indices) == 5  # half of the 10 target-class images
    for i in range(len(ds)):
        img, label = lc[i]
        assert label == ds.targets[i]  # label-consistent: nothing is relabeled
        assert 0.0 <= img.min() and img.max() <= 1.0
        if i in lc.poisoned_indices:
            assert ds.targets[i] == 0
            expected = apply_label_consistent_trigger(torch.from_numpy(harder[i]).permute(2, 0, 1) / 255.0)
            assert torch.allclose(img, expected)
        else:
            assert torch.equal(img, ds.images[i])


def test_label_consistent_test_mode_triggers_non_target_images():
    ds = TinyImages(num=40)
    lc = LabelConsistentBackdoorDataset(ds, target_class=0, mode="test_poison", trigger="all-corners")
    assert len(lc) == 36
    img, original, target = lc[0]
    assert original != 0 and target == 0
    assert torch.allclose(img, apply_label_consistent_trigger(ds.images[1], "all-corners"))


def test_label_consistent_needs_aligned_harder_images():
    with pytest.raises(ValueError, match="harder_images"):
        LabelConsistentBackdoorDataset(TinyImages(num=10), target_class=0, harder_images=np.zeros((5, 32, 32, 3)))


# ---- ASR triggers match the training-side triggers ----

def _normalized_dataset(ds):
    return [(_normalize(img), label) for img, label in ds]


def test_new_trigger_types_are_not_offered_to_the_anp_cli():
    assert "wanet" in ATTACK_TRIGGER_TYPES and "wanet" not in TRIGGER_TYPES


@pytest.mark.parametrize("kind", ["blend", "wanet", "label-consistent"])
def test_asr_triggers_equal_normalized_training_triggers(kind):
    ds = TinyImages(num=20)
    pattern = make_blend_pattern(seed=1)
    noise = make_wanet_noise_grid(seed=0)
    grid = make_wanet_grid(noise, s=0.5)
    triggered = build_triggered_dataset(
        _normalized_dataset(ds), config=FakeConfig(), attack_target=0, trigger_type=kind,
        blend_alpha=0.2, blend_pattern=pattern, wanet_grid=grid,
        lc_trigger="all-corners", lc_amplitude=None, samples=5, seed=0,
    )
    train_side = {
        "blend": BlendBackdoorDataset(ds, 1.0, 0, pattern, alpha=0.2, mode="test_poison"),
        "wanet": WaNetBackdoorDataset(ds, 0, noise, s=0.5, mode="test_poison"),
        "label-consistent": LabelConsistentBackdoorDataset(ds, 0, mode="test_poison", trigger="all-corners"),
    }[kind]
    for position in range(len(triggered)):
        img, label = triggered[position]
        source = triggered.indices[position]
        pixel_img, original, _ = train_side[train_side._non_target_indices.index(source)]
        assert label == original
        assert torch.allclose(img, _normalize(pixel_img), atol=1e-5)


def test_wanet_trigger_requires_its_grid():
    with pytest.raises(ValueError, match="wanet_grid"):
        build_triggered_dataset(_normalized_dataset(TinyImages(num=5)), config=FakeConfig(), attack_target=0,
                                trigger_type="wanet")


# ---- benchmark.json ----

def test_benchmark_records_and_replaces_entries(tmp_path):
    path = tmp_path / "benchmark.json"
    assert load_benchmark(path) == []
    record_model({"name": "wanet", "path": "models/a.pth", "asr": 91.0}, path)
    record_model({"name": "blend", "path": "models/b.pth", "asr": 95.0}, path)
    record_model({"name": "wanet", "path": "models/c.pth", "asr": 97.0}, path)
    assert [m["name"] for m in load_benchmark(path)] == ["blend", "wanet"]
    assert get_model("wanet", path)["path"] == "models/c.pth"
    with pytest.raises(KeyError):
        get_model("missing", path)
