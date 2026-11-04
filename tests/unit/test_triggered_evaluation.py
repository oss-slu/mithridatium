"""Tests for shared ASR evaluation and triggered datasets."""

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from mithridatium.evaluation.asr import evaluate_targeted_asr
from mithridatium.evaluation.triggered import (
    TriggeredDataset, WhiteSquareTrigger, WhiteSquareTriggeredDataset,
)


class FakeCifarConfig:
    def get_num_classes(self):
        return 3

    def get_normalize(self):
        return True

    def get_mean(self):
        return (0.5, 0.4, 0.3)

    def get_std(self):
        return (0.2, 0.25, 0.5)


def dataset():
    # Already normalized CIFAR-like 3x6x6 tensors, matching utils.dataloader_for.
    images = torch.zeros(6, 3, 6, 6)
    labels = torch.tensor([0, 1, 2, 1, 2, 0])
    return TensorDataset(images, labels)


@pytest.mark.parametrize("position,y,x", [
    ("bottom-right", slice(4, 6), slice(4, 6)),
    ("bottom-left", slice(4, 6), slice(0, 2)),
    ("top-right", slice(0, 2), slice(4, 6)),
    ("top-left", slice(0, 2), slice(0, 2)),
])
def test_white_square_matches_pixel_space_trigger_and_keeps_labels(position, y, x):
    clean = dataset()
    triggered = WhiteSquareTriggeredDataset(
        clean, attack_target=0, config=FakeCifarConfig(),
        trigger_size=2, trigger_pos=position, samples=4,
    )
    assert len(triggered) == 4
    for j in range(len(triggered)):
        image, original_label = triggered[j]
        assert int(original_label) != 0
        assert original_label == clean[triggered.indices[j]][1]
        expected = torch.zeros_like(image)
        expected[:, y, x] = torch.tensor([2.5, 2.4, 1.4]).view(3, 1, 1)
        torch.testing.assert_close(image, expected)
        # Applying the patch must not mutate clean data.
        assert clean[triggered.indices[j]][0].count_nonzero() == 0


def test_asr_uses_only_non_target_samples():
    triggered = WhiteSquareTriggeredDataset(
        dataset(), attack_target=0, config=FakeCifarConfig(), samples=100
    )

    class AlwaysTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, images):
            result = torch.zeros(len(images), 3)
            result[:, 0] = 1
            return result

    asr, count = evaluate_targeted_asr(AlwaysTargetModel(), triggered, target_label=0)
    assert asr == 1.0
    assert count == 4  # target-label examples are never included


def test_dataset_reproducible_and_bounded():
    a = WhiteSquareTriggeredDataset(dataset(), attack_target=0, config=FakeCifarConfig(), samples=2, seed=21)
    b = WhiteSquareTriggeredDataset(dataset(), attack_target=0, config=FakeCifarConfig(), samples=2, seed=21)
    assert a.indices == b.indices
    assert len(a) == 2


@pytest.mark.parametrize("kwargs,match", [
    ({"attack_target": 3}, "target"),
    ({"attack_target": 0, "trigger_size": 0}, "size"),
    ({"attack_target": 0, "trigger_size": 7}, "size"),
    ({"attack_target": 0, "trigger_pos": "middle"}, "position"),
    ({"attack_target": 0, "samples": 0}, "samples"),
])
def test_bad_specs_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WhiteSquareTriggeredDataset(dataset(), config=FakeCifarConfig(), **kwargs)


def test_no_non_target_samples_rejected():
    only_target = TensorDataset(torch.zeros(2, 3, 6, 6), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="outside the attack target"):
        WhiteSquareTriggeredDataset(only_target, attack_target=0, config=FakeCifarConfig())


def test_custom_trigger_is_architecture_independent():
    clean = dataset()
    # Different attacks can provide their own callable without ANP dependencies.
    def add_pattern(image):
        image[:, 1, 1] = 9
        return image

    triggered = TriggeredDataset(
        clean, trigger=add_pattern, attack_target=0, samples=3, seed=7,
    )
    assert len(triggered) == 3
    for index in range(len(triggered)):
        image, label = triggered[index]
        assert int(label) != 0
        assert torch.all(image[:, 1, 1] == 9)
        assert torch.all(clean[triggered.indices[index]][0][:, 1, 1] == 0)


def test_white_square_handles_other_image_sizes_and_preprocessing():
    class UnnormalizedConfig(FakeCifarConfig):
        def get_normalize(self):
            return False

    # Shared trigger works on any CHW RGB size, not only 32x32 CIFAR-10.
    image = torch.zeros(3, 8, 12)
    result = WhiteSquareTrigger(config=UnnormalizedConfig(), trigger_size=3)(image)
    assert torch.all(result[:, -3:, -3:] == 1)
    assert image.count_nonzero() == 0


def test_anp_previous_asr_import_remains_available():
    from mithridatium.repair.anp import evaluate_targeted_asr as old_path
    assert old_path is evaluate_targeted_asr

@pytest.mark.parametrize("kind", ["white-square", "checkerboard", "cross", "blend"])
def test_factory_supports_multiple_trigger_types(kind):
    from mithridatium.evaluation.triggered import build_triggered_dataset
    triggered = build_triggered_dataset(dataset(), config=FakeCifarConfig(),
                                         attack_target=0, trigger_type=kind,
                                         trigger_size=3, trigger_pos="top-left",
                                         blend_alpha=0.5, samples=3, seed=4)
    assert len(triggered) == 3
    image, label = triggered[0]
    original = dataset()[triggered.indices[0]][0]
    assert int(label) != 0
    assert not torch.equal(image, original)
    if kind == "white-square":
        torch.testing.assert_close(image[:, 0, 0], torch.tensor([2.5, 2.4, 1.4]))
        assert torch.count_nonzero(image[:, 3:, 3:]) == 0
    elif kind == "checkerboard":
        torch.testing.assert_close(image[:, 0, 0], torch.tensor([2.5, 2.4, 1.4]))
        torch.testing.assert_close(image[:, 0, 1], torch.tensor([-2.5, -1.6, -0.6]))
    elif kind == "cross":
        assert torch.count_nonzero(image[:, 0, 0]) == 0
        torch.testing.assert_close(image[:, 1, 0], torch.tensor([2.5, 2.4, 1.4]))
    else:
        torch.testing.assert_close(image[:, 0, 0], torch.tensor([1.25, 1.2, 0.7]))


def test_factory_does_not_modify_source_and_is_deterministic():
    from mithridatium.evaluation.triggered import build_triggered_dataset
    clean = dataset()
    a = build_triggered_dataset(clean, config=FakeCifarConfig(), attack_target=0,
                                trigger_type="checkerboard", samples=3, seed=42)
    b = build_triggered_dataset(clean, config=FakeCifarConfig(), attack_target=0,
                                trigger_type="checkerboard", samples=3, seed=42)
    assert a.indices == b.indices
    assert torch.count_nonzero(clean.tensors[0]) == 0


@pytest.mark.parametrize("params", [
    {"trigger_type": "unknown"},
    {"trigger_type": "blend", "blend_alpha": 0},
    {"trigger_type": "blend", "blend_alpha": float("nan")},
    {"trigger_type": "checkerboard", "trigger_size": 0},
])
def test_factory_rejects_invalid_triggers(params):
    from mithridatium.evaluation.triggered import build_triggered_dataset
    with pytest.raises(ValueError):
        build_triggered_dataset(dataset(), config=FakeCifarConfig(), attack_target=0, **params)
