"""
Unit tests for defense-specific configuration helpers.
"""

from __future__ import annotations

import pytest

from mithridatium.defense_config import apply_freeeagle_cli_options
from mithridatium.utils import get_preprocess_config


pytestmark = pytest.mark.unit


def test_apply_freeeagle_cli_options_sets_expected_attributes():
    config = get_preprocess_config("cifar10")

    apply_freeeagle_cli_options(
        config,
        num_classes=12,
        num_dummy=2,
        num_important_neurons=7,
        metric="softmax_score",
        use_transpose_correction=True,
        bound_on=False,
        optimize_steps=20,
        learning_rate=0.02,
        weight_decay=0.004,
        anomaly_threshold=2.7,
        inspect_layer_position=3,
    )

    assert config.freeeagle_num_classes == 12
    assert config.freeeagle_num_dummy == 2
    assert config.freeeagle_num_important_neurons == 7
    assert config.freeeagle_metric == "softmax_score"
    assert config.freeeagle_use_transpose_correction is True
    assert config.freeeagle_bound_on is False
    assert config.freeeagle_optimize_steps == 20
    assert config.freeeagle_learning_rate == 0.02
    assert config.freeeagle_weight_decay == 0.004
    assert config.freeeagle_anomaly_threshold == 2.7
    assert config.freeeagle_inspect_layer_position == 3


def test_apply_freeeagle_cli_options_does_not_set_num_classes_when_zero():
    config = get_preprocess_config("cifar10")

    apply_freeeagle_cli_options(config, num_classes=0)

    assert not hasattr(config, "freeeagle_num_classes")