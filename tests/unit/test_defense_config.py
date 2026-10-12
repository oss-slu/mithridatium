"""
Unit tests for defense-specific configuration helpers.
"""

from __future__ import annotations

import pytest

from mithridatium.defense_config import (
    apply_freeeagle_cli_options,
    apply_mmbd_cli_options,
    parse_mmbd_classes,
)
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


def test_parse_mmbd_classes_valid_comma_list():
    parsed = parse_mmbd_classes("0,3,7", 10)
    assert parsed == [0, 3, 7]

    # Tolerates whitespace around indices
    parsed_whitespace = parse_mmbd_classes(" 0 , 3 , 7 ", 10)
    assert parsed_whitespace == [0, 3, 7]


def test_parse_mmbd_classes_all():
    parsed = parse_mmbd_classes("all", 10)
    assert parsed == list(range(10))

    parsed_caps = parse_mmbd_classes("ALL", 5)
    assert parsed_caps == list(range(5))


def test_parse_mmbd_classes_none():
    assert parse_mmbd_classes(None, 10) is None


def test_parse_mmbd_classes_list_input():
    parsed = parse_mmbd_classes([0, 3, 7], 10)
    assert parsed == [0, 3, 7]


def test_parse_mmbd_classes_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        parse_mmbd_classes("0,10", 10)

    with pytest.raises(ValueError, match="out of range"):
        parse_mmbd_classes("-1,2", 10)

    with pytest.raises(ValueError, match="out of range"):
        parse_mmbd_classes([0, 10], 10)


def test_parse_mmbd_classes_rejects_non_integer():
    with pytest.raises(ValueError, match="must be an integer or 'all'"):
        parse_mmbd_classes("0,foo,2", 10)

    with pytest.raises(ValueError, match="must be an integer or 'all'"):
        parse_mmbd_classes("1.5", 10)

    with pytest.raises(ValueError, match="must be an integer"):
        parse_mmbd_classes([True, 2], 10)


def test_parse_mmbd_classes_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate class index"):
        parse_mmbd_classes("0,1,1", 10)

    with pytest.raises(ValueError, match="Duplicate class index"):
        parse_mmbd_classes("3,3", 10)

    with pytest.raises(ValueError, match="Duplicate class index"):
        parse_mmbd_classes([2, 2], 10)


def test_parse_mmbd_classes_rejects_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_mmbd_classes("", 10)

    with pytest.raises(ValueError, match="cannot be empty"):
        parse_mmbd_classes("   ", 10)

    with pytest.raises(ValueError, match="empty class index"):
        parse_mmbd_classes("0,,2", 10)

    with pytest.raises(ValueError, match="empty class index"):
        parse_mmbd_classes(",", 10)


def test_apply_mmbd_cli_options_sets_attributes():
    config = get_preprocess_config("cifar10")

    apply_mmbd_cli_options(config, classes="0,3,7", num_classes=10)
    assert config.mmbd_classes == [0, 3, 7]

    apply_mmbd_cli_options(config, classes=[1, 4])
    assert config.mmbd_classes == [1, 4]

    apply_mmbd_cli_options(config, classes=None)
    assert config.mmbd_classes is None