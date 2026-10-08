from __future__ import annotations

import pytest

from mithridatium.report import render_summary

pytestmark = pytest.mark.unit


def test_render_summary_aeva():
    report = {
        "mithridatium_version": "0.1.2",
        "model_path": "model.pt",
        "defense": "aeva",
        "dataset": "cifar10",
        "results": {
            "verdict": "likely backdoored",
            "suspected_target": 7,
            "suspicion_score": 5.25,
            "clean_accuracy": 0.91,
            "thresholds": {
                "anomaly_index_threshold": 4.0,
            },
            "parameters": {
                "sp": 0,
                "ep": 10,
            },
        },
    }

    summary = render_summary(report)

    assert "- verdict:           likely backdoored" in summary
    assert "- suspected_target:  7" in summary
    assert "- suspicion_score:   5.250000" in summary
    assert "- anomaly_thr:       4.0" in summary
    assert "- clean_accuracy:    0.910000" in summary
    assert "- class_range:       0-10" in summary


def test_render_summary_aeva_handles_missing_optional_fields():
    report = {
        "mithridatium_version": "0.1.2",
        "model_path": "model.pt",
        "defense": "aeva",
        "dataset": "cifar10",
        "results": {},
    }

    summary = render_summary(report)

    assert "defense=aeva" in summary
    assert "model.pt" in summary
