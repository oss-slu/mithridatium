"""ANP reports use the same repair format; ASR is optional."""

import pytest
from mithridatium import report as rpt


def make_report():
    return rpt.build_repair_report(
        "models/resnet18_poison.pth", "anp", "cifar10", "0.1.2",
        {
            "method": "anp",
            "verdict": "repair_applied",
            "status": "completed",
            "parameters": {"eps": 0.4, "alpha": 0.2, "steps": 10},
            "total_channels_pruned": 2,
            "layer_breakdown": [
                {"layer": "layer1.0.bn1", "channels_pruned": 2, "indices": [1, 3]}
            ],
            "clean_accuracy_before": 0.9,
            "clean_accuracy_after": 0.88,
        },
    )


def test_anp_without_asr():
    report = make_report()
    assert "asr_before" not in report["results"]
    rpt.validate_report_data(report)


def test_anp_with_asr():
    report = make_report()
    report["results"].update(
        asr_before=0.95, asr_after=0.10, asr_evaluation_samples=100
    )
    rpt.validate_report_data(report)


def test_anp_requires_common_repair_fields():
    for field in ("verdict", "parameters"):
        report = make_report()
        del report["results"][field]
        with pytest.raises(Exception):
            rpt.validate_report_data(report)


def test_anp_disallows_partial_asr():
    report = make_report()
    report["results"]["asr_before"] = 0.95
    with pytest.raises(Exception):
        rpt.validate_report_data(report)


def test_anp_disallows_invalid_asr():
    report = make_report()
    report["results"].update(
        asr_before=1.5, asr_after=0.10, asr_evaluation_samples=100
    )
    with pytest.raises(Exception):
        rpt.validate_report_data(report)
