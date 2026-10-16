"""
Unit tests for report.py helpers.
"""

from __future__ import annotations

import pytest
import numpy as np

from mithridatium.report import (
    to_json_safe,
    render_summary,
    validate_report_data,
    build_report,
)

pytestmark = pytest.mark.unit

def test_to_json_safe_handles_complex_nested_types():
    obj = {
        "arr": np.array([1, 2, 3]),
        "np_float": np.float32(1.5),
        "np_int": np.int64(42),
        "nested_dict": {
            "arr2": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "np_float2": np.float64(3.14),
        },
        "nested_list": [
            np.int32(10),
            [np.array([4, 5])],
            {"inner_tup": (np.float32(2.5), 6)}
        ],
        "tup": (np.int8(1), np.float16(2.0))
    }
    
    safe_obj = to_json_safe(obj)
    
    # Assertions
    assert isinstance(safe_obj["arr"], list)
    assert safe_obj["arr"] == [1, 2, 3]
    assert isinstance(safe_obj["np_float"], float)
    assert safe_obj["np_float"] == 1.5
    assert isinstance(safe_obj["np_int"], int)
    assert safe_obj["np_int"] == 42
    
    assert isinstance(safe_obj["nested_dict"]["arr2"], list)
    assert safe_obj["nested_dict"]["arr2"] == [[1.0, 2.0], [3.0, 4.0]]
    assert isinstance(safe_obj["nested_dict"]["np_float2"], float)
    assert safe_obj["nested_dict"]["np_float2"] == 3.14
    
    assert isinstance(safe_obj["nested_list"], list)
    assert isinstance(safe_obj["nested_list"][0], int)
    assert safe_obj["nested_list"][0] == 10
    assert isinstance(safe_obj["nested_list"][1], list)
    assert isinstance(safe_obj["nested_list"][1][0], list)
    assert safe_obj["nested_list"][1][0] == [4, 5]
    
    assert isinstance(safe_obj["nested_list"][2], dict)
    assert isinstance(safe_obj["nested_list"][2]["inner_tup"], list)
    assert safe_obj["nested_list"][2]["inner_tup"] == [2.5, 6]
    
    assert isinstance(safe_obj["tup"], list)
    assert safe_obj["tup"] == [1, 2.0]

def test_render_summary_mmbd():
    report = {
        "mithridatium_version": "0.1.1",
        "defense": "mmbd",
        "dataset": "cifar10",
        "model_path": "models/resnet.pth",
        "results": {
            "verdict": "likely backdoored",
            "p_value": 0.04,
            "suspected_target": 3,
            "per_class_scores": [1.0, 2.0, 3.0],
            "top_eigenvalue": 5.5
        }
    }
    
    summary = render_summary(report)
    assert "Mithridatium 0.1.1" in summary
    assert "defense=mmbd" in summary
    assert "dataset=cifar10" in summary
    assert "models/resnet.pth" in summary
    assert "verdict:           likely backdoored" in summary
    assert "p_value:           0.040000" in summary
    assert "suspected_target:  3" in summary
    assert "per_class_scores:  3 classes" in summary
    assert "top_eigenvalue:    5.5" in summary

def test_render_summary_mmbd_missing_optional_fields():
    report = {
        "mithridatium_version": "0.1.1",
        "defense": "mmbd",
        "dataset": "cifar10",
        "model_path": "models/resnet.pth",
        "results": {}
    }
    summary = render_summary(report)
    assert "Mithridatium 0.1.1" in summary
    assert "verdict" not in summary

def test_render_summary_strip():
    report = {
        "mithridatium_version": "0.1.1",
        "defense": "strip",
        "dataset": "cifar10",
        "model_path": "models/resnet.pth",
        "results": {
            "verdict": "likely clean",
            "thresholds": {"entropy_mean_threshold": 1.2},
            "parameters": {"num_bases": 5, "num_perturbations": 10},
            "statistics": {
                "entropy_mean": 2.5,
                "entropy_std": 0.1,
                "entropy_min": 2.0,
                "entropy_max": 3.0
            },
            "dataset": "cifar10",
            "entropies": [2.5, 2.6]
        }
    }
    
    summary = render_summary(report)
    assert "defense=strip" in summary
    assert "verdict:           likely clean" in summary
    assert "entropy_thr:       1.2" in summary
    assert "num_bases:         5" in summary
    assert "num_perturbations: 10" in summary
    assert "entropy_mean:      2.5" in summary
    assert "entropy_std:       0.1" in summary
    assert "entropy_min:       2.0" in summary
    assert "entropy_max:       3.0" in summary
    assert "dataset:           cifar10" in summary
    assert "entropies:" in summary
    assert "#0: 2.5" in summary

def test_render_summary_strip_missing_optional_fields():
    report = {
        "mithridatium_version": "0.1.1",
        "defense": "strip",
        "dataset": "cifar10",
        "model_path": "models/resnet.pth",
        "results": {}
    }
    summary = render_summary(report)
    assert "defense=strip" in summary
    assert "verdict" not in summary

def test_render_summary_freeeagle():
    report = {
        "mithridatium_version": "0.1.1",
        "defense": "freeeagle",
        "dataset": "cifar10",
        "model_path": "models/resnet.pth",
        "results": {
            "verdict": "likely clean",
            "anomaly_metric": 3.14,
            "thresholds": {"anomaly_metric_threshold": 5.0},
            "tendency_per_target": [1, 2, 3, 4],
            "parameters": {
                "inspect_layer_position": -1,
                "optimize_steps": 100
            }
        }
    }
    summary = render_summary(report)
    assert "defense=freeeagle" in summary
    assert "verdict:           likely clean" in summary
    assert "anomaly_metric:    3.140000" in summary
    assert "anomaly_thr:       5.0" in summary
    assert "targets_scored:    4" in summary
    assert "inspect_layer:     -1" in summary
    assert "optimize_steps:    100" in summary

def test_render_summary_freeeagle_missing_optional_fields():
    report = {
        "mithridatium_version": "0.1.1",
        "defense": "freeeagle",
        "dataset": "cifar10",
        "model_path": "models/resnet.pth",
        "results": {}
    }
    summary = render_summary(report)
    assert "defense=freeeagle" in summary
    assert "verdict" not in summary

def test_render_summary_fallback_legacy():
    report = {
        "mithridatium_version": "0.1.1",
        "defense": "legacy_defense",
        "dataset": "cifar10",
        "model_path": "models/resnet.pth",
        "results": {
            "suspected_backdoor": True,
            "num_flagged": 2,
            "top_eigenvalue": 10.5
        }
    }
    summary = render_summary(report)
    assert "defense=legacy_defense" in summary
    assert "suspected_backdoor:True" in summary
    assert "num_flagged:       2" in summary
    assert "top_eigenvalue:    10.5" in summary

def test_validate_report_data_valid():
    valid_report = {
        "mithridatium_version": "0.1.1",
        "timestamp_utc": "2023-10-10T00:00:00Z",
        "model_path": "models/resnet.pth",
        "defense": "mmbd",
        "dataset": "cifar10",
        "results": {}
    }
    # Should not raise
    validate_report_data(valid_report)

def test_validate_report_data_invalid_missing_field():
    import jsonschema
    invalid_report = {
        "mithridatium_version": "0.1.1",
        "timestamp_utc": "2023-10-10T00:00:00Z",
        # missing model_path
        "defense": "mmbd",
        "dataset": "cifar10",
        "results": {}
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_report_data(invalid_report)

def test_build_report_and_validate():
    report = build_report(
        model_path="my_model.pth",
        defense="mmbd",
        dataset="cifar10",
        version="1.0.0",
        results={"verdict": "likely clean"}
    )
    
    # Verify top-level fields
    assert report["mithridatium_version"] == "1.0.0"
    assert report["model_path"] == "my_model.pth"
    assert report["defense"] == "mmbd"
    assert report["dataset"] == "cifar10"
    assert report["results"] == {"verdict": "likely clean"}
    assert "timestamp_utc" in report
    
    # Should not raise validation error
    validate_report_data(report)
