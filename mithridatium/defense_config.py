"""
Helpers for applying defense-specific CLI options to PreprocessConfig objects.

The CLI should stay focused on command routing. Defense-specific configuration
belongs here so detect() does not become a giant collection of unrelated flags.
"""

from __future__ import annotations

from typing import Any


def apply_freeeagle_cli_options(
    config: Any,
    *,
    num_classes: int = 0,
    num_dummy: int = 1,
    num_important_neurons: int = 5,
    metric: str = "softmax_score",
    use_transpose_correction: bool = False,
    bound_on: bool = True,
    optimize_steps: int = 300,
    learning_rate: float = 1e-2,
    weight_decay: float = 5e-3,
    anomaly_threshold: float = 2.0,
    inspect_layer_position: int = 2,
) -> Any:
    """
    Apply FreeEagle CLI overrides to a config object.

    FreeEagle currently reads options from attributes on the preprocessing
    config object. This helper centralizes that mapping.
    """
    if num_classes > 0:
        setattr(config, "freeeagle_num_classes", num_classes)

    setattr(config, "freeeagle_num_dummy", num_dummy)
    setattr(config, "freeeagle_num_important_neurons", num_important_neurons)
    setattr(config, "freeeagle_metric", metric)
    setattr(config, "freeeagle_use_transpose_correction", use_transpose_correction)
    setattr(config, "freeeagle_bound_on", bound_on)
    setattr(config, "freeeagle_optimize_steps", optimize_steps)
    setattr(config, "freeeagle_learning_rate", learning_rate)
    setattr(config, "freeeagle_weight_decay", weight_decay)
    setattr(config, "freeeagle_anomaly_threshold", anomaly_threshold)
    setattr(config, "freeeagle_inspect_layer_position", inspect_layer_position)

    return config