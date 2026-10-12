"""
Helpers for applying defense-specific CLI options to PreprocessConfig objects.

The CLI should stay focused on command routing. Defense-specific configuration
belongs here so audit() does not become a giant collection of unrelated flags.
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


def parse_mmbd_classes(
    classes_input: str | list[int] | tuple[int, ...] | None,
    num_classes: int,
) -> list[int] | None:
    """
    Parse and validate the MMBD classes to probe.

    Parameters
    ----------
    classes_input : str | list[int] | tuple[int, ...] | None
        Comma-separated string of class indices (e.g. '0,3,7'), 'all', a collection of
        integers, or None (for default probing).
    num_classes : int
        Total number of classes in the target model / dataset.

    Returns
    -------
    list[int] | None
        Validated list of unique class indices, or None if classes_input is None.

    Raises
    ------
    ValueError
        If classes_input contains non-integers, out-of-range indices (< 0 or >= num_classes),
        duplicates, or is empty.
    """
    if classes_input is None:
        return None

    if isinstance(classes_input, str):
        cleaned = classes_input.strip()
        if not cleaned:
            raise ValueError("MMBD classes input cannot be empty.")

        if cleaned.lower() == "all":
            if num_classes <= 0:
                raise ValueError(f"Invalid num_classes {num_classes} for 'all'.")
            return list(range(num_classes))

        parts = [p.strip() for p in cleaned.split(",")]
        parsed_classes: list[int] = []
        seen: set[int] = set()

        for p in parts:
            if not p:
                raise ValueError("MMBD classes contains an empty class index in comma-separated list.")
            try:
                val = int(p)
            except ValueError:
                raise ValueError(f"Invalid class index '{p}': must be an integer or 'all'.")

            if val < 0 or val >= num_classes:
                raise ValueError(
                    f"Class index {val} is out of range for model with {num_classes} classes (must be between 0 and {num_classes - 1})."
                )

            if val in seen:
                raise ValueError(f"Duplicate class index '{val}' specified in MMBD classes.")

            seen.add(val)
            parsed_classes.append(val)

        return parsed_classes

    if isinstance(classes_input, (list, tuple)):
        parsed_classes = []
        seen = set()
        for c in classes_input:
            if isinstance(c, bool) or not isinstance(c, int):
                raise ValueError(f"Invalid class index '{c}': must be an integer.")
            if c < 0 or c >= num_classes:
                raise ValueError(
                    f"Class index {c} is out of range for model with {num_classes} classes (must be between 0 and {num_classes - 1})."
                )
            if c in seen:
                raise ValueError(f"Duplicate class index '{c}' specified in MMBD classes.")
            seen.add(c)
            parsed_classes.append(c)
        return parsed_classes

    raise ValueError(f"Unsupported type for MMBD classes: {type(classes_input).__name__}")


def apply_mmbd_cli_options(
    config: Any,
    *,
    classes: list[int] | str | None = None,
    num_classes: int | None = None,
) -> Any:
    """
    Apply MMBD CLI overrides to a config object.

    Sets the 'mmbd_classes' attribute on the config object if classes is provided.
    """
    if classes is not None:
        if isinstance(classes, str):
            if num_classes is None:
                if hasattr(config, "get_num_classes"):
                    num_classes = config.get_num_classes()
                else:
                    raise ValueError("num_classes must be provided when passing a string to apply_mmbd_cli_options.")
            parsed = parse_mmbd_classes(classes, num_classes)
        else:
            parsed = list(classes)
        setattr(config, "mmbd_classes", parsed)
    else:
        setattr(config, "mmbd_classes", None)

    return config