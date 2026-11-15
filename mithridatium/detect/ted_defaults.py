"""Shared default settings for TED and TED++ detection."""

TED_DEFAULTS = {
    "num_samples": 256,
    "reference_size": 1000,
    "contamination": 0.01,
}

TEDPP_DEFAULTS = {
    **TED_DEFAULTS,
    "samples_per_class": 5,
    "beta": 0.5,
}
