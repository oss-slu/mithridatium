"""Shared default settings for TED and TED++ detection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TEDDefaults:
    num_samples: int = 256
    reference_size: int = 500
    contamination: float = 0.01


@dataclass(frozen=True)
class TEDPPDefaults:
    samples_per_class: int = 5
    beta: float = 0.5


TED_DEFAULTS = TEDDefaults()
TEDPP_DEFAULTS = TEDPPDefaults()
