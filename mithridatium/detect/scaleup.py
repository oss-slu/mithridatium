"""SCALE-UP input-level detection (stub until scoring is implemented)."""

from __future__ import annotations

from typing import Any, Sequence


def scaleup_stub_results(
    *,
    method: str,
    num_samples: int,
    threshold: float,
    scales: Sequence[int | float],
) -> dict[str, Any]:
    """
    Placeholder report payload until SCALE-UP scoring lands.
    Field names follow docs/research/detection-additions.md.
    """
    return {
        "mode": "input-level",
        "method": method,
        "status": "stub_complete",
        "verdict": "unimplemented",
        "num_inputs": num_samples,
        "num_flagged": 0,
        "threshold": threshold,
        "parameters": {"scales": list(scales)},
        "per_sample": [],
    }
