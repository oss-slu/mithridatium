"""Shared batch-verdict rule for input-level detectors."""


def batch_verdict(flagged_fraction: float, contamination: float) -> str:
    return (
        "likely backdoored"
        if flagged_fraction > contamination
        else "likely clean"
    )
