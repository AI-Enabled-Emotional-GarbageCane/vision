from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_ACCEPT_THRESHOLD = 0.76


def should_allow_accept(
    recognition_result: Mapping[str, Any],
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
) -> bool:
    """Return whether the deployment policy should treat this result as accepted."""
    if not 0 <= accept_threshold <= 1:
        raise ValueError(f"accept_threshold out of range: {accept_threshold}")

    return (
        recognition_result.get("event") == "recognition_result"
        and recognition_result.get("class") == "accept"
        and float(recognition_result.get("confidence", -1)) >= accept_threshold
    )
