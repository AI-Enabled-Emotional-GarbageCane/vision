from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_ACCEPT_THRESHOLD = 0.76
DEFAULT_UNCERTAIN_THRESHOLD = 0.50
DEPLOYMENT_ACTIONS = {"accept", "reject", "uncertain"}


def _confidence(recognition_result: Mapping[str, Any]) -> float:
    return float(recognition_result.get("confidence", -1))


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
        and _confidence(recognition_result) >= accept_threshold
    )


def deployment_action(
    recognition_result: Mapping[str, Any],
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD,
) -> str:
    """Return the derived deployment action without rewriting the payload class."""
    if not 0 <= accept_threshold <= 1:
        raise ValueError(f"accept_threshold out of range: {accept_threshold}")
    if not 0 <= uncertain_threshold <= 1:
        raise ValueError(f"uncertain_threshold out of range: {uncertain_threshold}")
    if uncertain_threshold > accept_threshold:
        raise ValueError(
            "uncertain_threshold must be less than or equal to accept_threshold"
        )

    if recognition_result.get("event") != "recognition_result":
        return "uncertain"

    confidence = _confidence(recognition_result)
    if confidence < uncertain_threshold:
        return "uncertain"
    if recognition_result.get("class") == "accept" and confidence >= accept_threshold:
        return "accept"
    return "reject"
