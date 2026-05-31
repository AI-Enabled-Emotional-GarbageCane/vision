from __future__ import annotations

from datetime import datetime
from typing import Any


RECOGNITION_RESULT_FIELDS = {
    "event",
    "class",
    "confidence",
    "num_objects",
    "snapshot_path",
    "ts",
}
CLASS_VALUES = {"accept", "reject"}


def build_recognition_result(
    predicted_class: str,
    confidence: float,
    snapshot_path: str,
    ts: str | None = None,
) -> dict[str, Any]:
    if predicted_class not in CLASS_VALUES:
        raise ValueError(f"invalid class: {predicted_class}")
    if not 0 <= confidence <= 1:
        raise ValueError(f"confidence out of range: {confidence}")
    if not snapshot_path:
        raise ValueError("snapshot_path is required")

    return {
        "event": "recognition_result",
        "class": predicted_class,
        "confidence": confidence,
        "num_objects": 1,
        "snapshot_path": snapshot_path,
        "ts": ts or datetime.now().isoformat(timespec="seconds"),
    }


def run_stub_inference(snapshot_path: str = "fixtures/stub-l515-frame.jpg") -> dict[str, Any]:
    return build_recognition_result(
        predicted_class="reject",
        confidence=0.42,
        snapshot_path=snapshot_path,
        ts="2026-05-31T20:00:00",
    )
