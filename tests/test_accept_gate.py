from __future__ import annotations

from vision_contract import build_recognition_result
from vision_policy import DEFAULT_ACCEPT_THRESHOLD, deployment_action, should_allow_accept


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    high_confidence_accept = build_recognition_result(
        predicted_class="accept",
        confidence=0.91,
        snapshot_path="fixtures/high-confidence-accept.jpg",
        ts="2026-06-01T12:00:00",
    )
    require(should_allow_accept(high_confidence_accept), "high-confidence accept should be allowed")

    low_confidence_accept = build_recognition_result(
        predicted_class="accept",
        confidence=DEFAULT_ACCEPT_THRESHOLD - 0.01,
        snapshot_path="fixtures/low-confidence-accept.jpg",
        ts="2026-06-01T12:00:01",
    )
    require(
        not should_allow_accept(low_confidence_accept),
        "accept below the deployment threshold should not be allowed",
    )
    require(
        deployment_action(low_confidence_accept) == "reject",
        "accept below deployment threshold but above uncertain threshold should reject",
    )

    uncertain_accept = build_recognition_result(
        predicted_class="accept",
        confidence=0.42,
        snapshot_path="fixtures/uncertain-accept.jpg",
        ts="2026-06-01T12:00:01",
    )
    require(
        deployment_action(uncertain_accept) == "uncertain",
        "low-confidence best guesses should be uncertain, not hard reject",
    )

    confident_reject = build_recognition_result(
        predicted_class="reject",
        confidence=0.99,
        snapshot_path="fixtures/confident-reject.jpg",
        ts="2026-06-01T12:00:02",
    )
    require(not should_allow_accept(confident_reject), "reject payloads must never be allowed")
    require(
        deployment_action(confident_reject) == "reject",
        "high-confidence reject should map to reject action",
    )

    require(
        low_confidence_accept["class"] == "accept",
        "deployment gate must not rewrite the payload class",
    )

    require(
        deployment_action({"event": "camera_error", "confidence": 0.99}) == "uncertain",
        "non-recognition events should be treated as uncertain",
    )

    print("[OK] accept gate preserves payload class and derives explicit actions")


if __name__ == "__main__":
    main()
