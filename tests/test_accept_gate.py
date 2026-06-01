from __future__ import annotations

from vision_contract import build_recognition_result
from vision_policy import DEFAULT_ACCEPT_THRESHOLD, should_allow_accept


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

    confident_reject = build_recognition_result(
        predicted_class="reject",
        confidence=0.99,
        snapshot_path="fixtures/confident-reject.jpg",
        ts="2026-06-01T12:00:02",
    )
    require(not should_allow_accept(confident_reject), "reject payloads must never be allowed")

    require(
        low_confidence_accept["class"] == "accept",
        "deployment gate must not rewrite the payload class",
    )

    print("[OK] accept gate preserves payload class and blocks unsafe accepts")


if __name__ == "__main__":
    main()
