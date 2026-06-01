#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "tests/fixtures/public-baseline-metrics.json"


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_metrics() -> dict[str, Any]:
    require(METRICS_PATH.exists(), f"missing metrics fixture: {METRICS_PATH}")
    with METRICS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def confusion_counts(section: dict[str, Any]) -> tuple[int, int, int, int]:
    confusion = section.get("confusion", {})
    accept = confusion.get("accept", {})
    reject = confusion.get("reject", {})
    return (
        int(accept.get("accept", 0)),
        int(accept.get("reject", 0)),
        int(reject.get("accept", 0)),
        int(reject.get("reject", 0)),
    )


def validate_rate(
    section_name: str,
    section: dict[str, Any],
    expected_total: int,
) -> None:
    accept_accept, accept_reject, reject_accept, reject_reject = confusion_counts(section)
    total = accept_accept + accept_reject + reject_accept + reject_reject
    reject_total = reject_accept + reject_reject

    require(total == expected_total, f"{section_name} confusion total must be {expected_total}")
    require(reject_total > 0, f"{section_name} must include reject examples")

    expected_false_accept = reject_accept / reject_total
    expected_reject_recall = reject_reject / reject_total

    require(
        abs(float(section["false_accept_rate_on_reject"]) - expected_false_accept) < 1e-12,
        f"{section_name} false_accept_rate_on_reject does not match confusion matrix",
    )
    require(
        abs(float(section["reject_recall"]) - expected_reject_recall) < 1e-12,
        f"{section_name} reject_recall does not match confusion matrix",
    )


def main() -> None:
    metrics = load_metrics()
    targets = metrics["acceptance_targets"]
    expected_total = int(metrics["test_count"])
    argmax = metrics["argmax"]
    accept_gate = metrics["accept_gate"]

    validate_rate("argmax", argmax, expected_total)
    validate_rate("accept_gate", accept_gate, expected_total)

    require(
        float(argmax["top1"]) >= float(targets["top1_min"]),
        "argmax top1 must meet the public baseline target",
    )
    require(
        float(accept_gate["false_accept_rate_on_reject"])
        <= float(targets["false_accept_rate_on_reject_max"]),
        "accept gate must meet the false accept target",
    )
    require(
        float(accept_gate["reject_recall"]) >= float(targets["reject_recall_min"]),
        "accept gate must meet the reject recall target",
    )

    print("[OK] public baseline top-1 and accept gate safety metrics are consistent")


if __name__ == "__main__":
    main()
