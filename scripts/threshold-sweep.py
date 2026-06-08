#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize accept-gate behavior across confidence thresholds.",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--uncertain-threshold", type=float, default=0.50)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        help="Accept threshold to evaluate. Defaults to 0.50..0.95 plus 0.76.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def default_thresholds() -> list[float]:
    values = [round(value / 100, 2) for value in range(50, 100, 5)]
    values.append(0.76)
    return sorted(set(values))


def gate_action(row: dict[str, str], accept_threshold: float, uncertain_threshold: float) -> str:
    pred_class = row["pred_class"]
    confidence = float(row["confidence"])
    if confidence < uncertain_threshold:
        return "uncertain"
    if pred_class == "accept" and confidence >= accept_threshold:
        return "accept"
    return "reject"


def summarize_threshold(
    rows: list[dict[str, str]],
    accept_threshold: float,
    uncertain_threshold: float,
) -> dict[str, Any]:
    accept_total = 0
    accept_accepted = 0
    reject_total = 0
    reject_false_accept = 0
    total = 0
    matches = 0

    for row in rows:
        eval_label = row.get("eval_label", "").strip()
        if eval_label not in {"accept", "reject"}:
            continue
        action = gate_action(row, accept_threshold, uncertain_threshold)
        total += 1
        if action == eval_label:
            matches += 1
        if eval_label == "accept":
            accept_total += 1
            if action == "accept":
                accept_accepted += 1
        else:
            reject_total += 1
            if action == "accept":
                reject_false_accept += 1

    return {
        "accept_threshold": accept_threshold,
        "uncertain_threshold": uncertain_threshold,
        "count": total,
        "gate_accuracy": matches / total if total else None,
        "accept_count": accept_total,
        "accepted_accept_count": accept_accepted,
        "gate_accept_recall": accept_accepted / accept_total if accept_total else None,
        "reject_count": reject_total,
        "false_accept_count_on_reject": reject_false_accept,
        "false_accept_rate_on_reject": (
            reject_false_accept / reject_total if reject_total else None
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "accept_threshold",
        "uncertain_threshold",
        "count",
        "gate_accuracy",
        "accept_count",
        "accepted_accept_count",
        "gate_accept_recall",
        "reject_count",
        "false_accept_count_on_reject",
        "false_accept_rate_on_reject",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    thresholds = args.threshold if args.threshold else default_thresholds()
    prediction_rows = read_rows(args.predictions)
    sweep_rows = [
        summarize_threshold(prediction_rows, threshold, args.uncertain_threshold)
        for threshold in sorted(set(thresholds))
    ]
    write_csv(args.output, sweep_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "predictions": str(args.predictions),
        "output": str(args.output),
        "threshold_count": len(sweep_rows),
        "thresholds": [row["accept_threshold"] for row in sweep_rows],
        "rows": sweep_rows,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
