#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a weak-label manifest by accept rows and selected reject categories.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category-field", default="dominant_taco_category")
    parser.add_argument("--accept-all", action="store_true")
    parser.add_argument("--reject-category", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_categories = set(args.reject_category)
    with args.manifest.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            eval_label = row.get("eval_label", "")
            category = row.get(args.category_field, "")
            if args.accept_all and eval_label == "accept":
                rows.append(row)
            elif eval_label == "reject" and category in reject_categories:
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row.get("eval_label", "") for row in rows)
    category_counts = Counter(row.get(args.category_field, "") for row in rows)
    summary = {
        "source_manifest": str(args.manifest),
        "output": str(args.output),
        "accept_all": args.accept_all,
        "reject_categories": sorted(reject_categories),
        "counts": dict(counts),
        "category_counts": dict(sorted(category_counts.items())),
        "total": len(rows),
    }
    (args.output.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
