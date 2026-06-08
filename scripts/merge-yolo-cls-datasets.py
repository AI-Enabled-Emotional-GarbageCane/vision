#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


VALID_LABELS = {"accept", "reject"}
VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO classification datasets into one dataset.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source dataset in name:path form. Path must contain train/val/test/{accept,reject}.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="hardlink",
        help="Use hardlink to save disk space when source and destination are on same filesystem.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[FAIL] {message}")


def parse_source(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise SystemExit(f"[FAIL] source must be name:path: {value}")
    name, path = value.split(":", 1)
    name = name.strip()
    require(name, f"source name is empty: {value}")
    return name, Path(path).resolve()


def clean_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in VALID_SPLITS:
        for label in sorted(VALID_LABELS):
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)


def copy_or_link(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def merge_source(
    name: str,
    source_dir: Path,
    output_dir: Path,
    copy_mode: str,
) -> list[dict[str, str]]:
    require(source_dir.exists(), f"missing source dataset: {source_dir}")
    rows: list[dict[str, str]] = []
    for split in VALID_SPLITS:
        for label in sorted(VALID_LABELS):
            class_dir = source_dir / split / label
            require(class_dir.exists(), f"missing class directory: {class_dir}")
            for index, image_path in enumerate(sorted(path for path in class_dir.iterdir() if path.is_file())):
                destination_name = f"{name}_{split}_{label}_{index:06d}_{image_path.name}"
                destination = output_dir / split / label / destination_name
                copy_or_link(image_path, destination, copy_mode)
                rows.append(
                    {
                        "source_dataset": name,
                        "source_path": str(image_path.relative_to(source_dir)),
                        "split": split,
                        "eval_label": label,
                        "local_path": str(destination.relative_to(output_dir)),
                    }
                )
    return rows


def write_manifest(rows: list[dict[str, str]], output_dir: Path) -> None:
    manifest_path = output_dir / "weak_dataset_manifest.csv"
    fieldnames = sorted({field for row in rows for field in row})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    rows: list[dict[str, str]],
    sources: list[tuple[str, Path]],
    output_dir: Path,
    copy_mode: str,
) -> None:
    counts = Counter((row["split"], row["eval_label"]) for row in rows)
    source_counts = Counter((row["source_dataset"], row["split"], row["eval_label"]) for row in rows)
    summary = {
        "output_dir": str(output_dir),
        "copy_mode": copy_mode,
        "sources": {name: str(path) for name, path in sources},
        "counts": {
            f"{split}/{label}": counts[(split, label)]
            for split in VALID_SPLITS
            for label in sorted(VALID_LABELS)
        },
        "source_counts": {
            f"{source}/{split}/{label}": count
            for (source, split, label), count in sorted(source_counts.items())
        },
        "total": len(rows),
        "note": "Merged weak-label YOLO classification dataset. Review labels before final claims.",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    sources = [parse_source(value) for value in args.source]
    clean_output(args.output_dir)
    rows: list[dict[str, str]] = []
    for name, source_dir in sources:
        rows.extend(merge_source(name, source_dir, args.output_dir, args.copy_mode))
    write_manifest(rows, args.output_dir)
    write_summary(rows, sources, args.output_dir, args.copy_mode)


if __name__ == "__main__":
    main()
