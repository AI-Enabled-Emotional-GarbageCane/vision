#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VALID_LABELS = {"accept", "reject"}

REALWASTE_ACCEPT_CLASSES = {
    "Miscellaneous Trash",
    "Textile Trash",
}
REALWASTE_REJECT_CLASSES = {
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Vegetation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO classification dataset from class-named image folders.",
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mapping-preset",
        choices=("realwaste", "none"),
        default="realwaste",
        help="Use built-in source-class to accept/reject mappings.",
    )
    parser.add_argument("--accept-class", action="append", default=[])
    parser.add_argument("--reject-class", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--max-train-majority-ratio",
        type=float,
        default=2.0,
        help="Cap each train label to this multiple of the smallest train label; <=0 disables.",
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


def normalized_class_name(path: Path) -> str:
    return path.name.replace("_", " ").strip()


def class_mapping(args: argparse.Namespace) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if args.mapping_preset == "realwaste":
        mapping.update({name: "accept" for name in REALWASTE_ACCEPT_CLASSES})
        mapping.update({name: "reject" for name in REALWASTE_REJECT_CLASSES})

    for name in args.accept_class:
        mapping[name] = "accept"
    for name in args.reject_class:
        mapping[name] = "reject"
    return mapping


def find_image_rows(source_dir: Path, mapping: dict[str, str]) -> list[dict[str, str]]:
    require(source_dir.exists(), f"missing source directory: {source_dir}")
    rows: list[dict[str, str]] = []
    unmapped: Counter[str] = Counter()
    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        source_class = normalized_class_name(class_dir)
        eval_label = mapping.get(source_class)
        if eval_label not in VALID_LABELS:
            unmapped[source_class] += 1
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
                continue
            rows.append(
                {
                    "source_path": str(image_path.relative_to(source_dir)),
                    "source_class": source_class,
                    "eval_label": eval_label,
                    "eval_label_strength": "weak_source_class_mapping",
                    "label_hint": source_class,
                }
            )

    require(rows, f"no mapped images found in {source_dir}")
    labels = Counter(row["eval_label"] for row in rows)
    for label in sorted(VALID_LABELS):
        require(labels[label] > 0, f"no mapped images for label: {label}")
    return rows


def split_rows(
    rows: list[dict[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> list[dict[str, str]]:
    require(0 < train_ratio < 1, "train-ratio must be between 0 and 1")
    require(0 <= val_ratio < 1, "val-ratio must be between 0 and 1")
    require(train_ratio + val_ratio < 1, "train-ratio + val-ratio must be below 1")

    rng = random.Random(seed)
    result: list[dict[str, str]] = []
    for label in sorted(VALID_LABELS):
        label_rows = [row for row in rows if row["eval_label"] == label]
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_count = max(1, int(round(total * train_ratio)))
        val_count = max(1, int(round(total * val_ratio))) if total >= 3 else 0
        if train_count + val_count >= total:
            train_count = max(1, total - 2) if total >= 3 else max(1, total - 1)
            val_count = 1 if total >= 3 else 0
        for index, row in enumerate(label_rows):
            split = "train"
            if index >= train_count + val_count:
                split = "test"
            elif index >= train_count:
                split = "val"
            copied = dict(row)
            copied["split"] = split
            result.append(copied)
    return result


def balance_train_rows(
    rows: list[dict[str, str]],
    seed: int,
    max_train_majority_ratio: float,
) -> list[dict[str, str]]:
    if max_train_majority_ratio <= 0:
        return rows

    train_rows_by_label = {
        label: [row for row in rows if row["split"] == "train" and row["eval_label"] == label]
        for label in sorted(VALID_LABELS)
    }
    min_count = min(len(label_rows) for label_rows in train_rows_by_label.values())
    cap = max(1, int(round(min_count * max_train_majority_ratio)))

    rng = random.Random(seed)
    kept_ids: set[int] = set()
    for label_rows in train_rows_by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        for row in shuffled[:cap]:
            kept_ids.add(id(row))

    balanced: list[dict[str, str]] = []
    for row in rows:
        if row["split"] != "train" or id(row) in kept_ids:
            balanced.append(row)
    return balanced


def clean_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        for label in sorted(VALID_LABELS):
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def destination_name(row: dict[str, str], index: int) -> str:
    source_name = Path(row["source_path"]).name
    digest = hashlib.sha1(row["source_path"].encode("utf-8")).hexdigest()[:10]
    return f"{index:05d}_{safe_filename(row['source_class'])}_{digest}_{source_name}"


def copy_or_link(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def materialize_dataset(
    rows: list[dict[str, str]],
    source_dir: Path,
    output_dir: Path,
    copy_mode: str,
) -> list[dict[str, str]]:
    materialized: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        source = source_dir / row["source_path"]
        require(source.exists(), f"missing source image: {source}")
        destination = output_dir / row["split"] / row["eval_label"] / destination_name(row, index)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_or_link(source, destination, copy_mode)
        copied = dict(row)
        copied["local_path"] = str(destination.relative_to(output_dir))
        materialized.append(copied)
    return materialized


def write_manifest(rows: list[dict[str, str]], output_dir: Path) -> None:
    manifest_path = output_dir / "weak_dataset_manifest.csv"
    fieldnames = sorted({field for row in rows for field in row})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]], output_dir: Path, args: argparse.Namespace) -> None:
    counts = Counter((row["split"], row["eval_label"]) for row in rows)
    source_counts = Counter((row["source_class"], row["eval_label"]) for row in rows)
    summary = {
        "source_dir": str(args.source_dir),
        "output_dir": str(output_dir),
        "mapping_preset": args.mapping_preset,
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "max_train_majority_ratio": args.max_train_majority_ratio,
        "counts": {
            f"{split}/{label}": counts[(split, label)]
            for split in ("train", "val", "test")
            for label in sorted(VALID_LABELS)
        },
        "source_class_counts": {
            f"{source_class}/{label}": count
            for (source_class, label), count in sorted(source_counts.items())
        },
        "total": len(rows),
        "note": "Weak-label folder mapping. Review source-class mappings before final training claims.",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    mapping = class_mapping(args)
    rows = find_image_rows(args.source_dir, mapping)
    split = split_rows(rows, args.seed, args.train_ratio, args.val_ratio)
    balanced = balance_train_rows(split, args.seed, args.max_train_majority_ratio)
    clean_output(args.output_dir)
    materialized = materialize_dataset(balanced, args.source_dir, args.output_dir, args.copy_mode)
    write_manifest(materialized, args.output_dir)
    write_summary(materialized, args.output_dir, args)


if __name__ == "__main__":
    main()
