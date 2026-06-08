#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


VALID_LABELS = {"accept", "reject"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO classification dataset from a weak-label manifest.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Directory that manifest local_path values are relative to.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use hardlink to save disk space when source and destination are on same filesystem.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[FAIL] {message}")


def read_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"manifest has no rows: {manifest}")
    return rows


def clean_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        for label in sorted(VALID_LABELS):
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)


def split_rows(
    rows: list[dict[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> list[dict[str, str]]:
    require(0 < train_ratio < 1, "train-ratio must be between 0 and 1")
    require(0 <= val_ratio < 1, "val-ratio must be between 0 and 1")
    require(train_ratio + val_ratio < 1, "train-ratio + val-ratio must be below 1")

    usable = [row for row in rows if row.get("eval_label") in VALID_LABELS]
    require(usable, "no manifest rows with eval_label accept/reject")

    result: list[dict[str, str]] = []
    rng = random.Random(seed)
    for label in sorted(VALID_LABELS):
        label_rows = [row for row in usable if row["eval_label"] == label]
        require(label_rows, f"no rows for label: {label}")
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


def destination_name(row: dict[str, str], index: int) -> str:
    source_name = Path(row["local_path"]).name
    digest_source = row.get("sha256") or row["local_path"]
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"{index:05d}_{digest}_{source_name}"


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
    source_root: Path,
    output_dir: Path,
    copy_mode: str,
) -> list[dict[str, str]]:
    materialized: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        label = row["eval_label"]
        split = row["split"]
        source = source_root / row["local_path"]
        require(source.exists(), f"missing source image: {source}")
        destination = output_dir / split / label / destination_name(row, index)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_or_link(source, destination, copy_mode)
        copied = dict(row)
        copied["dataset_path"] = str(destination.relative_to(output_dir))
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
    summary = {
        "source_manifest": str(args.manifest),
        "source_root": str(args.source_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "counts": {
            f"{split}/{label}": counts[(split, label)]
            for split in ("train", "val", "test")
            for label in sorted(VALID_LABELS)
        },
        "total": len(rows),
        "note": "Weak-label YOLO classification dataset. Review labels before final training claims.",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root or args.manifest.parent
    rows = read_rows(args.manifest)
    split = split_rows(rows, args.seed, args.train_ratio, args.val_ratio)
    clean_output(args.output_dir)
    materialized = materialize_dataset(split, args.source_root, args.output_dir, args.copy_mode)
    write_manifest(materialized, args.output_dir)
    write_summary(materialized, args.output_dir, args)


if __name__ == "__main__":
    main()
