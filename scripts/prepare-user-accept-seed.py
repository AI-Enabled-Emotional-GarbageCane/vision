#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VALID_SPLITS = ("train", "val", "test")
VALID_LABELS = ("accept", "reject")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a user-provided accept-only seed dataset for YOLO classification.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("Dataset-20260608T092356Z-3-001.zip"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/sources/user_accept_seed_20260608"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/training/user_accept_seed_yolo_cls"),
    )
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-repeat", type=int, default=8)
    parser.add_argument("--capture-source", default="normal_camera")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="hardlink")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[FAIL] {message}")


def require_image_dependencies() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "Missing image dependency. Run with: "
            "uv run --with pillow python scripts/prepare-user-accept-seed.py ..."
        ) from exc
    return Image, ImageDraw, ImageOps


def bilinear_resample(image_module: Any) -> Any:
    resampling = getattr(image_module, "Resampling", image_module)
    return resampling.BILINEAR


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "image"


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def zip_image_names(zip_path: Path) -> list[str]:
    require(zip_path.exists(), f"missing zip file: {zip_path}")
    with ZipFile(zip_path) as archive:
        names = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
        ]
    return sorted(names)


def rgb_image_from_bytes(data: bytes, image_module: Any) -> Any:
    image = image_module.open(BytesIO(data))
    if image.mode in {"RGBA", "LA"}:
        background = image_module.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        return background.convert("RGB")
    return image.convert("RGB")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_source_images(
    zip_path: Path,
    source_dir: Path,
    capture_source: str,
) -> list[dict[str, str]]:
    image_module, _image_draw, _image_ops = require_image_dependencies()
    clean_dir(source_dir)
    image_dir = source_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with ZipFile(zip_path) as archive:
        for index, name in enumerate(zip_image_names(zip_path)):
            data = archive.read(name)
            digest = hashlib.sha256(data).hexdigest()
            source_name = safe_name(Path(name).stem)
            destination = image_dir / f"{index:05d}_{digest[:10]}_{source_name}.jpg"
            image = rgb_image_from_bytes(data, image_module)
            image.save(destination, format="JPEG", quality=95)
            rows.append(
                {
                    "local_path": str(destination.relative_to(source_dir)),
                    "original_zip_path": name,
                    "sha256": digest,
                    "eval_label": "accept",
                    "eval_label_strength": "user_confirmed_accept",
                    "capture_source": capture_source,
                    "label_hint": "user_accept_seed",
                }
            )

    require(rows, f"no images found in zip: {zip_path}")
    write_csv(source_dir / "manifest.csv", rows)
    write_contact_sheet(source_dir / "contact_sheet.jpg", rows, source_dir)
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
    require(len(rows) >= 3, "at least 3 images are required to create train/val/test splits")

    shuffled = [dict(row) for row in rows]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))
    if train_count + val_count >= total:
        train_count = max(1, total - 2)
        val_count = 1

    result: list[dict[str, str]] = []
    for index, row in enumerate(shuffled):
        split = "train"
        if index >= train_count + val_count:
            split = "test"
        elif index >= train_count:
            split = "val"
        copied = dict(row)
        copied["split"] = split
        result.append(copied)
    return result


def init_yolo_dirs(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in VALID_SPLITS:
        for label in VALID_LABELS:
            (dataset_dir / split / label).mkdir(parents=True, exist_ok=True)


def copy_or_link(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def materialized_name(row: dict[str, str], output_index: int, repeat_index: int) -> str:
    source_name = Path(row["local_path"]).name
    digest = row["sha256"][:10]
    return f"{output_index:05d}_r{repeat_index:02d}_{digest}_{source_name}"


def materialize_yolo_dataset(
    rows: list[dict[str, str]],
    source_dir: Path,
    dataset_dir: Path,
    train_repeat: int,
    copy_mode: str,
) -> list[dict[str, str]]:
    require(train_repeat >= 1, "train-repeat must be >= 1")
    init_yolo_dirs(dataset_dir)

    materialized: list[dict[str, str]] = []
    output_index = 0
    for row in rows:
        repeats = train_repeat if row["split"] == "train" else 1
        for repeat_index in range(repeats):
            source = source_dir / row["local_path"]
            destination = (
                dataset_dir
                / row["split"]
                / "accept"
                / materialized_name(row, output_index, repeat_index)
            )
            copy_or_link(source, destination, copy_mode)
            copied = dict(row)
            copied["source_path"] = row["local_path"]
            copied["local_path"] = str(destination.relative_to(dataset_dir))
            copied["repeat_index"] = str(repeat_index)
            copied["is_train_repeat"] = str(row["split"] == "train" and repeat_index > 0)
            materialized.append(copied)
            output_index += 1
    write_csv(dataset_dir / "weak_dataset_manifest.csv", materialized)
    write_dataset_summary(dataset_dir, rows, materialized, train_repeat, copy_mode)
    return materialized


def write_dataset_summary(
    dataset_dir: Path,
    source_rows: list[dict[str, str]],
    materialized_rows: list[dict[str, str]],
    train_repeat: int,
    copy_mode: str,
) -> None:
    counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for split in VALID_SPLITS:
        for label in VALID_LABELS:
            counts[f"{split}/{label}"] = sum(
                1
                for row in materialized_rows
                if row["split"] == split and row["eval_label"] == label
            )
            source_counts[f"{split}/{label}"] = sum(
                1 for row in source_rows if row.get("split") == split and row["eval_label"] == label
            )

    summary = {
        "output_dir": str(dataset_dir),
        "copy_mode": copy_mode,
        "train_repeat": train_repeat,
        "counts": counts,
        "source_counts_before_repeat": source_counts,
        "total": len(materialized_rows),
        "source_total_before_repeat": len(source_rows),
        "note": "User accept-only seed dataset. Reject folders are intentionally empty.",
    }
    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_contact_sheet(
    path: Path,
    rows: list[dict[str, str]],
    source_dir: Path,
    imgsz: tuple[int, int] = (170, 125),
    columns: int = 5,
) -> None:
    image_module, image_draw, image_ops = require_image_dependencies()
    thumb_width, thumb_height = imgsz
    label_height = 38
    row_count = (len(rows) + columns - 1) // columns
    sheet = image_module.new(
        "RGB",
        (columns * thumb_width, row_count * (thumb_height + label_height)),
        "white",
    )
    draw = image_draw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        image = image_module.open(source_dir / row["local_path"]).convert("RGB")
        image = image_ops.contain(
            image,
            (thumb_width, thumb_height),
            bilinear_resample(image_module),
        )
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y))
        draw.text(
            (x + 4, y + thumb_height + 3),
            f"{index:02d} accept\n{Path(row['original_zip_path']).name[:22]}",
            fill="black",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def main() -> None:
    args = parse_args()
    source_rows = extract_source_images(args.zip_path, args.source_dir, args.capture_source)
    split = split_rows(source_rows, args.seed, args.train_ratio, args.val_ratio)
    write_csv(args.source_dir / "manifest.csv", split)
    materialize_yolo_dataset(
        split,
        args.source_dir,
        args.dataset_dir,
        train_repeat=args.train_repeat,
        copy_mode=args.copy_mode,
    )


if __name__ == "__main__":
    main()
