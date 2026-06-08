from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare-user-accept-seed.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_user_accept_seed", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rgba_png_bytes(color: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGBA", (18, 12), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_fixture_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        for index in range(6):
            archive.writestr(
                f"Dataset/user_accept_{index}.png",
                rgba_png_bytes((10 * index, 40, 120, 160)),
            )
        archive.writestr("Dataset/資料集需求.docx", b"not an image")
        archive.writestr("Dataset/readme.txt", b"not an image either")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "fixture.zip"
        source_dir = tmp_path / "source"
        dataset_dir = tmp_path / "dataset"
        make_fixture_zip(zip_path)

        rows = module.extract_source_images(zip_path, source_dir, "normal_camera")
        require(len(rows) == 6, "docx and non-image files should be ignored")
        require(
            all(row["eval_label"] == "accept" for row in rows),
            "all source rows should be accept",
        )
        require(
            all(row["capture_source"] == "normal_camera" for row in rows),
            "capture source should be preserved",
        )
        require(
            (source_dir / "contact_sheet.jpg").exists(),
            "source contact sheet should be written",
        )

        first_image = Image.open(source_dir / rows[0]["local_path"])
        require(first_image.mode == "RGB", "RGBA source image should be converted to RGB")

        split_rows = module.split_rows(rows, seed=20260602, train_ratio=0.70, val_ratio=0.15)
        materialized = module.materialize_yolo_dataset(
            split_rows,
            source_dir,
            dataset_dir,
            train_repeat=3,
            copy_mode="copy",
        )

        split_counts = {
            split: sum(1 for row in split_rows if row["split"] == split)
            for split in module.VALID_SPLITS
        }
        materialized_counts = {
            split: sum(1 for row in materialized if row["split"] == split)
            for split in module.VALID_SPLITS
        }
        require(
            materialized_counts["train"] == split_counts["train"] * 3,
            "train split should be repeated",
        )
        require(
            materialized_counts["val"] == split_counts["val"],
            "val split should not be repeated",
        )
        require(
            materialized_counts["test"] == split_counts["test"],
            "test split should not be repeated",
        )

        for split in module.VALID_SPLITS:
            require((dataset_dir / split / "accept").exists(), f"missing accept folder: {split}")
            require((dataset_dir / split / "reject").exists(), f"missing reject folder: {split}")
            require(
                not any((dataset_dir / split / "reject").iterdir()),
                f"reject folder should stay empty: {split}",
            )

        manifest_rows = read_csv(dataset_dir / "weak_dataset_manifest.csv")
        require(len(manifest_rows) == len(materialized), "manifest should include every copy")
        require(
            all((dataset_dir / row["local_path"]).exists() for row in manifest_rows),
            "manifest local paths should point to materialized files",
        )

        summary = json.loads((dataset_dir / "dataset_summary.json").read_text(encoding="utf-8"))
        require(summary["source_total_before_repeat"] == 6, "summary should keep source count")
        require(summary["train_repeat"] == 3, "summary should keep train repeat")

    print("[OK] user accept seed preprocessing extracts, labels, and repeats train rows")


if __name__ == "__main__":
    main()
