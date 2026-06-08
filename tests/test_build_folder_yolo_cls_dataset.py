from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build-folder-yolo-cls-dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_folder_yolo_cls_dataset", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def touch_images(class_dir: Path, count: int) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (class_dir / f"image_{index}.jpg").write_bytes(b"fake image")


def main() -> None:
    module = load_module()
    args = SimpleNamespace(
        mapping_preset="realwaste",
        accept_class=[],
        reject_class=[],
    )
    mapping = module.class_mapping(args)
    require(mapping["Miscellaneous Trash"] == "accept", "RealWaste misc trash should be accept")
    require(mapping["Plastic"] == "reject", "RealWaste plastic should be reject")

    with tempfile.TemporaryDirectory() as tmp:
        source_dir = Path(tmp) / "source"
        output_dir = Path(tmp) / "dataset"
        touch_images(source_dir / "Miscellaneous Trash", 8)
        touch_images(source_dir / "Textile Trash", 4)
        touch_images(source_dir / "Plastic", 20)
        touch_images(source_dir / "Paper", 10)

        rows = module.find_image_rows(source_dir, mapping)
        split = module.split_rows(rows, seed=7, train_ratio=0.7, val_ratio=0.15)
        balanced = module.balance_train_rows(split, seed=7, max_train_majority_ratio=1.0)
        train_counts = {}
        for row in balanced:
            if row["split"] == "train":
                train_counts[row["eval_label"]] = train_counts.get(row["eval_label"], 0) + 1
        require(
            train_counts["accept"] == train_counts["reject"],
            "train balancing should cap the majority label",
        )

        module.clean_output(output_dir)
        materialized = module.materialize_dataset(
            balanced,
            source_dir,
            output_dir,
            copy_mode="copy",
        )
        module.write_manifest(materialized, output_dir)
        with (output_dir / "weak_dataset_manifest.csv").open(encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        require(len(manifest_rows) == len(materialized), "manifest row count should match")
        require(
            all((output_dir / row["local_path"]).exists() for row in manifest_rows),
            "manifest local_path values should point to materialized files",
        )

    print("[OK] folder classification dataset builder maps, balances, and materializes data")


if __name__ == "__main__":
    main()
