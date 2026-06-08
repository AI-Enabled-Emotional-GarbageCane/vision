from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build-yolo-cls-dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_yolo_cls_dataset", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    module = load_module()
    rows = [
        {"local_path": f"accept_{index}.jpg", "eval_label": "accept"}
        for index in range(6)
    ] + [
        {"local_path": f"reject_{index}.jpg", "eval_label": "reject"}
        for index in range(10)
    ]

    split_rows = module.split_rows(rows, seed=1, train_ratio=0.7, val_ratio=0.15)
    counts = {}
    for row in split_rows:
        counts[(row["split"], row["eval_label"])] = (
            counts.get((row["split"], row["eval_label"]), 0) + 1
        )

    require(counts[("train", "accept")] > 0, "accept train split should not be empty")
    require(counts[("val", "accept")] > 0, "accept val split should not be empty")
    require(counts[("test", "accept")] > 0, "accept test split should not be empty")
    require(counts[("train", "reject")] > 0, "reject train split should not be empty")
    require(counts[("val", "reject")] > 0, "reject val split should not be empty")
    require(counts[("test", "reject")] > 0, "reject test split should not be empty")

    with tempfile.TemporaryDirectory() as tmp:
        source_root = Path(tmp) / "source"
        output_dir = Path(tmp) / "dataset"
        source_root.mkdir()
        materialize_rows = []
        for index, row in enumerate(split_rows[:4]):
            image_path = source_root / row["local_path"]
            image_path.write_bytes(b"fake image")
            materialize_rows.append(row)
        materialized = module.materialize_dataset(
            materialize_rows,
            source_root,
            output_dir,
            copy_mode="copy",
        )
        require(len(materialized) == 4, "materialized row count should match input")
        require(
            all((output_dir / row["dataset_path"]).exists() for row in materialized),
            "materialized image paths should exist",
        )

        module.write_manifest(materialized, output_dir)
        manifest_path = output_dir / "weak_dataset_manifest.csv"
        with manifest_path.open(encoding="utf-8") as handle:
            require(len(list(csv.DictReader(handle))) == 4, "manifest should include rows")

    print("[OK] YOLO classification dataset builder splits and materializes data")


if __name__ == "__main__":
    main()
