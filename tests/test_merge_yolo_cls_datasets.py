from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge-yolo-cls-datasets.py"


def load_module():
    spec = importlib.util.spec_from_file_location("merge_yolo_cls_datasets", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_dataset(root: Path, prefix: str) -> None:
    for split in ("train", "val", "test"):
        for label in ("accept", "reject"):
            class_dir = root / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            (class_dir / f"{prefix}_{split}_{label}.jpg").write_bytes(b"fake image")


def main() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset_a = tmp_path / "a"
        dataset_b = tmp_path / "b"
        output_dir = tmp_path / "merged"
        make_dataset(dataset_a, "a")
        make_dataset(dataset_b, "b")

        module.clean_output(output_dir)
        rows = []
        rows.extend(module.merge_source("a", dataset_a, output_dir, "copy"))
        rows.extend(module.merge_source("b", dataset_b, output_dir, "copy"))
        module.write_manifest(rows, output_dir)

        require(len(rows) == 12, "two complete tiny datasets should produce 12 rows")
        with (output_dir / "weak_dataset_manifest.csv").open(encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        require(len(manifest_rows) == len(rows), "manifest should include merged rows")
        require(
            all((output_dir / row["local_path"]).exists() for row in manifest_rows),
            "merged manifest local_path values should exist",
        )

    print("[OK] YOLO classification dataset merger preserves split and label folders")


if __name__ == "__main__":
    main()
