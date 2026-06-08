from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download-hf-class-sample.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_hf_class_sample", SCRIPT)
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
    mapping = module.parse_class_maps(["trash:accept", "glass:reject"])
    require(mapping == {"trash": "accept", "glass": "reject"}, "class maps should parse")

    repo_files = [
        "README.md",
        "trash/a.jpg",
        "trash/b.png",
        "trash/c.txt",
        "glass/a.jpg",
        "glass/b.jpg",
        "glass/c.jpg",
        "paper/a.jpg",
    ]
    selected = module.select_files(
        repo_files=repo_files,
        class_mapping=mapping,
        per_class_cap=2,
        seed=1,
    )
    require(len(selected["trash"]) == 2, "image selection should include mapped accept files")
    require(len(selected["glass"]) == 2, "per-class cap should limit mapped reject files")
    require("paper" not in selected, "unmapped classes should not be selected")

    print("[OK] Hugging Face class sampler maps and caps selected image classes")


if __name__ == "__main__":
    main()
