from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train-yolo-cls.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_yolo_cls", SCRIPT)
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
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp)
        require(
            module.next_serial_name(project, "realwaste") == "realwaste-001",
            "empty project should start at 001",
        )
        (project / "realwaste-001").mkdir()
        (project / "realwaste-003").mkdir()
        (project / "realwaste-not-a-number").mkdir()
        (project / "other-002").mkdir()
        require(
            module.next_serial_name(project, "realwaste") == "realwaste-002",
            "serial naming should fill the first available gap",
        )

    print("[OK] training run serial names are deterministic and non-overwriting")


if __name__ == "__main__":
    main()
