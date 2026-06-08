from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-l515-demo-experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_l515_demo_experiment", SCRIPT)
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
        raw_dir = Path(tmp) / "l515_demo_raw"
        module.init_raw_dirs(raw_dir)

        for slug in module.ACCEPT_DIRS + module.REJECT_DIRS:
            require((raw_dir / slug).exists(), f"missing raw class folder: {slug}")

        (raw_dir / "flexible_wrapper" / "wrapper.jpg").write_bytes(b"fake")
        (raw_dir / "dirty_wrapper" / "dirty.png").write_bytes(b"fake")
        (raw_dir / "drink_can" / "can.jpg").write_bytes(b"fake")
        (raw_dir / "glass_metal" / "glass.webp").write_bytes(b"fake")

        counts = module.raw_counts(raw_dir)
        require(counts["accept_count"] == 2, "accept image count should include accept dirs")
        require(counts["reject_count"] == 2, "reject image count should include reject dirs")
        require(module.has_both_labels(counts), "dataset should have both labels")
        require(
            not module.has_minimum_data(counts, min_accept=300, min_reject=300),
            "tiny raw dataset should not pass the fine-tune gate",
        )
        require(
            module.class_name("paper_cardboard") == "paper cardboard",
            "folder slugs should map to source class names used by the folder builder",
        )

    print("[OK] L515 demo experiment raw classes and data gate are deterministic")


if __name__ == "__main__":
    main()
