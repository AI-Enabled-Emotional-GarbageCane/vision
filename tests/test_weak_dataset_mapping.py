from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare-weak-finetune-dataset.py"


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_weak_finetune_dataset", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    module = load_prepare_module()

    label, dominant, reason = module.assign_weak_label(["Tissues"])
    require(label == "accept", "tissues should map to weak accept")
    require(dominant == "Tissues", "dominant accept category should be preserved")
    require(reason == "weak_accept_category", "accept reason should be explicit")

    label, dominant, reason = module.assign_weak_label(["Clear plastic bottle"])
    require(label == "reject", "clear plastic bottle should map to weak reject")
    require(dominant == "Clear plastic bottle", "dominant reject category should be preserved")
    require(reason == "weak_reject_category", "reject reason should be explicit")

    label, dominant, reason = module.assign_weak_label(["Tissues", "Drink can"])
    require(label is None, "mixed accept/reject images should be skipped by default")
    require(dominant is None, "mixed skipped images should not have dominant category")
    require(reason == "mixed_accept_reject_categories", "mixed reason should be explicit")

    label, dominant, reason = module.assign_weak_label(["Tissues", "Drink can"], allow_mixed=True)
    require(label == "accept", "allow_mixed should prefer accept when accept category exists")
    require(dominant == "Tissues", "allow_mixed should preserve accept dominant category")

    label, dominant, reason = module.assign_weak_label(["Unlabeled litter"])
    require(label is None, "unmapped categories should be skipped")
    require(reason == "unmapped_categories", "unmapped reason should be explicit")

    print("[OK] weak dataset category mapping is explicit and conservative")


if __name__ == "__main__":
    main()
