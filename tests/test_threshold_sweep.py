from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "threshold-sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("threshold_sweep", SCRIPT)
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
        {"eval_label": "accept", "pred_class": "accept", "confidence": "0.90"},
        {"eval_label": "accept", "pred_class": "accept", "confidence": "0.70"},
        {"eval_label": "reject", "pred_class": "accept", "confidence": "0.80"},
        {"eval_label": "reject", "pred_class": "reject", "confidence": "0.99"},
    ]
    summary = module.summarize_threshold(rows, accept_threshold=0.76, uncertain_threshold=0.50)
    require(summary["accept_count"] == 2, "accept denominator should be counted")
    require(summary["accepted_accept_count"] == 1, "only high-confidence accept should pass")
    require(summary["gate_accept_recall"] == 0.5, "accept recall should reflect gate accepts")
    require(summary["reject_count"] == 2, "reject denominator should be counted")
    require(summary["false_accept_count_on_reject"] == 1, "reject false accept should be counted")
    require(
        summary["false_accept_rate_on_reject"] == 0.5,
        "reject false accept rate should reflect gate accepts",
    )
    strict_summary = module.summarize_threshold(
        rows,
        accept_threshold=0.85,
        uncertain_threshold=0.50,
    )
    require(
        strict_summary["false_accept_count_on_reject"] == 0,
        "stricter threshold should block the 0.80 reject false accept",
    )

    print("[OK] threshold sweep matches accept-gate semantics")


if __name__ == "__main__":
    main()
