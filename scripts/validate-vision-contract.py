#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_text(path: str) -> str:
    file_path = ROOT / path
    require(file_path.exists(), f"missing required file: {path}")
    return file_path.read_text(encoding="utf-8")


def read_json(path: str) -> dict:
    file_path = ROOT / path
    require(file_path.exists(), f"missing required file: {path}")
    with file_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require_contains(text: str, needle: str, path: str) -> None:
    require(needle in text, f"{path} must contain {needle!r}")


def validate_lock() -> None:
    lock = read_json("contracts/contract.lock.json")
    require(lock.get("contract_version") == "v0.3", "contract.lock.json must use v0.3")
    require(lock.get("module") == "vision", "contract.lock.json module must be vision")
    require(lock.get("owned_events") == ["recognition_result"], "vision must own recognition_result")
    require(lock.get("consumed_events") == ["user_detected"], "vision must consume user_detected")


def validate_docs() -> None:
    readme = read_text("README.md")
    spec = read_text("docs/vision-spec.md")

    for path, text in {"README.md": readme, "docs/vision-spec.md": spec}.items():
        for needle in [
            "v0.3",
            "YOLOv11n",
            "L515",
            "user_detected",
            "recognition_result",
            "accept",
            "reject",
            "num_objects",
            "snapshot_path",
        ]:
            require_contains(text, needle, path)

    require_contains(spec, "num_objects=1", "docs/vision-spec.md")
    require_contains(spec, "confidence < 0.5", "docs/vision-spec.md")
    require_contains(spec, "每類 50-100 張", "docs/vision-spec.md")
    require_contains(spec, "false_accept_rate_on_reject", "docs/vision-spec.md")
    require_contains(spec, "accept_threshold=0.76", "docs/vision-spec.md")
    require_contains(spec, "payload class 保留模型 best guess", "docs/vision-spec.md")


def validate_source_contract() -> None:
    source = read_text("src/vision_contract.py")
    policy_source = read_text("src/vision_policy.py")
    require_contains(source, "RECOGNITION_RESULT_FIELDS", "src/vision_contract.py")
    require_contains(source, "num_objects", "src/vision_contract.py")
    require_contains(source, '"recognition_result"', "src/vision_contract.py")
    require_contains(policy_source, "DEFAULT_ACCEPT_THRESHOLD", "src/vision_policy.py")
    require_contains(policy_source, "should_allow_accept", "src/vision_policy.py")


def main() -> None:
    validate_lock()
    validate_docs()
    validate_source_contract()
    print("[OK] vision contract docs and lock are consistent")


if __name__ == "__main__":
    main()
