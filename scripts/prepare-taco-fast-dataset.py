#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "scripts" / "prepare-weak-finetune-dataset.py"


def load_prepare_module() -> Any:
    spec = importlib.util.spec_from_file_location("prepare_weak_finetune_dataset", PREPARE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PREPARE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a larger TACO weak-label dataset with parallel image downloads.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-target", type=int, default=1000)
    parser.add_argument("--reject-target", type=int, default=1000)
    parser.add_argument("--per-category-cap", type=int, default=999)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--allow-mixed", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[FAIL] {message}")


def select_candidates(
    candidates: list[dict[str, Any]],
    accept_target: int,
    reject_target: int,
    per_category_cap: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    per_category_counts: Counter[tuple[str, str]] = Counter()
    for candidate in candidates:
        label_hint = candidate["label_hint"]
        if label_hint == "accept_weak" and counts[label_hint] >= accept_target:
            continue
        if label_hint == "reject_weak" and counts[label_hint] >= reject_target:
            continue
        category_key = (label_hint, candidate["dominant_taco_category"])
        if per_category_counts[category_key] >= per_category_cap:
            continue
        selected.append(candidate)
        counts[label_hint] += 1
        per_category_counts[category_key] += 1
        if counts["accept_weak"] >= accept_target and counts["reject_weak"] >= reject_target:
            break
    return selected


def download_image(url: str, destination: Path, user_agent: str, timeout: int) -> int:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    if not content_type.startswith("image/"):
        raise RuntimeError(f"not an image: {content_type}")
    destination.write_bytes(data)
    return len(data)


def materialize_candidate(
    candidate: dict[str, Any],
    output_dir: Path,
    module: Any,
    timeout: int,
) -> tuple[dict[str, Any] | None, str | None]:
    image = candidate["image"]
    source_url = candidate["source_url"]
    category = candidate["dominant_taco_category"]
    relative_path = (
        Path("images")
        / candidate["label_hint"]
        / module.safe_filename(int(candidate["image_id"]), category, source_url)
    )
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if destination.exists():
            byte_count = destination.stat().st_size
        else:
            byte_count = download_image(source_url, destination, module.USER_AGENT, timeout)
        sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    except Exception as exc:  # noqa: BLE001 - keep batch collection moving.
        return None, f"skip image_id={candidate['image_id']} {category}: {exc}"

    return (
        {
            "local_path": str(relative_path),
            "source": "TACO dataset sample",
            "source_dataset_url": module.TACO_SOURCE_URL,
            "image_id": candidate["image_id"],
            "source_file_name": image.get("file_name", ""),
            "source_url": source_url,
            "license": str(image.get("license") or "see TACO annotation/image source"),
            "label_hint": candidate["label_hint"],
            "eval_label": candidate["eval_label"],
            "eval_label_strength": candidate["mapping_reason"],
            "dominant_taco_category": category,
            "all_taco_categories": "|".join(candidate["all_taco_categories"]),
            "review_status": "needs_human_review",
            "bytes": byte_count,
            "sha256": sha256,
        },
        None,
    )


def main() -> None:
    args = parse_args()
    require(args.workers > 0, "workers must be positive")
    module = load_prepare_module()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "accept_weak").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "reject_weak").mkdir(parents=True, exist_ok=True)

    data = module.load_taco_annotations(module.TACO_ANNOTATIONS_URL)
    candidates = module.build_candidates(data, allow_mixed=args.allow_mixed)
    selected = select_candidates(
        candidates,
        accept_target=args.accept_target,
        reject_target=args.reject_target,
        per_category_cap=args.per_category_cap,
    )
    print(f"[INFO] selected {len(selected)} TACO candidates for download")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(materialize_candidate, candidate, output_dir, module, args.timeout)
            for candidate in selected
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            row, warning = future.result()
            if warning:
                warnings.append(warning)
                print(f"[WARN] {warning}", flush=True)
            if row:
                rows.append(row)
            if index % 50 == 0:
                counts = Counter(row["label_hint"] for row in rows)
                print(
                    f"[INFO] processed {index}/{len(selected)} "
                    f"accept={counts['accept_weak']} reject={counts['reject_weak']}",
                    flush=True,
                )

    rows.sort(key=lambda row: int(row["image_id"]))
    deduped: list[dict[str, Any]] = []
    seen_sha256: set[str] = set()
    for row in rows:
        if row["sha256"] in seen_sha256:
            continue
        seen_sha256.add(row["sha256"])
        deduped.append(row)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.FIELDNAMES)
        writer.writeheader()
        writer.writerows(deduped)

    counts = Counter(row["label_hint"] for row in deduped)
    per_category_counts = Counter(
        (row["label_hint"], row["dominant_taco_category"]) for row in deduped
    )
    summary = {
        "source": "TACO",
        "source_url": module.TACO_SOURCE_URL,
        "purpose": "general-trash positive weak-label fine-tune candidates; review before claims",
        "counts": dict(counts),
        "per_category_counts": {
            f"{label}:{category}": count
            for (label, category), count in sorted(per_category_counts.items())
        },
        "total": len(deduped),
        "selected_candidates": len(selected),
        "download_warnings": warnings[:200],
        "download_warning_count": len(warnings),
        "manifest": str(manifest_path),
        "review_required": True,
        "accept_weak_categories": sorted(module.ACCEPT_WEAK_CATEGORIES),
        "reject_weak_categories": sorted(module.REJECT_WEAK_CATEGORIES),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
