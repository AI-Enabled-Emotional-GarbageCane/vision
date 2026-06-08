#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


TACO_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)
TACO_SOURCE_URL = "https://github.com/pedropro/TACO"
USER_AGENT = "Codex local academic weak-label dataset collector"

# Weak accept categories are intentionally focused on general-trash items that the
# current model missed: cigarette butts, tissue/napkins, garbage bags, wrappers,
# dirty packaging residue, and small non-recyclable miscellany.
ACCEPT_WEAK_CATEGORIES = {
    "Cigarette",
    "Tissues",
    "Garbage bag",
    "Crisp packet",
    "Other plastic wrapper",
    "Plastic film",
    "Single-use carrier bag",
    "Styrofoam piece",
    "Disposable food container",
    "Foam food container",
    "Plastic utensils",
}

# Keep obvious recyclable/reject objects as hard negatives. Categories that can be
# policy-dependent when contaminated are kept out of this set unless they are clear
# hard negatives for the current public contract.
REJECT_WEAK_CATEGORIES = {
    "Other plastic bottle",
    "Clear plastic bottle",
    "Glass bottle",
    "Plastic bottle cap",
    "Metal bottle cap",
    "Broken glass",
    "Food Can",
    "Aerosol",
    "Drink can",
    "Other carton",
    "Egg carton",
    "Drink carton",
    "Corrugated carton",
    "Meal carton",
    "Pizza box",
    "Paper cup",
    "Disposable plastic cup",
    "Glass cup",
    "Other plastic cup",
    "Food waste",
    "Glass jar",
    "Plastic lid",
    "Metal lid",
    "Other plastic",
    "Magazine paper",
    "Wrapping paper",
    "Normal paper",
    "Paper bag",
    "Plastified paper bag",
    "Spread tub",
    "Tupperware",
    "Other plastic container",
    "Pop tab",
    "Scrap metal",
    "Plastic straw",
    "Paper straw",
    "Aluminium foil",
    "Battery",
}

FIELDNAMES = [
    "local_path",
    "source",
    "source_dataset_url",
    "image_id",
    "source_file_name",
    "source_url",
    "license",
    "label_hint",
    "eval_label",
    "eval_label_strength",
    "dominant_taco_category",
    "all_taco_categories",
    "review_status",
    "bytes",
    "sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a weak-label inference/fine-tune image sample from TACO.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/inference_extra_waste/taco_reject_safety"),
    )
    parser.add_argument("--accept-target", type=int, default=20)
    parser.add_argument("--reject-target", type=int, default=80)
    parser.add_argument(
        "--per-category-cap",
        type=int,
        default=12,
        help="Maximum images sampled per dominant TACO category.",
    )
    parser.add_argument("--annotations-url", default=TACO_ANNOTATIONS_URL)
    parser.add_argument(
        "--allow-mixed",
        action="store_true",
        help="Allow images containing both weak accept and reject categories.",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=0.45,
        help="Seconds to sleep after each newly downloaded image.",
    )
    return parser.parse_args()


def request_bytes(url: str, timeout: int = 90, attempt: int = 0) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        if attempt < 3:
            time.sleep(2 * (attempt + 1))
            return request_bytes(url, timeout=timeout, attempt=attempt + 1)
        raise exc


def load_taco_annotations(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url, timeout=90).decode("utf-8"))


def ordered_categories_for_image(
    image_id: int,
    annotations_by_image: dict[int, list[dict[str, Any]]],
    category_by_id: dict[int, str],
) -> list[str]:
    categories: list[str] = []
    for annotation in annotations_by_image.get(image_id, []):
        name = category_by_id[int(annotation["category_id"])]
        if name not in categories:
            categories.append(name)
    return categories


def assign_weak_label(
    categories: list[str],
    allow_mixed: bool = False,
) -> tuple[str | None, str | None, str]:
    category_set = set(categories)
    accept_matches = [name for name in categories if name in ACCEPT_WEAK_CATEGORIES]
    reject_matches = [name for name in categories if name in REJECT_WEAK_CATEGORIES]

    if accept_matches and reject_matches and not allow_mixed:
        return None, None, "mixed_accept_reject_categories"
    if accept_matches:
        return "accept", accept_matches[0], "weak_accept_category"
    if reject_matches:
        return "reject", reject_matches[0], "weak_reject_category"
    if category_set:
        return None, None, "unmapped_categories"
    return None, None, "no_categories"


def safe_filename(image_id: int, category: str, url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix not in {".jpg", ".png", ".webp"}:
        suffix = ".jpg"
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", category).strip("_")[:40] or "sample"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"taco_{image_id:05d}_{slug}_{digest}{suffix}"


def download_image(url: str, destination: Path, download_delay: float) -> tuple[int, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=90) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    if not content_type.startswith("image/"):
        raise RuntimeError(f"not an image: {content_type}")
    destination.write_bytes(data)
    if download_delay > 0:
        time.sleep(download_delay)
    return len(data), content_type


def build_candidates(data: dict[str, Any], allow_mixed: bool) -> list[dict[str, Any]]:
    category_by_id = {int(category["id"]): category["name"] for category in data["categories"]}
    image_by_id = {int(image["id"]): image for image in data["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in data["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    candidates: list[dict[str, Any]] = []
    for image_id in sorted(image_by_id):
        image = image_by_id[image_id]
        source_url = image.get("flickr_640_url") or image.get("flickr_url")
        if not source_url:
            continue

        categories = ordered_categories_for_image(image_id, annotations_by_image, category_by_id)
        eval_label, dominant_category, reason = assign_weak_label(categories, allow_mixed)
        if eval_label is None or dominant_category is None:
            continue

        candidates.append(
            {
                "image_id": image_id,
                "image": image,
                "source_url": source_url,
                "eval_label": eval_label,
                "label_hint": f"{eval_label}_weak",
                "dominant_taco_category": dominant_category,
                "all_taco_categories": categories,
                "mapping_reason": reason,
            }
        )
    return candidates


def selected_enough(counts: Counter[str], accept_target: int, reject_target: int) -> bool:
    return counts["accept_weak"] >= accept_target and counts["reject_weak"] >= reject_target


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    image_root = output_dir / "images"
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    (image_root / "accept_weak").mkdir(parents=True, exist_ok=True)
    (image_root / "reject_weak").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading TACO annotations: {args.annotations_url}")
    data = load_taco_annotations(args.annotations_url)
    candidates = build_candidates(data, allow_mixed=args.allow_mixed)

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    per_category_counts: Counter[tuple[str, str]] = Counter()
    seen_sha256: set[str] = set()

    for candidate in candidates:
        label_hint = candidate["label_hint"]
        if label_hint == "accept_weak" and counts[label_hint] >= args.accept_target:
            continue
        if label_hint == "reject_weak" and counts[label_hint] >= args.reject_target:
            continue

        category = candidate["dominant_taco_category"]
        category_key = (label_hint, category)
        if per_category_counts[category_key] >= args.per_category_cap:
            continue

        image = candidate["image"]
        source_url = candidate["source_url"]
        relative_path = (
            Path("images")
            / label_hint
            / safe_filename(int(candidate["image_id"]), category, source_url)
        )
        destination = output_dir / relative_path

        try:
            if destination.exists():
                byte_count = destination.stat().st_size
            else:
                byte_count, _content_type = download_image(
                    source_url,
                    destination,
                    args.download_delay,
                )
        except Exception as exc:  # noqa: BLE001 - keep batch collection moving.
            print(f"[WARN] skip image_id={candidate['image_id']} {category}: {exc}")
            continue

        sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        if sha256 in seen_sha256:
            continue
        seen_sha256.add(sha256)

        counts[label_hint] += 1
        per_category_counts[category_key] += 1
        rows.append(
            {
                "local_path": str(relative_path),
                "source": "TACO dataset sample",
                "source_dataset_url": TACO_SOURCE_URL,
                "image_id": candidate["image_id"],
                "source_file_name": image.get("file_name", ""),
                "source_url": source_url,
                "license": str(image.get("license") or "see TACO annotation/image source"),
                "label_hint": label_hint,
                "eval_label": candidate["eval_label"],
                "eval_label_strength": candidate["mapping_reason"],
                "dominant_taco_category": category,
                "all_taco_categories": "|".join(candidate["all_taco_categories"]),
                "review_status": "needs_human_review",
                "bytes": byte_count,
                "sha256": sha256,
            }
        )
        print(
            f"[{label_hint}] {counts[label_hint]:02d} "
            f"image_id={candidate['image_id']} category={category}",
            flush=True,
        )

        if selected_enough(counts, args.accept_target, args.reject_target):
            break

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": "TACO",
        "source_url": TACO_SOURCE_URL,
        "purpose": (
            "reject-safety weak-label inference/fine-tune candidates; "
            "review before training"
        ),
        "counts": dict(counts),
        "per_category_counts": {
            f"{label}:{category}": count
            for (label, category), count in sorted(per_category_counts.items())
        },
        "total": len(rows),
        "manifest": str(manifest_path),
        "review_required": True,
        "accept_weak_categories": sorted(ACCEPT_WEAK_CATEGORIES),
        "reject_weak_categories": sorted(REJECT_WEAK_CATEGORIES),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
