#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VALID_LABELS = {"accept", "reject"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected class folders from a Hugging Face image dataset.",
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--class-map",
        action="append",
        required=True,
        help="Mapping in source_class:accept or source_class:reject form.",
    )
    parser.add_argument(
        "--per-class-cap",
        type=int,
        default=0,
        help="Maximum images per source class; <=0 downloads all mapped images.",
    )
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[FAIL] {message}")


def parse_class_maps(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise SystemExit(f"[FAIL] class-map must be source_class:label: {value}")
        source_class, label = value.split(":", 1)
        source_class = source_class.strip().strip("/")
        label = label.strip()
        require(source_class, f"empty source class in class-map: {value}")
        require(label in VALID_LABELS, f"invalid label in class-map: {value}")
        mapping[source_class] = label
    return mapping


def source_class_for_path(repo_path: str) -> str:
    return repo_path.split("/", 1)[0]


def is_image_path(repo_path: str) -> bool:
    return Path(repo_path).suffix.lower() in IMAGE_EXTENSIONS


def select_files(
    repo_files: list[str],
    class_mapping: dict[str, str],
    per_class_cap: int,
    seed: int,
) -> dict[str, list[str]]:
    by_class: dict[str, list[str]] = {source_class: [] for source_class in class_mapping}
    for repo_path in repo_files:
        source_class = source_class_for_path(repo_path)
        if source_class not in class_mapping or not is_image_path(repo_path):
            continue
        by_class[source_class].append(repo_path)

    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for source_class, paths in by_class.items():
        require(paths, f"no image files found for mapped class: {source_class}")
        paths = sorted(paths)
        if per_class_cap > 0 and len(paths) > per_class_cap:
            paths = sorted(rng.sample(paths, per_class_cap))
        selected[source_class] = paths
    return selected


def download_one(repo_id: str, output_dir: Path, repo_path: str) -> str:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=repo_path,
        local_dir=output_dir,
    )
    return str(Path(downloaded).relative_to(output_dir))


def download_selected(
    repo_id: str,
    output_dir: Path,
    selected: dict[str, list[str]],
    workers: int,
) -> tuple[list[str], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_paths = [repo_path for paths in selected.values() for repo_path in paths]
    downloaded: list[str] = []
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(download_one, repo_id, output_dir, repo_path): repo_path
            for repo_path in repo_paths
        }
        for future in as_completed(futures):
            repo_path = futures[future]
            try:
                downloaded.append(future.result())
            except Exception as exc:  # noqa: BLE001 - record per-file download failures.
                warnings.append(f"{repo_path}: {exc}")
    return sorted(downloaded), warnings


def write_summary(
    output_dir: Path,
    repo_id: str,
    class_mapping: dict[str, str],
    selected: dict[str, list[str]],
    downloaded: list[str],
    warnings: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_counts = Counter(
        (source_class, class_mapping[source_class])
        for source_class, paths in selected.items()
        for _path in paths
    )
    downloaded_counts = Counter(
        (source_class_for_path(path), class_mapping[source_class_for_path(path)])
        for path in downloaded
        if source_class_for_path(path) in class_mapping
    )
    summary = {
        "repo_id": repo_id,
        "output_dir": str(output_dir),
        "class_mapping": class_mapping,
        "per_class_cap": args.per_class_cap,
        "seed": args.seed,
        "workers": args.workers,
        "selected_counts": {
            f"{source_class}/{label}": count
            for (source_class, label), count in sorted(selected_counts.items())
        },
        "downloaded_counts": {
            f"{source_class}/{label}": count
            for (source_class, label), count in sorted(downloaded_counts.items())
        },
        "selected_total": sum(len(paths) for paths in selected.values()),
        "downloaded_total": len(downloaded),
        "warning_count": len(warnings),
        "warnings": warnings[:50],
    }
    (output_dir / "hf_source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    from huggingface_hub import HfApi

    class_mapping = parse_class_maps(args.class_map)
    repo_files = HfApi().list_repo_files(args.repo_id, repo_type="dataset")
    selected = select_files(
        repo_files=repo_files,
        class_mapping=class_mapping,
        per_class_cap=args.per_class_cap,
        seed=args.seed,
    )
    downloaded, warnings = download_selected(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        selected=selected,
        workers=args.workers,
    )
    summary = write_summary(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        class_mapping=class_mapping,
        selected=selected,
        downloaded=downloaded,
        warnings=warnings,
        args=args,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    require(not warnings, f"{len(warnings)} download(s) failed; see hf_source_summary.json")


if __name__ == "__main__":
    main()
