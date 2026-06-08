#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_MODEL = Path("runs/user-accept-seed-finetune/user-accept-seed-001/weights/best.onnx")
DEFAULT_CANDIDATES_DIR = Path("demo_candidates/accept_props")
DEFAULT_RUN_ROOT = Path("runs/demo-accept-recall")
DEFAULT_SERIAL_PREFIX = "demo-accept-recall"
DEFAULT_ACCEPT_THRESHOLD = 0.50
DEFAULT_UNCERTAIN_THRESHOLD = 0.50
DEFAULT_SHOTS_REQUIRED = 3
DEFAULT_MIN_ACCEPTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate demo accept-only candidate props with a demo-only low accept gate.",
    )
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--name", default="")
    parser.add_argument("--serial-prefix", default=DEFAULT_SERIAL_PREFIX)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--accept-threshold", type=float, default=DEFAULT_ACCEPT_THRESHOLD)
    parser.add_argument("--uncertain-threshold", type=float, default=DEFAULT_UNCERTAIN_THRESHOLD)
    parser.add_argument("--shots-required", type=int, default=DEFAULT_SHOTS_REQUIRED)
    parser.add_argument("--min-accepts", type=int, default=DEFAULT_MIN_ACCEPTS)
    parser.add_argument("--min-props", type=int, default=10)
    parser.add_argument("--min-total-images", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create the demo run config and candidate folder, but do not run inference.",
    )
    parser.add_argument(
        "--enforce-smoke",
        action="store_true",
        help="Exit non-zero when the candidate set does not meet the demo smoke criteria.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[FAIL] {message}")


def validate_thresholds(accept_threshold: float, uncertain_threshold: float) -> None:
    require(0 <= accept_threshold <= 1, f"accept-threshold out of range: {accept_threshold}")
    require(0 <= uncertain_threshold <= 1, f"uncertain-threshold out of range: {uncertain_threshold}")
    require(
        uncertain_threshold <= accept_threshold,
        "uncertain-threshold must be less than or equal to accept-threshold",
    )


def next_serial_name(project: Path, prefix: str) -> str:
    escaped_prefix = re.escape(prefix)
    pattern = re.compile(rf"^{escaped_prefix}-(\d+)$")
    used_numbers: set[int] = set()
    if project.exists():
        for child in project.iterdir():
            if child.is_dir():
                match = pattern.match(child.name)
                if match:
                    used_numbers.add(int(match.group(1)))

    number = 1
    while number in used_numbers:
        number += 1
    return f"{prefix}-{number:03d}"


def resolve_run_dir(run_root: Path, name: str, serial_prefix: str) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    run_name = name or next_serial_name(run_root, serial_prefix)
    return run_root / run_name


def scan_candidate_images(candidates_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not candidates_dir.exists():
        return rows

    for item_dir in sorted(path for path in candidates_dir.iterdir() if path.is_dir()):
        item_name = item_dir.name
        images = sorted(
            path
            for path in item_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in images:
            rows.append(
                {
                    "local_path": str(image_path.relative_to(candidates_dir)),
                    "item_name": item_name,
                    "eval_label": "accept",
                    "eval_label_strength": "demo_user_confirmed_accept",
                    "label_hint": f"demo_accept_candidate:{item_name}",
                    "capture_source": "demo_rgb",
                    "split": "test",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({field for row in rows for field in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv(
        path,
        rows,
        [
            "local_path",
            "item_name",
            "eval_label",
            "eval_label_strength",
            "label_hint",
            "capture_source",
            "split",
        ],
    )


def write_demo_config(
    path: Path,
    *,
    model: Path,
    candidates_dir: Path,
    accept_threshold: float,
    uncertain_threshold: float,
    shots_required: int,
    min_accepts: int,
    min_props: int,
    min_total_images: int,
    imgsz: int,
) -> None:
    config = {
        "purpose": "demo_accept_recall_only",
        "production_ready": False,
        "model": str(model),
        "accept_threshold": accept_threshold,
        "uncertain_threshold": uncertain_threshold,
        "candidates_dir": str(candidates_dir),
        "candidate_policy": {
            "eval_label": "accept",
            "shots_required_per_prop": shots_required,
            "min_accepts_per_prop": min_accepts,
            "min_props_for_smoke": min_props,
            "min_total_images_for_smoke": min_total_images,
            "pass_rule": f"at least {min_accepts}/{shots_required} shots gate to accept",
        },
        "imgsz": imgsz,
        "contract_note": "Demo profile only. Does not modify recognition_result payload or production accept gate defaults.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def run_evaluator(
    *,
    candidates_dir: Path,
    manifest: Path,
    model: Path,
    predictions: Path,
    summary: Path,
    contact_sheet: Path,
    accept_threshold: float,
    uncertain_threshold: float,
    imgsz: int,
) -> None:
    cmd = [
        sys.executable,
        "scripts/evaluate-weak-manifest.py",
        "--dataset-dir",
        str(candidates_dir),
        "--manifest",
        str(manifest),
        "--model",
        str(model),
        "--output",
        str(predictions),
        "--summary",
        str(summary),
        "--contact-sheet",
        str(contact_sheet),
        "--accept-threshold",
        str(accept_threshold),
        "--uncertain-threshold",
        str(uncertain_threshold),
        "--imgsz",
        str(imgsz),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


def item_from_local_path(local_path: str) -> str:
    parts = Path(local_path).parts
    return parts[0] if parts else ""


def summarize_props(
    prediction_rows: list[dict[str, str]],
    *,
    shots_required: int,
    min_accepts: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[item_from_local_path(row["local_path"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for item_name in sorted(grouped):
        rows = grouped[item_name]
        gate_counts = Counter(row["gate_action"] for row in rows)
        accept_count = gate_counts.get("accept", 0)
        image_count = len(rows)
        enough_shots = image_count >= shots_required
        accepted = enough_shots and accept_count >= min_accepts
        reason = "pass"
        if not enough_shots:
            reason = f"needs_{shots_required}_shots"
        elif accept_count < min_accepts:
            reason = f"needs_{min_accepts}_accepts"

        summary_rows.append(
            {
                "item_name": item_name,
                "image_count": image_count,
                "accept_count": accept_count,
                "reject_count": gate_counts.get("reject", 0),
                "uncertain_count": gate_counts.get("uncertain", 0),
                "pass_demo": str(accepted),
                "reason": reason,
            }
        )
    return summary_rows


def write_accepted_props(path: Path, prop_rows: list[dict[str, Any]]) -> None:
    accepted = [row["item_name"] for row in prop_rows if row["pass_demo"] == "True"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(accepted) + ("\n" if accepted else ""), encoding="utf-8")


def write_demo_summary(
    path: Path,
    *,
    prop_rows: list[dict[str, Any]],
    image_count: int,
    min_props: int,
    min_total_images: int,
) -> dict[str, Any]:
    prop_count = len(prop_rows)
    accepted_props = [row for row in prop_rows if row["pass_demo"] == "True"]
    accepted_prop_count = len(accepted_props)
    accepted_prop_rate = accepted_prop_count / prop_count if prop_count else None
    smoke_pass = (
        prop_count >= min_props
        and image_count >= min_total_images
        and accepted_prop_rate is not None
        and accepted_prop_rate >= 0.90
    )
    summary = {
        "purpose": "demo_accept_recall_only",
        "prop_count": prop_count,
        "image_count": image_count,
        "accepted_prop_count": accepted_prop_count,
        "accepted_prop_rate": accepted_prop_rate,
        "min_props": min_props,
        "min_total_images": min_total_images,
        "smoke_pass": smoke_pass,
        "note": "Demo-only accept candidate screening. Passing props are suitable for accept-only demo flow, not production safety claims.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    validate_thresholds(args.accept_threshold, args.uncertain_threshold)
    require(args.shots_required >= 1, "shots-required must be >= 1")
    require(args.min_accepts >= 1, "min-accepts must be >= 1")
    require(
        args.min_accepts <= args.shots_required,
        "min-accepts must be less than or equal to shots-required",
    )

    run_dir = resolve_run_dir(args.run_root, args.name, args.serial_prefix)
    run_dir.mkdir(parents=True, exist_ok=False)
    args.candidates_dir.mkdir(parents=True, exist_ok=True)

    write_demo_config(
        run_dir / "demo_config.json",
        model=args.model,
        candidates_dir=args.candidates_dir,
        accept_threshold=args.accept_threshold,
        uncertain_threshold=args.uncertain_threshold,
        shots_required=args.shots_required,
        min_accepts=args.min_accepts,
        min_props=args.min_props,
        min_total_images=args.min_total_images,
        imgsz=args.imgsz,
    )

    if args.init_only:
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "demo_config": str(run_dir / "demo_config.json"),
                    "candidates_dir": str(args.candidates_dir),
                    "init_only": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    require(args.model.exists(), f"missing demo model: {args.model}")
    rows = scan_candidate_images(args.candidates_dir)
    require(rows, f"no candidate images found under: {args.candidates_dir}")

    manifest = run_dir / "demo_candidates_manifest.csv"
    predictions = run_dir / "demo_predictions.csv"
    eval_summary = run_dir / "demo_eval_summary.json"
    contact_sheet = run_dir / "demo_contact_sheet.jpg"
    prop_summary_csv = run_dir / "demo_prop_summary.csv"
    accepted_props = run_dir / "demo_accepted_props.txt"
    smoke_summary = run_dir / "demo_smoke_summary.json"

    write_manifest(manifest, rows)
    run_evaluator(
        candidates_dir=args.candidates_dir,
        manifest=manifest,
        model=args.model,
        predictions=predictions,
        summary=eval_summary,
        contact_sheet=contact_sheet,
        accept_threshold=args.accept_threshold,
        uncertain_threshold=args.uncertain_threshold,
        imgsz=args.imgsz,
    )

    prop_rows = summarize_props(
        read_csv(predictions),
        shots_required=args.shots_required,
        min_accepts=args.min_accepts,
    )
    write_csv(
        prop_summary_csv,
        prop_rows,
        [
            "item_name",
            "image_count",
            "accept_count",
            "reject_count",
            "uncertain_count",
            "pass_demo",
            "reason",
        ],
    )
    write_accepted_props(accepted_props, prop_rows)
    summary = write_demo_summary(
        smoke_summary,
        prop_rows=prop_rows,
        image_count=len(rows),
        min_props=args.min_props,
        min_total_images=args.min_total_images,
    )
    summary["run_dir"] = str(run_dir)
    summary["accepted_props"] = str(accepted_props)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.enforce_smoke and not summary["smoke_pass"]:
        raise SystemExit("[FAIL] demo candidate smoke criteria were not met")


if __name__ == "__main__":
    main()
