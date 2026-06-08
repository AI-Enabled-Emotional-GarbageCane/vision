#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX classification over a weak-label manifest.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory that local_path values are relative to.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("exports/20260601T122805Z/best.onnx"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        help="Evaluate only rows with this split value when the manifest includes split.",
    )
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--accept-threshold", type=float, default=0.76)
    parser.add_argument("--uncertain-threshold", type=float, default=0.50)
    parser.add_argument(
        "--contact-sheet-max-images",
        type=int,
        default=120,
        help="Maximum reviewed rows to render in the contact sheet. Predictions and summary remain full-size.",
    )
    return parser.parse_args()


def require_ml_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "Missing ML/image dependencies. Run with: "
            "uv run --with onnxruntime --with pillow --with numpy "
            "python scripts/evaluate-weak-manifest.py ..."
        ) from exc
    return np, ort, Image, (ImageDraw, ImageOps)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preprocess_image(path: Path, image_module: Any, image_ops: Any, np: Any, imgsz: int) -> Any:
    image = image_module.open(path)
    if image.mode in {"RGBA", "LA"}:
        background = image_module.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")
    image = image_ops.fit(
        image,
        (imgsz, imgsz),
        method=image_module.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def normalize_output(raw_output: Any, class_count: int, np: Any) -> Any:
    vector = np.asarray(raw_output).reshape(-1).astype(np.float64)
    if vector.size != class_count:
        vector = vector[:class_count]
    if (
        np.all(vector >= -1e-6)
        and np.all(vector <= 1 + 1e-6)
        and abs(float(vector.sum()) - 1.0) < 1e-3
    ):
        return vector
    vector = vector - np.max(vector)
    exp = np.exp(vector)
    return exp / exp.sum()


def gate_action(
    pred_class: str,
    confidence: float,
    accept_threshold: float,
    uncertain_threshold: float,
) -> str:
    if confidence < uncertain_threshold:
        return "uncertain"
    if pred_class == "accept" and confidence >= accept_threshold:
        return "accept"
    return "reject"


def run_predictions(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    np, ort, image_module, draw_modules = require_ml_dependencies()
    _image_draw, image_ops = draw_modules

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    metadata = session.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(metadata.get("names", "{0: 'accept', 1: 'reject'}"))

    rows: list[dict[str, str]] = []
    for manifest_row in read_manifest(args.manifest):
        if args.split and manifest_row.get("split") != args.split:
            continue
        eval_label = manifest_row.get("eval_label", "").strip()
        if eval_label == "ignore":
            continue
        local_path = manifest_row["local_path"]
        image_path = args.dataset_dir / local_path
        probabilities = normalize_output(
            session.run(
                None,
                {
                    input_name: preprocess_image(
                        image_path,
                        image_module,
                        image_ops,
                        np,
                        args.imgsz,
                    )
                },
            )[0],
            class_count=len(names),
            np=np,
        )
        pred_index = int(np.argmax(probabilities))
        pred_class = names[pred_index]
        confidence = float(probabilities[pred_index])
        action = gate_action(
            pred_class=pred_class,
            confidence=confidence,
            accept_threshold=args.accept_threshold,
            uncertain_threshold=args.uncertain_threshold,
        )
        weak_match = bool(eval_label and eval_label == pred_class)
        rows.append(
            {
                "local_path": local_path,
                "split": manifest_row.get("split", ""),
                "label_hint": manifest_row.get("label_hint", ""),
                "eval_label": eval_label,
                "eval_label_strength": manifest_row.get("eval_label_strength", ""),
                "dominant_category": manifest_row.get(
                    "dominant_taco_category",
                    manifest_row.get("title_hint", manifest_row.get("label_hint", "")),
                ),
                "all_categories": manifest_row.get("all_taco_categories", ""),
                "pred_class": pred_class,
                "confidence": f"{confidence:.6f}",
                "accept_prob": f"{float(probabilities[0]):.6f}",
                "reject_prob": f"{float(probabilities[1]):.6f}",
                "gate_action": action,
                "weak_label_match": str(weak_match),
            }
        )

    summary = summarize_predictions(
        rows,
        args.model,
        accept_threshold=args.accept_threshold,
        uncertain_threshold=args.uncertain_threshold,
    )
    return rows, summary


def summarize_predictions(
    rows: list[dict[str, str]],
    model_path: Path,
    accept_threshold: float,
    uncertain_threshold: float,
) -> dict[str, Any]:
    by_eval_label: dict[str, dict[str, int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    gate_confusion: dict[str, dict[str, int]] = {}
    mistake_counter: Counter[str] = Counter()

    for row in rows:
        eval_label = row["eval_label"] or "unlabeled"
        pred_class = row["pred_class"]
        by_eval_label.setdefault(eval_label, {"count": 0, "accept": 0, "reject": 0, "match": 0})
        by_eval_label[eval_label]["count"] += 1
        by_eval_label[eval_label][pred_class] += 1
        if eval_label == pred_class:
            by_eval_label[eval_label]["match"] += 1
        else:
            mistake_counter[f"{eval_label}:{row['dominant_category']}->{pred_class}"] += 1

        confusion.setdefault(eval_label, {})
        confusion[eval_label][pred_class] = confusion[eval_label].get(pred_class, 0) + 1
        gate_confusion.setdefault(eval_label, {})
        gate_action_value = row["gate_action"]
        gate_confusion[eval_label][gate_action_value] = (
            gate_confusion[eval_label].get(gate_action_value, 0) + 1
        )

    total = len(rows)
    match = sum(row["weak_label_match"] == "True" for row in rows)
    reject_gate = gate_confusion.get("reject", {})
    reject_total = sum(reject_gate.values())
    accept_gate = gate_confusion.get("accept", {})
    accept_total = sum(accept_gate.values())
    return {
        "model": str(model_path),
        "count": total,
        "accept_threshold": accept_threshold,
        "uncertain_threshold": uncertain_threshold,
        "weak_label_agreement": match / total if total else None,
        "by_eval_label": by_eval_label,
        "confusion": confusion,
        "gate_confusion": gate_confusion,
        "reject_safety": {
            "reject_count": reject_total,
            "false_accept_count_on_reject": reject_gate.get("accept", 0),
            "false_accept_rate_on_reject": (
                reject_gate.get("accept", 0) / reject_total if reject_total else None
            ),
            "blocked_reject_rate": (
                (reject_gate.get("reject", 0) + reject_gate.get("uncertain", 0)) / reject_total
                if reject_total
                else None
            ),
        },
        "accept_behavior": {
            "accept_count": accept_total,
            "accepted_accept_count": accept_gate.get("accept", 0),
            "gate_accept_recall": (
                accept_gate.get("accept", 0) / accept_total if accept_total else None
            ),
        },
        "top_mistakes": dict(mistake_counter.most_common(20)),
        "note": "Weak-label inference only; eval_label is derived from source mapping and needs human review.",
    }


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheet(
    path: Path,
    rows: list[dict[str, str]],
    dataset_dir: Path,
    imgsz: tuple[int, int] = (170, 125),
    columns: int = 5,
    max_images: int = 120,
) -> None:
    _np, _ort, image_module, draw_modules = require_ml_dependencies()
    image_draw, image_ops = draw_modules
    if max_images <= 0:
        return
    review_rows = select_contact_sheet_rows(rows, max_images)
    if not review_rows:
        return
    thumb_width, thumb_height = imgsz
    label_height = 72
    row_count = (len(review_rows) + columns - 1) // columns
    sheet = image_module.new(
        "RGB",
        (columns * thumb_width, row_count * (thumb_height + label_height)),
        "white",
    )
    draw = image_draw.Draw(sheet)

    for index, row in enumerate(review_rows):
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        image_path = dataset_dir / row["local_path"]
        try:
            image = image_module.open(image_path).convert("RGB")
            image = image_ops.contain(
                image,
                (thumb_width, thumb_height),
                image_module.Resampling.BILINEAR,
            )
            sheet.paste(
                image,
                (x + (thumb_width - image.width) // 2, y + (thumb_height - image.height) // 2),
            )
        except Exception:  # noqa: BLE001 - show broken image box in review sheet.
            draw.rectangle([x, y, x + thumb_width - 1, y + thumb_height - 1], outline="red")

        status = "OK" if row["weak_label_match"] == "True" else "MISS"
        label = (
            f"{status} {row['eval_label']}=>{row['pred_class']} {row['confidence']}\n"
            f"gate={row['gate_action']}\n"
            f"{row['dominant_category'][:28]}\n"
            f"{Path(row['local_path']).name[:24]}"
        )
        draw.text((x + 4, y + thumb_height + 3), label, fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def select_contact_sheet_rows(rows: list[dict[str, str]], max_images: int) -> list[dict[str, str]]:
    if len(rows) <= max_images:
        return rows
    misses = [row for row in rows if row["weak_label_match"] != "True"]
    false_accepts = [
        row
        for row in rows
        if row["eval_label"] == "reject" and row["gate_action"] == "accept"
    ]
    false_rejects = [
        row
        for row in rows
        if row["eval_label"] == "accept" and row["gate_action"] != "accept"
    ]
    correct_accepts = [
        row
        for row in rows
        if row["eval_label"] == "accept" and row["gate_action"] == "accept"
    ]
    correct_rejects = [
        row
        for row in rows
        if row["eval_label"] == "reject" and row["gate_action"] != "accept"
    ]

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in (false_accepts, false_rejects, misses, correct_accepts, correct_rejects, rows):
        for row in group:
            key = row["local_path"]
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= max_images:
                return selected
    return selected


def main() -> None:
    args = parse_args()
    rows, summary = run_predictions(args)
    write_predictions(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.contact_sheet:
        write_contact_sheet(
            args.contact_sheet,
            rows,
            args.dataset_dir,
            max_images=args.contact_sheet_max_images,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
