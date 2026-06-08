#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO classification locally.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("exports/20260601T122805Z/best.pt"))
    parser.add_argument("--project", type=Path, default=Path("runs/reject-safety"))
    parser.add_argument("--name", default="")
    parser.add_argument(
        "--serial-prefix",
        default="run",
        help="Run-name prefix used when --name is omitted. The script picks prefix-001, prefix-002, ...",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def next_serial_name(project: Path, prefix: str) -> str:
    escaped_prefix = re.escape(prefix)
    pattern = re.compile(rf"^{escaped_prefix}-(\d+)$")
    used_numbers: set[int] = set()
    if project.exists():
        for child in project.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                used_numbers.add(int(match.group(1)))

    number = 1
    while number in used_numbers:
        number += 1
    return f"{prefix}-{number:03d}"


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.model = args.model.resolve()
    args.project = args.project.resolve()
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Run with: "
            "uv run --with ultralytics --with torch --with torchvision "
            "--with onnx --with onnxruntime --with onnxslim "
            "python scripts/train-yolo-cls.py ..."
        ) from exc

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("[FAIL] CUDA is not available; use --device cpu or install CUDA torch.")

    args.project.mkdir(parents=True, exist_ok=True)
    run_name = args.name or next_serial_name(args.project, args.serial_prefix)
    model = YOLO(str(args.model))
    results = model.train(
        data=str(args.data),
        task="classify",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        project=str(args.project),
        name=run_name,
        exist_ok=False,
    )

    save_dir = Path(getattr(results, "save_dir", args.project / run_name))
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    export_path = ""
    if args.export_onnx:
        export_model = YOLO(str(best_pt if best_pt.exists() else last_pt))
        export_path = str(export_model.export(format="onnx", imgsz=args.imgsz))

    summary = {
        "data": str(args.data),
        "base_model": str(args.model),
        "save_dir": str(save_dir),
        "run_name": run_name,
        "serial_prefix": args.serial_prefix,
        "best_pt": str(best_pt),
        "last_pt": str(last_pt),
        "export_onnx": export_path,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    (save_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
