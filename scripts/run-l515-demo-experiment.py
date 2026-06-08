#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

ACCEPT_DIRS = (
    "flexible_wrapper",
    "dirty_wrapper",
    "tissue_napkin",
    "cigarette_butt",
    "garbage_bag",
    "small_misc",
)
REJECT_DIRS = (
    "rigid_plastic_bottle",
    "drink_can",
    "bottle_cap",
    "paper_cardboard",
    "glass_metal",
)
ALL_DIRS = ACCEPT_DIRS + REJECT_DIRS

DATASET_SOURCES = (
    ("realwaste", "data/training/realwaste_yolo_cls_train_balanced"),
    ("taco", "data/training/taco_general_trash_hard_reject_yolo_cls"),
    ("tidy", "data/training/tidy_general_trash_yolo_cls"),
    ("l515", "data/training/l515_demo_yolo_cls"),
)

EVAL_SPECS = (
    {
        "name": "l515_holdout",
        "dataset_dir": "data/training/l515_demo_yolo_cls",
        "manifest": "data/training/l515_demo_yolo_cls/weak_dataset_manifest.csv",
        "split": "test",
    },
    {
        "name": "taco_full",
        "dataset_dir": "data/inference_general_trash_positive/taco_full_accept_focus",
        "manifest": "data/inference_general_trash_positive/taco_full_accept_focus/manifest.csv",
    },
    {
        "name": "taco_reject_safety",
        "dataset_dir": "data/inference_extra_waste/taco_reject_safety",
        "manifest": "data/inference_extra_waste/taco_reject_safety/manifest.csv",
    },
    {
        "name": "tidy_test",
        "dataset_dir": "data/training/tidy_general_trash_yolo_cls",
        "manifest": "data/training/tidy_general_trash_yolo_cls/weak_dataset_manifest.csv",
        "split": "test",
    },
    {
        "name": "realwaste_full_test",
        "dataset_dir": "data/training/realwaste_yolo_cls_full",
        "manifest": "data/training/realwaste_yolo_cls_full/weak_dataset_manifest.csv",
        "split": "test",
    },
    {
        "name": "taiwan_mapped",
        "dataset_dir": "data/inference_taiwan_waste",
        "manifest": "runs/realwaste-accuracy/realwaste-accuracy-002/weak_eval/taiwan_waste_mapped_manifest.csv",
    },
)

TRAIN_PACKAGES = (
    "ultralytics",
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "onnxslim",
)
EVAL_PACKAGES = ("onnxruntime", "pillow", "numpy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, train, and evaluate the L515 demo positive experiment.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/l515_demo_raw"))
    parser.add_argument("--l515-dataset-dir", type=Path, default=Path("data/training/l515_demo_yolo_cls"))
    parser.add_argument(
        "--combined-dataset-dir",
        type=Path,
        default=Path("data/training/l515_demo_combined_yolo_cls"),
    )
    parser.add_argument("--project", type=Path, default=Path("runs/l515-demo-positive"))
    parser.add_argument("--serial-prefix", default="l515-demo-positive")
    parser.add_argument("--min-accept", type=int, default=300)
    parser.add_argument("--min-reject", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="hardlink")
    parser.add_argument(
        "--runner",
        choices=("uv", "python"),
        default="uv",
        help="Dependency runner for training/evaluation commands.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create raw dataset folders and write the current report without building or training.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def class_name(slug: str) -> str:
    return slug.replace("_", " ")


def init_raw_dirs(raw_dir: Path) -> None:
    for slug in ALL_DIRS:
        (raw_dir / slug).mkdir(parents=True, exist_ok=True)
    readme = raw_dir / "README.md"
    if not readme.exists():
        readme.write_text(raw_readme(), encoding="utf-8")


def raw_readme() -> str:
    return """# L515 Demo Raw Dataset

Put AGX/L515 RGB snapshots into these class folders before running the experiment.

Accept folders:

- flexible_wrapper: candy wrappers, snack bags, soft plastic film, mixed-material wrappers
- dirty_wrapper: food-stained packaging and contaminated film
- tissue_napkin: tissues, napkins, paper towels
- cigarette_butt: cigarette butts
- garbage_bag: small trash bags and bag fragments
- small_misc: small non-recyclable miscellaneous trash

Reject folders:

- rigid_plastic_bottle: plastic bottles and rigid recyclable plastic containers
- drink_can: aluminum/metal drink cans
- bottle_cap: plastic or metal bottle caps
- paper_cardboard: clean paper and cardboard
- glass_metal: glass and metal recyclables

Minimum fine-tune gate: 300 accept images and 300 reject images. If the raw
dataset is smaller, the orchestration writes a blocked report and skips training.
"""


def image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for image_path in path.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def raw_counts(raw_dir: Path) -> dict[str, Any]:
    by_class = {slug: image_count(raw_dir / slug) for slug in ALL_DIRS}
    accept_count = sum(by_class[slug] for slug in ACCEPT_DIRS)
    reject_count = sum(by_class[slug] for slug in REJECT_DIRS)
    return {
        "raw_dir": str(raw_dir),
        "accept_count": accept_count,
        "reject_count": reject_count,
        "by_class": by_class,
    }


def has_both_labels(counts: dict[str, Any]) -> bool:
    return counts["accept_count"] > 0 and counts["reject_count"] > 0


def has_minimum_data(counts: dict[str, Any], min_accept: int, min_reject: int) -> bool:
    return counts["accept_count"] >= min_accept and counts["reject_count"] >= min_reject


def command_prefix(packages: tuple[str, ...], runner: str) -> list[str]:
    if runner == "python":
        return [sys.executable]
    command = ["uv", "run"]
    for package in packages:
        command.extend(["--with", package])
    command.append("python")
    return command


def run_command(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def python_script(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def ml_script(packages: tuple[str, ...], runner: str, script: str, *args: str) -> list[str]:
    return [*command_prefix(packages, runner), script, *args]


def build_l515_dataset(args: argparse.Namespace) -> None:
    raw_dir = resolve(args.raw_dir)
    output_dir = resolve(args.l515_dataset_dir)
    command = python_script(
        "scripts/build-folder-yolo-cls-dataset.py",
        "--source-dir",
        str(raw_dir),
        "--output-dir",
        str(output_dir),
        "--mapping-preset",
        "none",
        "--max-train-majority-ratio",
        "0",
        "--copy-mode",
        args.copy_mode,
    )
    for slug in ACCEPT_DIRS:
        command.extend(["--accept-class", class_name(slug)])
    for slug in REJECT_DIRS:
        command.extend(["--reject-class", class_name(slug)])
    run_command(command)


def require_source_datasets() -> list[str]:
    missing = []
    for _name, path in DATASET_SOURCES:
        if not resolve(Path(path)).exists():
            missing.append(path)
    return missing


def merge_combined_dataset(args: argparse.Namespace) -> None:
    command = python_script(
        "scripts/merge-yolo-cls-datasets.py",
        "--output-dir",
        str(resolve(args.combined_dataset_dir)),
        "--copy-mode",
        args.copy_mode,
    )
    for name, path in DATASET_SOURCES:
        command.extend(["--source", f"{name}:{resolve(Path(path))}"])
    run_command(command)


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


def train_candidate(
    args: argparse.Namespace,
    candidate: str,
    base_model: Path,
) -> dict[str, str]:
    project = resolve(args.project)
    project.mkdir(parents=True, exist_ok=True)
    run_name = next_serial_name(project, args.serial_prefix)
    run_dir = project / run_name
    command = ml_script(
        TRAIN_PACKAGES,
        args.runner,
        "scripts/train-yolo-cls.py",
        "--data",
        str(resolve(args.combined_dataset_dir)),
        "--model",
        str(resolve(base_model)),
        "--project",
        str(project),
        "--name",
        run_name,
        "--serial-prefix",
        args.serial_prefix,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--patience",
        str(args.patience),
        "--export-onnx",
    )
    run_command(command)
    return {
        "candidate": candidate,
        "base_model": str(resolve(base_model)),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "best_pt": str(run_dir / "weights" / "best.pt"),
        "best_onnx": str(run_dir / "weights" / "best.onnx"),
    }


def evaluate_model(args: argparse.Namespace, candidate: dict[str, str]) -> dict[str, Any]:
    run_dir = Path(candidate["run_dir"])
    model_path = Path(candidate["best_onnx"])
    eval_results: dict[str, Any] = {}
    for spec in EVAL_SPECS:
        dataset_dir = resolve(Path(spec["dataset_dir"]))
        manifest = resolve(Path(spec["manifest"]))
        if not dataset_dir.exists() or not manifest.exists():
            eval_results[spec["name"]] = {
                "status": "skipped",
                "missing_dataset_dir": not dataset_dir.exists(),
                "missing_manifest": not manifest.exists(),
            }
            continue
        output = run_dir / f"{spec['name']}_predictions.csv"
        summary = run_dir / f"{spec['name']}_summary.json"
        contact_sheet = run_dir / f"{spec['name']}_contact_sheet.jpg"
        command = ml_script(
            EVAL_PACKAGES,
            args.runner,
            "scripts/evaluate-weak-manifest.py",
            "--dataset-dir",
            str(dataset_dir),
            "--manifest",
            str(manifest),
            "--model",
            str(model_path),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--contact-sheet",
            str(contact_sheet),
        )
        if "split" in spec:
            command.extend(["--split", spec["split"]])
        run_command(command)
        sweep_csv = run_dir / f"{spec['name']}_threshold_sweep.csv"
        sweep_summary = run_dir / f"{spec['name']}_threshold_sweep_summary.json"
        run_command(
            python_script(
                "scripts/threshold-sweep.py",
                "--predictions",
                str(output),
                "--output",
                str(sweep_csv),
                "--summary",
                str(sweep_summary),
            )
        )
        eval_results[spec["name"]] = read_json(summary)
    return eval_results


def evaluate_existing_baselines(args: argparse.Namespace) -> list[dict[str, Any]]:
    baselines = (
        (
            "eval_only_realwaste_accuracy_002",
            Path("runs/realwaste-accuracy/realwaste-accuracy-002/weights/best.onnx"),
        ),
        (
            "eval_only_general_trash_positive_002",
            Path("runs/general-trash-positive/general-trash-positive-002/weights/best.onnx"),
        ),
        ("eval_only_public_export", Path("exports/20260601T122805Z/best.onnx")),
    )
    evaluated: list[dict[str, Any]] = []
    project = resolve(args.project)
    project.mkdir(parents=True, exist_ok=True)
    for candidate_name, model_path in baselines:
        resolved_model = resolve(model_path)
        if not resolved_model.exists():
            evaluated.append(
                {
                    "candidate": candidate_name,
                    "status": "skipped",
                    "reason": "missing baseline ONNX",
                    "best_onnx": str(resolved_model),
                }
            )
            continue
        run_name = next_serial_name(project, "l515-demo-eval")
        run_dir = project / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        candidate = {
            "candidate": candidate_name,
            "status": "evaluation_only",
            "run_name": run_name,
            "run_dir": str(run_dir),
            "best_onnx": str(resolved_model),
        }
        eval_results = evaluate_model(args, candidate)
        candidate["eval_results"] = eval_results
        candidate["acceptance"] = candidate_acceptance(eval_results)
        evaluated.append(candidate)
    return evaluated


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], section: str, key: str) -> float | None:
    value = summary.get(section, {}).get(key)
    return float(value) if value is not None else None


def candidate_acceptance(eval_results: dict[str, Any]) -> dict[str, Any]:
    l515 = eval_results.get("l515_holdout", {})
    taco = eval_results.get("taco_reject_safety", {})
    l515_recall = metric(l515, "accept_behavior", "gate_accept_recall")
    l515_false_accept = metric(l515, "reject_safety", "false_accept_rate_on_reject")
    taco_false_accept = metric(taco, "reject_safety", "false_accept_rate_on_reject")
    checks = {
        "l515_accept_recall_at_least_0_90": l515_recall is not None and l515_recall >= 0.90,
        "l515_reject_false_accept_at_most_0_10": (
            l515_false_accept is not None and l515_false_accept <= 0.10
        ),
        "taco_reject_safety_false_accept_at_most_0_10": (
            taco_false_accept is not None and taco_false_accept <= 0.10
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "l515_gate_accept_recall": l515_recall,
        "l515_reject_false_accept_rate": l515_false_accept,
        "taco_reject_safety_false_accept_rate": taco_false_accept,
    }


def choose_candidate(candidates: list[dict[str, Any]]) -> str:
    passing = [
        candidate
        for candidate in candidates
        if candidate.get("acceptance", {}).get("passed")
    ]
    if not passing:
        return ""
    passing.sort(
        key=lambda candidate: (
            candidate["acceptance"]["l515_reject_false_accept_rate"],
            candidate["acceptance"]["taco_reject_safety_false_accept_rate"],
        )
    )
    return passing[0]["run_name"]


def write_markdown_report(report_path: Path, state: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counts = state["raw_counts"]
    lines = [
        "# L515 Demo Positive Experiment Report",
        "",
        f"Status: `{state['status']}`",
        "",
        "## Raw Data",
        "",
        f"- Raw dir: `{counts['raw_dir']}`",
        f"- Accept images: `{counts['accept_count']}`",
        f"- Reject images: `{counts['reject_count']}`",
        "",
        "## Class Counts",
        "",
        "| class | label | count |",
        "| --- | --- | ---: |",
    ]
    for slug, count in counts["by_class"].items():
        label = "accept" if slug in ACCEPT_DIRS else "reject"
        lines.append(f"| `{slug}` | {label} | {count} |")
    lines.extend(["", "## Result", ""])
    if state["status"] != "completed":
        lines.extend(
            [
                f"- Block reason: {state.get('block_reason', 'n/a')}",
                f"- Minimum required: accept `{state['min_accept']}`, reject `{state['min_reject']}`",
                "- Training was skipped by design.",
            ]
        )
        if state.get("candidates"):
            lines.append("")
            lines.append("## Evaluation-Only Results")
            lines.append("")
            lines.append(
                "| candidate | status | L515 recall | L515 reject false accept | TACO safety false accept |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: |")
            for candidate in state["candidates"]:
                acceptance = candidate.get("acceptance", {})
                lines.append(
                    "| `{}` | {} | {} | {} | {} |".format(
                        candidate.get("candidate", candidate.get("run_name", "unknown")),
                        candidate.get("status", "unknown"),
                        format_percent(acceptance.get("l515_gate_accept_recall")),
                        format_percent(acceptance.get("l515_reject_false_accept_rate")),
                        format_percent(acceptance.get("taco_reject_safety_false_accept_rate")),
                    )
                )
    else:
        selected = state.get("selected_candidate") or "none"
        lines.append(f"- Selected review candidate: `{selected}`")
        lines.append("")
        lines.append("| candidate | passed | L515 recall | L515 reject false accept | TACO safety false accept |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for candidate in state["candidates"]:
            acceptance = candidate.get("acceptance")
            if not acceptance:
                lines.append(
                    "| `{}` | skipped | n/a | n/a | n/a |".format(
                        candidate.get("candidate", candidate.get("run_name", "unknown"))
                    )
                )
                continue
            lines.append(
                "| `{}` | {} | {} | {} | {} |".format(
                    candidate["run_name"],
                    "yes" if acceptance["passed"] else "no",
                    format_percent(acceptance["l515_gate_accept_recall"]),
                    format_percent(acceptance["l515_reject_false_accept_rate"]),
                    format_percent(acceptance["taco_reject_safety_false_accept_rate"]),
                )
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This experiment does not modify the v0.3 queue contract.",
            "- This experiment does not replace the recommended export or AGX default model.",
            "- Soft plastic wrappers, candy wrappers, snack bags, dirty film, and mixed-material wrappers are labeled accept for this experiment.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def write_json_report(report_path: Path, state: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def count_manifest_labels(manifest: Path) -> dict[str, int]:
    if not manifest.exists():
        return {}
    counts: Counter[str] = Counter()
    with manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row.get("eval_label", "")] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    raw_dir = resolve(args.raw_dir)
    project = resolve(args.project)
    init_raw_dirs(raw_dir)
    counts = raw_counts(raw_dir)
    state: dict[str, Any] = {
        "status": "initialized",
        "raw_counts": counts,
        "min_accept": args.min_accept,
        "min_reject": args.min_reject,
        "l515_dataset_dir": str(resolve(args.l515_dataset_dir)),
        "combined_dataset_dir": str(resolve(args.combined_dataset_dir)),
        "project": str(project),
        "candidates": [],
    }
    json_report = project / "l515_demo_fix_report.json"
    markdown_report = project / "l515_demo_fix_report.md"

    if args.init_only:
        state["status"] = "initialized"
        state["block_reason"] = "init-only requested"
        write_json_report(json_report, state)
        write_markdown_report(markdown_report, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    if not has_both_labels(counts):
        state["status"] = "blocked"
        state["block_reason"] = "raw dataset needs at least one accept and one reject image"
        write_json_report(json_report, state)
        write_markdown_report(markdown_report, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    build_l515_dataset(args)
    state["l515_dataset_manifest_counts"] = count_manifest_labels(
        resolve(args.l515_dataset_dir) / "weak_dataset_manifest.csv"
    )

    if not has_minimum_data(counts, args.min_accept, args.min_reject):
        state["status"] = "blocked"
        state["block_reason"] = "raw dataset below minimum fine-tune gate"
        state["candidates"] = evaluate_existing_baselines(args)
        write_json_report(json_report, state)
        write_markdown_report(markdown_report, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    missing_sources = require_source_datasets()
    if missing_sources:
        state["status"] = "blocked"
        state["block_reason"] = "missing source dataset(s): " + ", ".join(missing_sources)
        write_json_report(json_report, state)
        write_markdown_report(markdown_report, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    merge_combined_dataset(args)
    candidates = [
        ("candidate_a_safe_baseline", Path("runs/realwaste-accuracy/realwaste-accuracy-002/weights/best.pt")),
        (
            "candidate_b_demo_recall",
            Path("runs/general-trash-positive/general-trash-positive-002/weights/best.pt"),
        ),
    ]
    for candidate_name, base_model in candidates:
        if not resolve(base_model).exists():
            state["candidates"].append(
                {
                    "candidate": candidate_name,
                    "base_model": str(resolve(base_model)),
                    "status": "skipped",
                    "reason": "missing base model",
                }
            )
            continue
        candidate = train_candidate(args, candidate_name, base_model)
        eval_results = evaluate_model(args, candidate)
        candidate["status"] = "evaluated"
        candidate["eval_results"] = eval_results
        candidate["acceptance"] = candidate_acceptance(eval_results)
        state["candidates"].append(candidate)

    state["selected_candidate"] = choose_candidate(state["candidates"])
    state["status"] = "completed"
    write_json_report(json_report, state)
    write_markdown_report(markdown_report, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
