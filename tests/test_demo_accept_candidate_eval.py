from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-demo-accept-candidate-eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_demo_accept_candidate_eval", SCRIPT)
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
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidates = root / "demo_candidates" / "accept_props"
        for item_name, image_count in {
            "napkin": 3,
            "wrapper": 3,
            "tiny_bag": 2,
        }.items():
            item_dir = candidates / item_name
            item_dir.mkdir(parents=True)
            for index in range(image_count):
                suffix = ".png" if index == 0 else ".jpg"
                (item_dir / f"shot_{index + 1:03d}{suffix}").write_bytes(b"fake image")
            (item_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

        rows = module.scan_candidate_images(candidates)
        require(len(rows) == 8, "scanner should include images and ignore non-images")
        require(
            {row["eval_label"] for row in rows} == {"accept"},
            "all demo candidate rows should be accept",
        )
        require(
            all(row["capture_source"] == "demo_rgb" for row in rows),
            "candidate manifest should preserve demo capture source",
        )

        prediction_rows = [
            {"local_path": "napkin/shot_001.png", "gate_action": "accept"},
            {"local_path": "napkin/shot_002.jpg", "gate_action": "accept"},
            {"local_path": "napkin/shot_003.jpg", "gate_action": "reject"},
            {"local_path": "wrapper/shot_001.png", "gate_action": "accept"},
            {"local_path": "wrapper/shot_002.jpg", "gate_action": "reject"},
            {"local_path": "wrapper/shot_003.jpg", "gate_action": "reject"},
            {"local_path": "tiny_bag/shot_001.png", "gate_action": "accept"},
            {"local_path": "tiny_bag/shot_002.jpg", "gate_action": "accept"},
        ]
        prop_rows = module.summarize_props(
            prediction_rows,
            shots_required=3,
            min_accepts=2,
        )
        by_item = {row["item_name"]: row for row in prop_rows}
        require(by_item["napkin"]["pass_demo"] == "True", "2/3 accepted shots should pass")
        require(
            by_item["wrapper"]["reason"] == "needs_2_accepts",
            "1/3 accepted shots should fail",
        )
        require(
            by_item["tiny_bag"]["reason"] == "needs_3_shots",
            "props with too few shots should fail even if accepted",
        )

        accepted_path = root / "accepted.txt"
        module.write_accepted_props(accepted_path, prop_rows)
        require(
            accepted_path.read_text(encoding="utf-8").splitlines() == ["napkin"],
            "accepted props list should include only passing props",
        )

        summary = module.write_demo_summary(
            root / "summary.json",
            prop_rows=prop_rows,
            image_count=8,
            min_props=3,
            min_total_images=8,
        )
        require(not summary["smoke_pass"], "smoke should fail when accepted prop rate is below 90%")
        require(
            json.loads((root / "summary.json").read_text(encoding="utf-8"))["prop_count"] == 3,
            "demo summary JSON should be written",
        )

        run_root = root / "runs"
        (run_root / "demo-accept-recall-001").mkdir(parents=True)
        require(
            module.next_serial_name(run_root, "demo-accept-recall") == "demo-accept-recall-002",
            "demo serial names should not overwrite existing runs",
        )

    print("[OK] demo accept candidate evaluator scans props and applies 2/3 pass rule")


if __name__ == "__main__":
    main()
