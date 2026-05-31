from __future__ import annotations

from vision_contract import CLASS_VALUES, RECOGNITION_RESULT_FIELDS, run_stub_inference


def main() -> None:
    result = run_stub_inference()

    assert set(result) == RECOGNITION_RESULT_FIELDS
    assert result["event"] == "recognition_result"
    assert result["class"] in CLASS_VALUES
    assert 0 <= result["confidence"] <= 1
    assert result["num_objects"] == 1
    assert result["snapshot_path"]
    assert result["ts"]

    print("[OK] stub inference produced a valid recognition_result payload")


if __name__ == "__main__":
    main()
