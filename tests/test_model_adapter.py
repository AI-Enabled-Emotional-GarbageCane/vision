from __future__ import annotations

from pathlib import Path

import numpy as np

from model_adapter import (
    DEFAULT_EXPORT_DIR,
    class_prediction_from_scores,
    preprocess_rgb,
    select_export_model,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    selected = select_export_model(DEFAULT_EXPORT_DIR)
    require(selected == DEFAULT_EXPORT_DIR / "best.onnx", "default export should prefer best.onnx")
    require(selected.exists(), "selected model export must exist")

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    tensor = preprocess_rgb(image)
    require(tensor.shape == (1, 3, 224, 224), "preprocess must produce NCHW tensor")
    require(tensor.dtype == np.float32, "preprocess must produce float32 tensor")

    prediction = class_prediction_from_scores(np.array([[0.1, 0.9]], dtype=np.float32))
    require(prediction.label == "reject", "highest class score should choose reject")
    require(abs(prediction.confidence - 0.9) < 1e-6, "probability confidence should be preserved")

    logits_prediction = class_prediction_from_scores(np.array([[4.0, 1.0]], dtype=np.float32))
    require(logits_prediction.label == "accept", "highest logit should choose accept")
    require(0 <= logits_prediction.confidence <= 1, "logit confidence should be normalized")

    print("[OK] model adapter selects export and normalizes predictions")


if __name__ == "__main__":
    main()
