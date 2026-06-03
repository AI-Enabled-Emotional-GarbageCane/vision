from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = ROOT / "exports" / "20260601T122805Z"
DEFAULT_CLASS_NAMES = ("accept", "reject")


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    confidence: float


class ImageClassifier(Protocol):
    def predict(self, image_rgb: np.ndarray) -> ModelPrediction: ...


def select_export_model(export_dir: Path = DEFAULT_EXPORT_DIR) -> Path:
    onnx_path = export_dir / "best.onnx"
    if onnx_path.exists():
        return onnx_path

    pt_path = export_dir / "best.pt"
    if pt_path.exists():
        return pt_path

    raise FileNotFoundError(f"missing model export in {export_dir}")


def preprocess_rgb(image_rgb: np.ndarray, image_size: int = 224) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")

    pil_image = Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB")
    resized = pil_image.resize((image_size, image_size))
    tensor = np.asarray(resized, dtype=np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))[None, ...]


def class_prediction_from_scores(
    scores: np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> ModelPrediction:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size < len(class_names):
        raise ValueError("model output has fewer scores than class names")

    values = values[: len(class_names)]
    if np.all(values >= 0) and abs(float(np.sum(values)) - 1.0) < 1e-3:
        probabilities = values
    else:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values)

    index = int(np.argmax(probabilities))
    return ModelPrediction(
        label=str(class_names[index]),
        confidence=float(probabilities[index]),
    )


class OnnxYoloClassifier:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_EXPORT_DIR / "best.onnx",
        *,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        image_size: int = 224,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"missing ONNX model: {self.model_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required to run ONNX vision inference") from exc

        self._class_names = tuple(class_names)
        self._image_size = image_size
        self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, image_rgb: np.ndarray) -> ModelPrediction:
        tensor = preprocess_rgb(image_rgb, image_size=self._image_size)
        outputs = self._session.run(None, {self._input_name: tensor})
        return class_prediction_from_scores(outputs[0], self._class_names)


class UltralyticsYoloClassifier:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_EXPORT_DIR / "best.pt",
        *,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"missing PyTorch model: {self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required to run PyTorch vision inference") from exc

        self._class_names = tuple(class_names)
        self._model = YOLO(str(self.model_path))

    def predict(self, image_rgb: np.ndarray) -> ModelPrediction:
        results = self._model.predict(source=np.asarray(image_rgb), verbose=False)
        if not results:
            raise RuntimeError("model returned no prediction")

        probabilities = results[0].probs
        if probabilities is None:
            raise RuntimeError("classification model did not return probabilities")

        label_index = int(probabilities.top1)
        confidence = float(probabilities.top1conf)
        return ModelPrediction(label=self._class_names[label_index], confidence=confidence)


class StaticClassifier:
    def __init__(self, prediction: ModelPrediction) -> None:
        self.prediction = prediction

    def predict(self, image_rgb: np.ndarray) -> ModelPrediction:
        return self.prediction


def create_default_classifier(model_path: Path | str | None = None) -> ImageClassifier:
    selected = Path(model_path) if model_path is not None else select_export_model()
    if selected.suffix == ".onnx":
        return OnnxYoloClassifier(selected)
    if selected.suffix == ".pt":
        return UltralyticsYoloClassifier(selected)
    raise ValueError(f"unsupported model export type: {selected}")
