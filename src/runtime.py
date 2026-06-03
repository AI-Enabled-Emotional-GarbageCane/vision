from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from model_adapter import ImageClassifier, create_default_classifier
from vision_contract import build_recognition_result


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_DIR = ROOT / "snapshots"


class QueueLike(Protocol):
    def get(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def put(self, item: dict[str, Any]) -> None: ...


class RGBFrameSource(Protocol):
    def capture_rgb_frame(self) -> np.ndarray: ...


@dataclass(frozen=True)
class VisionRuntimeConfig:
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR
    input_timeout_sec: float = 0.25
    max_events: int | None = None


class L515ColorCamera:
    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        timeout_ms: int = 3000,
        warmup_frames: int = 30,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_ms = timeout_ms
        self.warmup_frames = warmup_frames
        self._pipeline = None

    def _ensure_started(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required to capture L515 RGB frames") from exc

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        pipeline.start(config)
        try:
            for _ in range(max(0, int(self.warmup_frames))):
                pipeline.wait_for_frames(self.timeout_ms)
        except RuntimeError:
            pipeline.stop()
            raise
        self._pipeline = pipeline
        return pipeline

    def capture_rgb_frame(self) -> np.ndarray:
        pipeline = self._ensure_started()
        frames = pipeline.wait_for_frames(self.timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("missing L515 color frame")
        return np.asarray(color_frame.get_data())

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


def _timestamp_for_filename(ts: str) -> str:
    return ts.replace(":", "").replace("-", "").replace(".", "").replace("+", "Z")


def save_snapshot(image_rgb: np.ndarray, snapshot_dir: Path, ts: str) -> str:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    base_path = snapshot_dir / f"l515-{_timestamp_for_filename(ts)}.jpg"
    path = base_path
    counter = 1
    while path.exists():
        path = snapshot_dir / f"{base_path.stem}-{counter:03d}{base_path.suffix}"
        counter += 1
    Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB").save(path, quality=90)
    return str(path)


def process_user_detected_event(
    event: dict[str, Any],
    *,
    frame_source: RGBFrameSource,
    classifier: ImageClassifier,
    q_result: QueueLike,
    snapshot_dir: Path,
    now: Callable[[], datetime] = datetime.now,
) -> dict[str, Any] | None:
    if event.get("event") != "user_detected":
        return None
    if "distance_cm" not in event or "ts" not in event:
        raise ValueError("user_detected payload must include distance_cm and ts")

    ts = now().isoformat(timespec="seconds")
    image_rgb = frame_source.capture_rgb_frame()
    prediction = classifier.predict(image_rgb)
    snapshot_path = save_snapshot(image_rgb, snapshot_dir, ts)
    result = build_recognition_result(
        predicted_class=prediction.label,
        confidence=prediction.confidence,
        snapshot_path=snapshot_path,
        ts=ts,
    )
    q_result.put(result)
    return result


def run_vision_runtime_loop(
    q_detected: QueueLike,
    q_result: QueueLike,
    *,
    frame_source: RGBFrameSource | None = None,
    classifier: ImageClassifier | None = None,
    config: VisionRuntimeConfig | None = None,
    now: Callable[[], datetime] = datetime.now,
) -> int:
    runtime_config = config or VisionRuntimeConfig()
    source = frame_source or L515ColorCamera()
    model = classifier or create_default_classifier()
    consumed = 0
    emitted = 0

    try:
        while runtime_config.max_events is None or consumed < runtime_config.max_events:
            try:
                event = q_detected.get(timeout=runtime_config.input_timeout_sec)
            except Empty:
                continue

            consumed += 1
            result = process_user_detected_event(
                event,
                frame_source=source,
                classifier=model,
                q_result=q_result,
                snapshot_dir=runtime_config.snapshot_dir,
                now=now,
            )
            if result is not None:
                emitted += 1
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()

    return emitted
