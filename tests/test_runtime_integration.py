from __future__ import annotations

from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image

from model_adapter import ModelPrediction, StaticClassifier
from runtime import L515ColorCamera, VisionRuntimeConfig, run_vision_runtime_loop
from vision_contract import RECOGNITION_RESULT_FIELDS


class FakeFrameSource:
    def __init__(self) -> None:
        self.closed = False
        self.frame = np.zeros((12, 16, 3), dtype=np.uint8)
        self.frame[:, :, 0] = 80
        self.frame[:, :, 1] = 120
        self.frame[:, :, 2] = 160

    def capture_rgb_frame(self) -> np.ndarray:
        return self.frame

    def close(self) -> None:
        self.closed = True


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_l515_color_camera_warms_up_before_capture() -> None:
    class FakeColorFrame:
        def __init__(self, value: int) -> None:
            self.value = value

        def get_data(self) -> np.ndarray:
            return np.full((2, 4, 3), self.value, dtype=np.uint8)

    class FakeFrames:
        def __init__(self, value: int) -> None:
            self.value = value

        def get_color_frame(self) -> FakeColorFrame:
            return FakeColorFrame(self.value)

    class FakePipeline:
        instances: list["FakePipeline"] = []

        def __init__(self) -> None:
            self.wait_timeouts: list[int] = []
            self.next_value = 0
            self.stopped = False
            FakePipeline.instances.append(self)

        def start(self, config: object) -> None:
            self.config = config

        def wait_for_frames(self, timeout_ms: int) -> FakeFrames:
            self.wait_timeouts.append(timeout_ms)
            self.next_value += 1
            return FakeFrames(self.next_value)

        def stop(self) -> None:
            self.stopped = True

    class FakeConfig:
        def enable_stream(self, *args: object) -> None:
            self.args = args

    fake_rs = SimpleNamespace(
        pipeline=FakePipeline,
        config=FakeConfig,
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(rgb8="rgb8"),
    )
    original_rs = sys.modules.get("pyrealsense2")
    sys.modules["pyrealsense2"] = fake_rs
    try:
        camera = L515ColorCamera(width=4, height=2, fps=15, timeout_ms=3000, warmup_frames=2)
        image = camera.capture_rgb_frame()
        pipeline = FakePipeline.instances[-1]

        require(pipeline.wait_timeouts == [3000, 3000, 3000], "camera should warm up before capture")
        require(int(image[0, 0, 0]) == 3, "captured image should come after warmup frames")
        camera.close()
        require(pipeline.stopped, "camera close should stop the pipeline")
    finally:
        if original_rs is None:
            del sys.modules["pyrealsense2"]
        else:
            sys.modules["pyrealsense2"] = original_rs


def main() -> None:
    test_l515_color_camera_warms_up_before_capture()

    q_detected: Queue[dict[str, object]] = Queue()
    q_result: Queue[dict[str, object]] = Queue()
    q_voice: Queue[dict[str, object]] = Queue()
    sent_voice_cues: list[dict[str, object]] = []
    frame_source = FakeFrameSource()
    classifier = StaticClassifier(ModelPrediction(label="accept", confidence=0.91))

    q_detected.put(
        {
            "event": "user_detected",
            "distance_cm": 25.0,
            "ts": "2026-06-04T12:00:00",
        }
    )

    with TemporaryDirectory() as tmp:
        emitted = run_vision_runtime_loop(
            q_detected,
            q_result,
            q_voice=q_voice,
            frame_source=frame_source,
            classifier=classifier,
            voice_sink=lambda cue: sent_voice_cues.append(dict(cue)),
            config=VisionRuntimeConfig(snapshot_dir=Path(tmp), max_events=1),
            now=lambda: datetime(2026, 6, 4, 12, 0, 1),
        )

        require(emitted == 1, "runtime should emit exactly one recognition_result")
        result = q_result.get(timeout=1)
        require(set(result) == RECOGNITION_RESULT_FIELDS, "recognition_result fields drifted")
        require(result["event"] == "recognition_result", "event name must be recognition_result")
        require(result["class"] == "accept", "classifier label should become payload class")
        require(result["confidence"] == 0.91, "classifier confidence should be preserved")
        require(result["num_objects"] == 1, "v0.3 classification runtime must emit num_objects=1")
        require(result["ts"] == "2026-06-04T12:00:01", "runtime should stamp result timestamp")

        snapshot_path = Path(str(result["snapshot_path"]))
        require(snapshot_path.exists(), "runtime should save a snapshot")
        with Image.open(snapshot_path) as image:
            require(image.size == (16, 12), "snapshot should preserve frame dimensions")

        voice = q_voice.get(timeout=1)
        require(voice["event"] == "voice_feedback_cue", "optional q_voice should emit a voice cue")
        require(voice["category"] == "accept", "voice cue should match high-confidence accept")
        require(voice["source_class"] == result["class"], "voice cue should preserve source class")
        require(voice["source_ts"] == result["ts"], "voice cue should preserve source timestamp")
        require(sent_voice_cues == [voice], "optional voice sink should receive the same cue as q_voice")

    require(frame_source.closed, "runtime should close the frame source")

    q_detected = Queue()
    q_result = Queue()
    q_voice = Queue()
    frame_source = FakeFrameSource()
    reject_classifier = StaticClassifier(ModelPrediction(label="reject", confidence=0.88))
    q_detected.put(
        {
            "event": "user_detected",
            "distance_cm": 25.0,
            "ts": "2026-06-04T12:00:30",
        }
    )

    with TemporaryDirectory() as tmp:
        emitted = run_vision_runtime_loop(
            q_detected,
            q_result,
            q_voice=q_voice,
            frame_source=frame_source,
            classifier=reject_classifier,
            config=VisionRuntimeConfig(snapshot_dir=Path(tmp), max_events=1),
            now=lambda: datetime(2026, 6, 4, 12, 0, 31),
        )

        require(emitted == 1, "runtime should emit a reject recognition_result")
        result = q_result.get(timeout=1)
        require(result["class"] == "reject", "classifier reject label should become payload class")
        voice = q_voice.get(timeout=1)
        require(voice["category"] == "reject", "reject recognition_result should emit reject voice cue")
        require(voice["source_class"] == "reject", "reject voice cue should preserve source class")
        require(
            str(voice["audio_path"]).startswith("assets/voice/gpt-sovits/reject/reject-"),
            "reject voice cue should choose from the recorded reject WAV pool",
        )

    q_detected = Queue()
    q_result = Queue()
    frame_source = FakeFrameSource()
    for _ in range(2):
        q_detected.put(
            {
                "event": "user_detected",
                "distance_cm": 25.0,
                "ts": "2026-06-04T12:00:00",
            }
        )

    with TemporaryDirectory() as tmp:
        emitted = run_vision_runtime_loop(
            q_detected,
            q_result,
            frame_source=frame_source,
            classifier=classifier,
            config=VisionRuntimeConfig(snapshot_dir=Path(tmp), max_events=2),
            now=lambda: datetime(2026, 6, 4, 12, 0, 1),
        )

        require(emitted == 2, "runtime should emit both same-second recognition_result payloads")
        first = q_result.get(timeout=1)
        second = q_result.get(timeout=1)
        first_snapshot = Path(str(first["snapshot_path"]))
        second_snapshot = Path(str(second["snapshot_path"]))
        require(first_snapshot.exists(), "first same-second snapshot should exist")
        require(second_snapshot.exists(), "second same-second snapshot should exist")
        require(first_snapshot != second_snapshot, "same-second snapshots must not overwrite each other")

    print("[OK] fake queue runtime produced a valid recognition_result payload")


if __name__ == "__main__":
    main()
