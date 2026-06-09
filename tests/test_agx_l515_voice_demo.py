from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory

import numpy as np

from agx_l515_voice_demo import (
    RealtimeMonitorConfig,
    _depth_frame_to_uint16,
    _last_frame_bytes,
    run_depth_to_voice_loop,
    run_realtime_monitor_loop,
    select_latest_export_model,
)
from firmware_l515.distance_trigger import DistanceTriggerConfig
from model_adapter import ModelPrediction, StaticClassifier


class FakeRGBDCamera:
    def __init__(self, *, depth_value: int = 250, depth_values: list[int] | None = None) -> None:
        self.depth_scale_m = 0.001
        self.started = False
        self.stopped = False
        self.latest_color = np.zeros((12, 16, 3), dtype=np.uint8)
        self.latest_color[:, :, 0] = 90
        values = depth_values if depth_values is not None else [depth_value]
        self.depth_frames = [np.full((6, 6), value, dtype=np.uint16) for value in values]

    def start(self) -> None:
        self.started = True

    def read_depth_frame(self) -> np.ndarray:
        return self.depth_frames.pop(0)

    def read_rgbd_frame(self) -> tuple[np.ndarray, np.ndarray]:
        return self.read_depth_frame(), self.capture_rgb_frame()

    def capture_rgb_frame(self) -> np.ndarray:
        return self.latest_color

    def stop(self) -> None:
        self.stopped = True


class FakeIndependentStreamCamera(FakeRGBDCamera):
    def __init__(self, *, depth_values: list[int]) -> None:
        super().__init__(depth_values=depth_values)
        self.stream_started = False

    def start_rgb_stream(self, frame_sink, *, target_fps=None) -> bool:
        self.stream_started = True
        frame_sink(self.latest_color)
        return True


class FakeChangingRGBCamera(FakeRGBDCamera):
    def __init__(self, *, depth_values: list[int], color_values: list[int]) -> None:
        super().__init__(depth_values=depth_values)
        self.color_values = color_values
        self.frame_index = -1

    def read_depth_frame(self) -> np.ndarray:
        self.frame_index += 1
        return super().read_depth_frame()

    def capture_rgb_frame(self) -> np.ndarray:
        index = min(max(self.frame_index, 0), len(self.color_values) - 1)
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        image[:, :, 0] = self.color_values[index]
        return image


class SequenceClassifier:
    def __init__(self, predictions: list[ModelPrediction]) -> None:
        self.predictions = predictions
        self.calls = 0

    def predict(self, image_rgb: np.ndarray) -> ModelPrediction:
        index = min(self.calls, len(self.predictions) - 1)
        self.calls += 1
        return self.predictions[index]


def drain_queue(queue: Queue[dict[str, object]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    while not queue.empty():
        items.append(queue.get(timeout=1))
    return items


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    packed_depth = np.array([[[1, 0], [255, 1]]], dtype=np.uint8)
    converted = _depth_frame_to_uint16(packed_depth)
    require(converted.shape == (1, 2), "packed V4L2 depth bytes should become one uint16 channel")
    require(converted[0, 0] == 1, "low byte should be preserved")
    require(converted[0, 1] == 511, "little-endian high byte should be applied")
    require(_last_frame_bytes(b"aaaabbbb", 4, label="test") == b"bbbb", "V4L2 capture should use the final frame")

    latest = select_latest_export_model()
    require(latest.name in {"best.onnx", "best.pt"}, "latest export selector should return a model artifact")
    require("20260608-demo-accept-recall" in latest.as_posix(), "AGX demo should default to the newest curated export")

    camera = FakeRGBDCamera()
    classifier = StaticClassifier(ModelPrediction(label="accept", confidence=0.91))
    voice_cues: list[dict[str, object]] = []
    q_result: Queue[dict[str, object]] = Queue()

    with TemporaryDirectory() as tmp:
        emitted = run_depth_to_voice_loop(
            camera=camera,
            classifier=classifier,
            voice_sink=lambda cue: voice_cues.append(dict(cue)),
            trigger_config=DistanceTriggerConfig(
                trigger_distance_cm=30.0,
                cooldown_sec=0.0,
                required_consecutive_frames=1,
                invalid_ratio_threshold=0.9,
            ),
            snapshot_dir=Path(tmp),
            q_result=q_result,
            max_events=1,
            max_frames=1,
            now=lambda: datetime(2026, 6, 9, 10, 0, 0),
            log=None,
        )

        require(emitted == 1, "depth trigger should produce one vision result")
        result = q_result.get(timeout=1)
        require(result["class"] == "accept", "classifier result should be sent to q_result")
        require(Path(str(result["snapshot_path"])).exists(), "RGB snapshot should be saved")
        require(len(voice_cues) == 1, "vision result should be routed to one voice cue")
        require(voice_cues[0]["category"] == "accept", "accept result should choose an accept WAV")
        require(str(voice_cues[0]["audio_path"]).endswith(".wav"), "voice cue should point at a WAV")

    require(camera.started, "camera should be started")
    require(camera.stopped, "camera should be stopped")

    invalid_camera = FakeRGBDCamera(depth_value=0)
    invalid_voice_cues: list[dict[str, object]] = []
    with TemporaryDirectory() as tmp:
        emitted = run_depth_to_voice_loop(
            camera=invalid_camera,
            classifier=classifier,
            voice_sink=lambda cue: invalid_voice_cues.append(dict(cue)),
            trigger_config=DistanceTriggerConfig(
                required_consecutive_frames=1,
                cooldown_sec=0.0,
            ),
            snapshot_dir=Path(tmp),
            max_events=1,
            max_frames=1,
            now=lambda: datetime(2026, 6, 9, 10, 0, 1),
            log=None,
        )

    require(emitted == 1, "invalid center depth fallback should emit a demo trigger")
    require(len(invalid_voice_cues) == 1, "invalid depth trigger should still route to voice")

    with TemporaryDirectory() as tmp:
        stream_camera = FakeIndependentStreamCamera(depth_values=[300, 300])
        stream_frames: list[tuple[int, int, int]] = []
        run_realtime_monitor_loop(
            camera=stream_camera,
            classifier=SequenceClassifier([ModelPrediction("accept", 0.93)] * 2),
            voice_sink=lambda cue: None,
            snapshot_dir=Path(tmp),
            monitor_config=RealtimeMonitorConfig(
                inference_interval_sec=0.0,
                stable_required=2,
                max_events=1,
                max_frames=2,
            ),
            camera_frame_sink=lambda image: stream_frames.append(tuple(image.shape)),
            now=lambda: datetime(2026, 6, 9, 10, 0, 2),
            log=None,
        )
        require(stream_camera.stream_started, "realtime loop should start independent RGB stream when camera supports it")
        require(stream_frames == [(12, 16, 3)], "independent RGB stream should own display frame publishing")

    with TemporaryDirectory() as tmp:
        realtime_q: Queue[dict[str, object]] = Queue()
        realtime_voice: list[dict[str, object]] = []
        frame_count = 0
        realtime_emitted = run_realtime_monitor_loop(
            camera=FakeRGBDCamera(depth_values=[300] * 6),
            classifier=SequenceClassifier([ModelPrediction("accept", 0.93)] * 6),
            voice_sink=lambda cue: realtime_voice.append(dict(cue)),
            trigger_config=DistanceTriggerConfig(invalid_ratio_threshold=0.9),
            snapshot_dir=Path(tmp),
            q_result=realtime_q,
            camera_frame_sink=lambda image: globals().__setitem__("frame_count_marker", image.shape),
            monitor_config=RealtimeMonitorConfig(
                inference_interval_sec=0.0,
                stable_required=2,
                max_events=1,
                max_frames=6,
            ),
            now=lambda: datetime(2026, 6, 9, 10, 0, 2),
            log=None,
        )
        realtime_items = drain_queue(realtime_q)
        recognition_items = [item for item in realtime_items if item["event"] == "recognition_result"]
        preview_items = [item for item in realtime_items if item["event"] == "vision_preview"]

        require(realtime_emitted == 1, "stable realtime accept should emit once")
        require(len(recognition_items) == 1, "realtime loop should send one formal recognition_result")
        require(recognition_items[0]["class"] == "accept", "realtime recognition should keep classifier label")
        require(Path(str(recognition_items[0]["snapshot_path"])).exists(), "realtime result should save event snapshot")
        require(len(preview_items) >= 1, "realtime loop should send preview payloads before formal result")
        require(len(realtime_voice) == 1, "stable realtime result should play one voice cue")
        require(globals().get("frame_count_marker") == (12, 16, 3), "realtime loop should publish live RGB frames")

    with TemporaryDirectory() as tmp:
        repeated_q: Queue[dict[str, object]] = Queue()
        repeated_voice: list[dict[str, object]] = []
        repeated_emitted = run_realtime_monitor_loop(
            camera=FakeRGBDCamera(depth_values=[300] * 6),
            classifier=SequenceClassifier([ModelPrediction("accept", 0.94)] * 6),
            voice_sink=lambda cue: repeated_voice.append(dict(cue)),
            snapshot_dir=Path(tmp),
            q_result=repeated_q,
            monitor_config=RealtimeMonitorConfig(
                inference_interval_sec=0.0,
                stable_required=2,
                max_frames=6,
            ),
            now=lambda: datetime(2026, 6, 9, 10, 0, 3),
            log=None,
        )
        repeated_recognitions = [item for item in drain_queue(repeated_q) if item["event"] == "recognition_result"]
        require(repeated_emitted == 1, "same object should not emit repeatedly while still present")
        require(len(repeated_recognitions) == 1, "same object should produce only one recognition_result")
        require(len(repeated_voice) == 1, "same object should play only one voice cue")

    with TemporaryDirectory() as tmp:
        release_q: Queue[dict[str, object]] = Queue()
        release_voice: list[dict[str, object]] = []
        release_emitted = run_realtime_monitor_loop(
            camera=FakeRGBDCamera(depth_values=[300, 300, 800, 800, 300, 300]),
            classifier=SequenceClassifier(
                [
                    ModelPrediction("accept", 0.95),
                    ModelPrediction("accept", 0.95),
                    ModelPrediction("reject", 0.91),
                    ModelPrediction("reject", 0.91),
                ]
            ),
            voice_sink=lambda cue: release_voice.append(dict(cue)),
            snapshot_dir=Path(tmp),
            q_result=release_q,
            monitor_config=RealtimeMonitorConfig(
                inference_interval_sec=0.0,
                stable_required=2,
                release_required_frames=1,
                max_frames=6,
            ),
            now=lambda: datetime(2026, 6, 9, 10, 0, 4),
            log=None,
        )
        release_recognitions = [item for item in drain_queue(release_q) if item["event"] == "recognition_result"]
        require(release_emitted == 2, "release should allow the next object to emit")
        require([item["class"] for item in release_recognitions] == ["accept", "reject"], "second object should use the later stable label")
        require(len(release_voice) == 2, "two released objects should play two cues")

    with TemporaryDirectory() as tmp:
        visual_q: Queue[dict[str, object]] = Queue()
        visual_voice: list[dict[str, object]] = []
        visual_emitted = run_realtime_monitor_loop(
            camera=FakeChangingRGBCamera(depth_values=[300, 300, 300], color_values=[20, 20, 120]),
            classifier=SequenceClassifier(
                [
                    ModelPrediction("accept", 0.95),
                    ModelPrediction("reject", 0.91),
                ]
            ),
            voice_sink=lambda cue: visual_voice.append(dict(cue)),
            snapshot_dir=Path(tmp),
            q_result=visual_q,
            monitor_config=RealtimeMonitorConfig(
                inference_interval_sec=0.0,
                stable_required=1,
                visual_rearm_cooldown_sec=0.0,
                visual_rearm_diff_threshold=12.0,
                max_frames=3,
            ),
            now=lambda: datetime(2026, 6, 9, 10, 0, 6),
            log=None,
        )
        visual_recognitions = [item for item in drain_queue(visual_q) if item["event"] == "recognition_result"]
        require(visual_emitted == 2, "visual change should re-arm detection without depth release")
        require([item["class"] for item in visual_recognitions] == ["accept", "reject"], "visual re-arm should run the next model prediction")
        require(len(visual_voice) == 2, "visual re-arm should play a second voice cue")

    with TemporaryDirectory() as tmp:
        low_q: Queue[dict[str, object]] = Queue()
        low_voice: list[dict[str, object]] = []
        low_emitted = run_realtime_monitor_loop(
            camera=FakeRGBDCamera(depth_values=[300] * 4),
            classifier=SequenceClassifier([ModelPrediction("accept", 0.4)] * 4),
            voice_sink=lambda cue: low_voice.append(dict(cue)),
            snapshot_dir=Path(tmp),
            q_result=low_q,
            monitor_config=RealtimeMonitorConfig(
                inference_interval_sec=0.0,
                confidence_threshold=0.5,
                stable_required=2,
                max_frames=4,
            ),
            now=lambda: datetime(2026, 6, 9, 10, 0, 5),
            log=None,
        )
        low_items = drain_queue(low_q)
        require(low_emitted == 0, "low confidence should not emit formal realtime recognition")
        require(all(item["event"] == "vision_preview" for item in low_items), "low confidence should only send preview payloads")
        require(len(low_voice) == 0, "low confidence preview should not play voice")

    print("[OK] AGX L515 demo loop connects depth trigger, vision result, and voice cue")


if __name__ == "__main__":
    main()
