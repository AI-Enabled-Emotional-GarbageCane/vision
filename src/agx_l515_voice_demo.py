from __future__ import annotations

import sys
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Literal, Protocol

import numpy as np

from agx_audio import DEFAULT_AUDIO_ROOT, AgxWavVoiceSink
from model_adapter import ImageClassifier, create_default_classifier
from runtime import DEFAULT_SNAPSHOT_DIR, process_user_detected_event, save_snapshot
from vision_contract import build_recognition_result
from voice_feedback import VoiceCueRouter


VISION_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = VISION_ROOT.parent
FIRMWARE_ROOT = WORKSPACE_ROOT / "firmware"
if FIRMWARE_ROOT.exists() and str(FIRMWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIRMWARE_ROOT))

from firmware_l515.distance_trigger import (
    DistanceTriggerConfig,
    L515DistanceTrigger,
    compute_center_roi_distance_cm,
    is_depth_frame_invalid,
)


LogSink = Callable[[str], None]
FrameSink = Callable[[np.ndarray], None]


class RGBDFrameSource(Protocol):
    depth_scale_m: float | None

    def start(self) -> None: ...

    def read_rgbd_frame(self) -> tuple[np.ndarray, np.ndarray]: ...

    def read_depth_frame(self) -> np.ndarray: ...

    def capture_rgb_frame(self) -> np.ndarray: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class L515RGBDCameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 30
    timeout_ms: int = 3000
    laser_on: bool = True


class L515RGBDCamera:
    """Single RealSense pipeline that provides both depth trigger frames and RGB snapshots."""

    def __init__(self, config: L515RGBDCameraConfig | None = None) -> None:
        self._config = config or L515RGBDCameraConfig()
        self._rs = None
        self._pipeline = None
        self._latest_color: np.ndarray | None = None
        self._lock = Lock()
        self.depth_scale_m: float | None = None

    def start(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.depth,
            self._config.width,
            self._config.height,
            rs.format.z16,
            self._config.fps,
        )
        cfg.enable_stream(
            rs.stream.color,
            self._config.width,
            self._config.height,
            rs.format.rgb8,
            self._config.fps,
        )

        profile = pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale_m = float(depth_sensor.get_depth_scale())

        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1 if self._config.laser_on else 0)

        try:
            for _ in range(max(0, int(self._config.warmup_frames))):
                self._read_frames(pipeline, during_warmup=True)
        except RuntimeError:
            pipeline.stop()
            raise

        self._rs = rs
        self._pipeline = pipeline

    def read_depth_frame(self) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("L515RGBDCamera.start() must be called before reading frames")
        depth_raw, _ = self._read_frames(self._pipeline, during_warmup=False)
        return depth_raw

    def read_rgbd_frame(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("L515RGBDCamera.start() must be called before reading frames")
        depth_raw, color_rgb = self._read_frames(self._pipeline, during_warmup=False)
        if color_rgb is None:
            raise RuntimeError("missing L515 color frame")
        return depth_raw, color_rgb

    def capture_rgb_frame(self) -> np.ndarray:
        with self._lock:
            if self._latest_color is not None:
                return self._latest_color.copy()

        if self._pipeline is None:
            raise RuntimeError("L515RGBDCamera.start() must be called before capturing RGB frames")
        _, color_rgb = self._read_frames(self._pipeline, during_warmup=False)
        if color_rgb is None:
            raise RuntimeError("missing L515 color frame")
        return color_rgb

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def close(self) -> None:
        self.stop()

    def _read_frames(self, pipeline: Any, *, during_warmup: bool) -> tuple[np.ndarray, np.ndarray | None]:
        timeout_ms = int(self._config.timeout_ms)
        try:
            frames = pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as exc:
            phase = " during warmup" if during_warmup else ""
            raise RuntimeError(f"timed out waiting for L515 RGB-D frame{phase} after {timeout_ms} ms") from exc

        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError("missing L515 depth frame")
        depth_raw = np.asanyarray(depth_frame.get_data()).copy()

        color_rgb = None
        color_frame = frames.get_color_frame()
        if color_frame:
            color_rgb = np.asanyarray(color_frame.get_data()).copy()
            with self._lock:
                self._latest_color = color_rgb

        return depth_raw, color_rgb


@dataclass(frozen=True)
class V4L2RGBDCameraConfig:
    depth_device: str = "/dev/video2"
    color_device: str = "/dev/video6"
    depth_width: int = 480
    depth_height: int = 640
    color_width: int = 640
    color_height: int = 480
    fps: int = 30
    depth_scale_m: float = 0.00025
    warmup_frames: int = 0
    color_capture_frames: int = 30
    capture_timeout_sec: float = 5.0


def _depth_frame_to_uint16(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.dtype == np.uint16:
        if image.ndim == 2:
            return image.copy()
        if image.ndim == 3:
            return image[:, :, 0].copy()

    if image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] >= 2:
        low = image[:, :, 0].astype(np.uint16)
        high = image[:, :, 1].astype(np.uint16)
        return low | (high << 8)

    if image.dtype == np.uint8 and image.ndim == 2:
        return image.astype(np.uint16)

    raise ValueError(f"unsupported V4L2 depth frame shape/dtype: {image.shape} {image.dtype}")


def _last_frame_bytes(payload: bytes, frame_size: int, *, label: str) -> bytes:
    if frame_size <= 0:
        raise ValueError(f"{label} frame_size must be positive")
    if len(payload) < frame_size:
        raise RuntimeError(
            f"unexpected V4L2 {label} payload size: got {len(payload)} bytes, expected at least {frame_size}"
        )
    if len(payload) % frame_size != 0:
        raise RuntimeError(
            f"unexpected V4L2 {label} payload size: got {len(payload)} bytes, not a multiple of {frame_size}"
        )
    return payload[-frame_size:]


class V4L2RGBDCamera:
    """V4L2 fallback for Jetson setups where librealsense cannot enumerate L515."""

    def __init__(self, config: V4L2RGBDCameraConfig | None = None) -> None:
        self._config = config or V4L2RGBDCameraConfig()
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._cv2 = None
        self._color_cap = None
        self._depth_cap = None
        self._latest_color: np.ndarray | None = None
        self._color_lock = Lock()
        self._color_capture_lock = Lock()
        self._color_stop_event = Event()
        self._color_thread: Thread | None = None
        self.depth_scale_m: float | None = self._config.depth_scale_m

    def start(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="l515-v4l2-")
        self._open_streaming_captures()
        if self._color_cap is None and shutil.which("v4l2-ctl") is None:
            raise RuntimeError("V4L2 L515 color capture requires OpenCV VideoCapture or v4l2-ctl")
        if self._depth_cap is None and shutil.which("v4l2-ctl") is None:
            raise RuntimeError("V4L2 L515 depth capture requires OpenCV VideoCapture or v4l2-ctl")

        for _ in range(max(0, int(self._config.warmup_frames))):
            self.capture_rgb_frame()

    def read_depth_frame(self) -> np.ndarray:
        if self._depth_cap is not None:
            ok, frame = self._depth_cap.read()
            if ok and frame is not None:
                return _depth_frame_to_uint16(frame)

        if self._tmpdir is None:
            raise RuntimeError("V4L2RGBDCamera.start() must be called before reading depth frames")
        path = Path(self._tmpdir.name) / "depth.raw"
        self._capture_raw(
            device=self._config.depth_device,
            width=self._config.depth_width,
            height=self._config.depth_height,
            pixelformat="Z16 ",
            output_path=path,
            frame_count=1,
        )
        expected = self._config.depth_width * self._config.depth_height * 2
        data = _last_frame_bytes(path.read_bytes(), expected, label="depth")
        return np.frombuffer(data, dtype="<u2").reshape(self._config.depth_height, self._config.depth_width).copy()

    def capture_rgb_frame(self) -> np.ndarray:
        with self._color_lock:
            if self._latest_color is not None:
                return self._latest_color.copy()

        if self._color_cap is not None:
            image_rgb = self._read_color_stream_frame()
            if image_rgb is not None:
                return image_rgb

        if self._tmpdir is None:
            raise RuntimeError("V4L2RGBDCamera.start() must be called before capturing RGB frames")
        import cv2

        path = Path(self._tmpdir.name) / "color.yuyv"
        self._capture_raw(
            device=self._config.color_device,
            width=self._config.color_width,
            height=self._config.color_height,
            pixelformat="YUYV",
            output_path=path,
            frame_count=max(1, int(self._config.color_capture_frames)),
        )
        expected = self._config.color_width * self._config.color_height * 2
        data = _last_frame_bytes(path.read_bytes(), expected, label="color")
        yuyv = np.frombuffer(data, dtype=np.uint8).reshape(
            self._config.color_height,
            self._config.color_width,
            2,
        )
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)

    def read_rgbd_frame(self) -> tuple[np.ndarray, np.ndarray]:
        depth_raw = self.read_depth_frame()
        color_rgb = self.capture_rgb_frame()
        return depth_raw, color_rgb

    def start_rgb_stream(self, frame_sink: FrameSink, *, target_fps: float | None = None) -> bool:
        if self._color_cap is None or self._color_thread is not None:
            return False

        fps = float(target_fps or self._config.fps or 30)
        self._color_stop_event.clear()
        self._color_thread = Thread(
            target=self._run_color_stream,
            args=(frame_sink, max(1.0, fps)),
            daemon=True,
            name="l515-v4l2-rgb-stream",
        )
        self._color_thread.start()
        return True

    def stop(self) -> None:
        self._color_stop_event.set()
        if self._color_thread is not None:
            self._color_thread.join(timeout=1.0)
            self._color_thread = None
        for capture in (self._color_cap, self._depth_cap):
            if capture is not None:
                capture.release()
        self._color_cap = None
        self._depth_cap = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def close(self) -> None:
        self.stop()

    def _open_streaming_captures(self) -> None:
        try:
            import cv2
        except ImportError:
            return

        self._cv2 = cv2
        self._color_cap = self._open_capture(
            self._config.color_device,
            width=self._config.color_width,
            height=self._config.color_height,
            fps=self._config.fps,
            fourcc="YUYV",
            convert_rgb=True,
        )
        self._depth_cap = self._open_capture(
            self._config.depth_device,
            width=self._config.depth_width,
            height=self._config.depth_height,
            fps=self._config.fps,
            fourcc="Z16 ",
            convert_rgb=False,
        )

    def _open_capture(
        self,
        device: str,
        *,
        width: int,
        height: int,
        fps: int,
        fourcc: str,
        convert_rgb: bool,
    ) -> Any | None:
        cv2 = self._cv2
        if cv2 is None:
            return None

        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            return None

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        capture.set(cv2.CAP_PROP_FPS, int(fps))
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 1 if convert_rgb else 0)
        return capture if capture.isOpened() else None

    def _run_color_stream(self, frame_sink: FrameSink, target_fps: float) -> None:
        min_interval_sec = 1.0 / target_fps
        while not self._color_stop_event.is_set():
            started_at = time.monotonic()
            image_rgb = self._read_color_stream_frame()
            if image_rgb is not None:
                try:
                    frame_sink(image_rgb)
                except Exception:
                    pass
            elapsed = time.monotonic() - started_at
            sleep_sec = min_interval_sec - elapsed
            if sleep_sec > 0:
                self._color_stop_event.wait(sleep_sec)

    def _read_color_stream_frame(self) -> np.ndarray | None:
        if self._color_cap is None:
            return None
        with self._color_capture_lock:
            ok, frame = self._color_cap.read()
        if not ok or frame is None:
            return None
        image_rgb = self._color_frame_to_rgb(frame)
        with self._color_lock:
            self._latest_color = image_rgb
        return image_rgb.copy()

    def _color_frame_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        if cv2 is None:
            raise RuntimeError("OpenCV is required to convert V4L2 color frames")

        image = np.asarray(frame)
        if image.ndim == 3 and image.shape[2] == 2:
            return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUY2)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        raise ValueError(f"unsupported V4L2 color frame shape/dtype: {image.shape} {image.dtype}")

    def _capture_raw(
        self,
        *,
        device: str,
        width: int,
        height: int,
        pixelformat: str,
        output_path: Path,
        frame_count: int,
    ) -> None:
        command = [
            "v4l2-ctl",
            "-d",
            device,
            f"--set-fmt-video=width={width},height={height},pixelformat={pixelformat}",
            "--stream-mmap",
            f"--stream-count={max(1, int(frame_count))}",
            f"--stream-to={output_path}",
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self._config.capture_timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"failed to capture V4L2 frame from {device}: {completed.stderr.strip()}"
            )


def select_latest_export_model(exports_root: Path = VISION_ROOT / "exports") -> Path:
    if not exports_root.is_dir():
        raise FileNotFoundError(f"missing exports directory: {exports_root}")

    export_dirs = [path for path in exports_root.iterdir() if path.is_dir()]
    for export_dir in sorted(export_dirs, key=lambda path: path.name, reverse=True):
        for filename in ("best.onnx", "best.pt"):
            model_path = export_dir / filename
            if model_path.is_file():
                return model_path

    raise FileNotFoundError(f"missing best.onnx or best.pt under {exports_root}")


CameraBackend = Literal["auto", "realsense", "v4l2"]


def realsense_device_count() -> int:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return 0

    try:
        return len(rs.context().query_devices())
    except RuntimeError:
        return 0


def create_camera(
    *,
    backend: CameraBackend = "auto",
    camera_config: L515RGBDCameraConfig | None = None,
    v4l2_config: V4L2RGBDCameraConfig | None = None,
    log: LogSink | None = print,
) -> RGBDFrameSource:
    selected = backend
    if selected == "auto":
        selected = "realsense" if realsense_device_count() > 0 else "v4l2"
        if log is not None:
            log(f"[agx-demo] camera_backend=auto selected {selected}")

    if selected == "realsense":
        return L515RGBDCamera(camera_config)
    if selected == "v4l2":
        return V4L2RGBDCamera(v4l2_config)
    raise ValueError(f"unsupported camera backend: {backend}")


def run_depth_to_voice_loop(
    *,
    camera: RGBDFrameSource,
    classifier: ImageClassifier,
    voice_sink: Callable[[dict[str, Any]], None],
    trigger_config: DistanceTriggerConfig | None = None,
    voice_router: VoiceCueRouter | None = None,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    q_result: Queue[dict[str, Any]] | None = None,
    max_events: int | None = None,
    max_frames: int | None = None,
    poll_delay_sec: float = 0.0,
    trigger_on_invalid_center: bool = True,
    now: Callable[[], datetime] = datetime.now,
    log: LogSink | None = print,
) -> int:
    q_detected: Queue[dict[str, Any]] = Queue()
    results = q_result or Queue()
    cfg = trigger_config or DistanceTriggerConfig()
    trigger = L515DistanceTrigger(q_detected, config=cfg)
    router = voice_router or VoiceCueRouter(low_confidence_threshold=0.0)
    processed_frames = 0
    emitted_events = 0
    invalid_frame_count = 0
    invalid_detected_active = False
    last_invalid_emit_at: float | None = None

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    def maybe_emit_invalid_depth_event(depth_raw: np.ndarray, depth_scale_m: float) -> bool:
        nonlocal invalid_frame_count, invalid_detected_active, last_invalid_emit_at
        if not trigger_on_invalid_center:
            return False

        invalid = is_depth_frame_invalid(
            depth_raw,
            depth_scale_m=depth_scale_m,
            center_fraction=cfg.invalid_center_fraction,
            invalid_ratio_threshold=cfg.invalid_ratio_threshold,
            min_valid_cm=cfg.min_valid_cm,
            max_valid_cm=cfg.max_valid_cm,
        )
        if not invalid:
            invalid_frame_count = 0
            invalid_detected_active = False
            return False

        if invalid_detected_active:
            return False

        invalid_frame_count += 1
        if invalid_frame_count < max(1, int(cfg.required_consecutive_frames)):
            return False

        now_mono = time.monotonic()
        if last_invalid_emit_at is not None and (now_mono - last_invalid_emit_at) < cfg.cooldown_sec:
            return False

        last_invalid_emit_at = now_mono
        invalid_detected_active = True
        q_detected.put(
            {
                "event": "user_detected",
                "distance_cm": 0.0,
                "ts": now().isoformat(timespec="seconds"),
                "trigger": "invalid_depth_center",
            }
        )
        return True

    try:
        camera.start()
        emit("[agx-demo] L515 RGB-D camera started")
        while True:
            if max_frames is not None and processed_frames >= max_frames:
                break
            if max_events is not None and emitted_events >= max_events:
                break

            depth_raw = camera.read_depth_frame()
            processed_frames += 1
            depth_scale_m = camera.depth_scale_m
            if depth_scale_m is None:
                raise RuntimeError("camera did not provide depth_scale_m")

            if trigger.process_depth_frame(depth_raw, depth_scale_m=depth_scale_m):
                emit("[agx-demo] depth trigger emitted user_detected")
            elif maybe_emit_invalid_depth_event(depth_raw, depth_scale_m):
                emit("[agx-demo] invalid center depth emitted user_detected")

            while True:
                try:
                    event = q_detected.get_nowait()
                except Empty:
                    break

                result = process_user_detected_event(
                    event,
                    frame_source=camera,
                    classifier=classifier,
                    q_result=results,
                    voice_router=router,
                    voice_sink=voice_sink,
                    snapshot_dir=snapshot_dir,
                    now=now,
                )
                if result is None:
                    continue

                emitted_events += 1
                emit(
                    "[agx-demo] vision result class={label} confidence={confidence:.3f} snapshot={snapshot}".format(
                        label=result["class"],
                        confidence=float(result["confidence"]),
                        snapshot=result["snapshot_path"],
                    )
                )
                if max_events is not None and emitted_events >= max_events:
                    break

            if poll_delay_sec > 0:
                time.sleep(poll_delay_sec)
    finally:
        camera.stop()
        emit("[agx-demo] L515 RGB-D camera stopped")

    return emitted_events


@dataclass(frozen=True)
class RealtimeMonitorConfig:
    inference_interval_sec: float = 0.2
    present_distance_cm: float = 45.0
    release_distance_cm: float = 60.0
    confidence_threshold: float = 0.5
    stable_required: int = 4
    release_required_frames: int = 3
    visual_rearm_cooldown_sec: float = 1.0
    visual_rearm_diff_threshold: float = 12.0
    max_events: int | None = None
    max_frames: int | None = None
    poll_delay_sec: float = 0.0


def _object_presence_from_depth(
    depth_raw: np.ndarray,
    *,
    depth_scale_m: float,
    was_present: bool,
    monitor_config: RealtimeMonitorConfig,
    trigger_config: DistanceTriggerConfig,
    trigger_on_invalid_center: bool,
) -> tuple[bool, float | None]:
    invalid = is_depth_frame_invalid(
        depth_raw,
        depth_scale_m=depth_scale_m,
        center_fraction=trigger_config.invalid_center_fraction,
        invalid_ratio_threshold=trigger_config.invalid_ratio_threshold,
        min_valid_cm=trigger_config.min_valid_cm,
        max_valid_cm=trigger_config.max_valid_cm,
    )
    if invalid and trigger_on_invalid_center:
        return True, 0.0

    distance_cm = compute_center_roi_distance_cm(
        depth_raw,
        depth_scale_m=depth_scale_m,
        roi_fraction=trigger_config.roi_fraction,
        min_valid_cm=trigger_config.min_valid_cm,
        max_valid_cm=trigger_config.max_valid_cm,
    )
    if distance_cm is None:
        return False, None

    if was_present:
        return distance_cm < monitor_config.release_distance_cm, distance_cm
    return distance_cm <= monitor_config.present_distance_cm, distance_cm


def _vision_preview_payload(
    *,
    object_present: bool,
    ts: str,
    predicted_class: str | None = None,
    confidence: float | None = None,
    distance_cm: float | None = None,
    stable_count: int = 0,
    stable_required: int = 1,
) -> dict[str, Any]:
    return {
        "event": "vision_preview",
        "object_present": object_present,
        "class": predicted_class,
        "confidence": confidence,
        "distance_cm": distance_cm,
        "stable_count": stable_count,
        "stable_required": stable_required,
        "ts": ts,
    }


def _rgb_change_score(image_a: np.ndarray, image_b: np.ndarray) -> float:
    first = np.asarray(image_a)
    second = np.asarray(image_b)
    if first.shape != second.shape:
        return float("inf")
    first_sample = first[::16, ::16, :].astype(np.float32, copy=False)
    second_sample = second[::16, ::16, :].astype(np.float32, copy=False)
    return float(np.mean(np.abs(first_sample - second_sample)))


def run_realtime_monitor_loop(
    *,
    camera: RGBDFrameSource,
    classifier: ImageClassifier,
    voice_sink: Callable[[dict[str, Any]], None],
    trigger_config: DistanceTriggerConfig | None = None,
    voice_router: VoiceCueRouter | None = None,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    q_result: Queue[dict[str, Any]] | None = None,
    camera_frame_sink: FrameSink | None = None,
    monitor_config: RealtimeMonitorConfig | None = None,
    trigger_on_invalid_center: bool = True,
    now: Callable[[], datetime] = datetime.now,
    monotonic: Callable[[], float] = time.monotonic,
    log: LogSink | None = print,
) -> int:
    results = q_result or Queue()
    cfg = monitor_config or RealtimeMonitorConfig()
    depth_cfg = trigger_config or DistanceTriggerConfig()
    router = voice_router or VoiceCueRouter(low_confidence_threshold=0.0)
    processed_frames = 0
    emitted_events = 0
    object_active = False
    emitted_for_object = False
    absent_frame_count = 0
    stable_label: str | None = None
    stable_count = 0
    last_inference_at = -float("inf")
    last_emitted_image: np.ndarray | None = None
    last_emitted_at: float | None = None

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    def publish_preview(
        *,
        object_present: bool,
        predicted_class: str | None = None,
        confidence: float | None = None,
        distance_cm: float | None = None,
    ) -> None:
        results.put(
            _vision_preview_payload(
                object_present=object_present,
                predicted_class=predicted_class,
                confidence=confidence,
                distance_cm=distance_cm,
                stable_count=stable_count,
                stable_required=max(1, int(cfg.stable_required)),
                ts=now().isoformat(timespec="seconds"),
            )
        )

    try:
        camera.start()
        live_stream_started = False
        if camera_frame_sink is not None:
            start_rgb_stream = getattr(camera, "start_rgb_stream", None)
            if callable(start_rgb_stream):
                live_stream_started = bool(start_rgb_stream(camera_frame_sink))
                if live_stream_started:
                    emit("[agx-demo] independent RGB display stream started")
        emit("[agx-demo] L515 realtime monitor started")
        while True:
            if cfg.max_frames is not None and processed_frames >= cfg.max_frames:
                break
            if cfg.max_events is not None and emitted_events >= cfg.max_events:
                break

            depth_raw = camera.read_depth_frame()
            image_rgb = camera.capture_rgb_frame()
            processed_frames += 1

            if camera_frame_sink is not None and not live_stream_started:
                try:
                    camera_frame_sink(image_rgb)
                except Exception as exc:  # pragma: no cover - defensive for UI stream clients
                    emit(f"[agx-demo] camera frame sink ignored frame: {exc}")

            depth_scale_m = camera.depth_scale_m
            if depth_scale_m is None:
                raise RuntimeError("camera did not provide depth_scale_m")

            present_candidate, distance_cm = _object_presence_from_depth(
                depth_raw,
                depth_scale_m=depth_scale_m,
                was_present=object_active,
                monitor_config=cfg,
                trigger_config=depth_cfg,
                trigger_on_invalid_center=trigger_on_invalid_center,
            )
            if present_candidate:
                absent_frame_count = 0
                if not object_active:
                    emit("[agx-demo] realtime object present")
                object_active = True
            elif object_active:
                absent_frame_count += 1
                if absent_frame_count >= max(1, int(cfg.release_required_frames)):
                    object_active = False
                    emitted_for_object = False
                    stable_label = None
                    stable_count = 0
                    absent_frame_count = 0
                    publish_preview(object_present=False, distance_cm=distance_cm)
                    emit("[agx-demo] realtime object released")
            else:
                stable_label = None
                stable_count = 0

            if not object_active:
                if cfg.poll_delay_sec > 0:
                    time.sleep(cfg.poll_delay_sec)
                continue

            if emitted_for_object:
                can_rearm_by_time = (
                    last_emitted_at is not None
                    and (monotonic() - last_emitted_at) >= max(0.0, float(cfg.visual_rearm_cooldown_sec))
                )
                changed_enough = (
                    last_emitted_image is not None
                    and _rgb_change_score(last_emitted_image, image_rgb) >= float(cfg.visual_rearm_diff_threshold)
                )
                if can_rearm_by_time and changed_enough:
                    emitted_for_object = False
                    stable_label = None
                    stable_count = 0
                    emit("[agx-demo] realtime visual change re-armed detection")
                else:
                    if cfg.poll_delay_sec > 0:
                        time.sleep(cfg.poll_delay_sec)
                    continue

            now_mono = monotonic()
            if (now_mono - last_inference_at) < max(0.0, float(cfg.inference_interval_sec)):
                if cfg.poll_delay_sec > 0:
                    time.sleep(cfg.poll_delay_sec)
                continue
            last_inference_at = now_mono

            prediction = classifier.predict(image_rgb)
            if prediction.confidence >= cfg.confidence_threshold:
                if stable_label == prediction.label:
                    stable_count += 1
                else:
                    stable_label = prediction.label
                    stable_count = 1
            else:
                stable_label = None
                stable_count = 0

            publish_preview(
                object_present=True,
                predicted_class=prediction.label,
                confidence=prediction.confidence,
                distance_cm=distance_cm,
            )

            if stable_count < max(1, int(cfg.stable_required)):
                continue

            ts = now().isoformat(timespec="seconds")
            snapshot_path = save_snapshot(image_rgb, snapshot_dir, ts)
            result = build_recognition_result(
                predicted_class=prediction.label,
                confidence=prediction.confidence,
                snapshot_path=snapshot_path,
                ts=ts,
            )
            results.put(result)
            voice_sink(router.route(result).to_payload())
            emitted_for_object = True
            last_emitted_image = image_rgb.copy()
            last_emitted_at = monotonic()
            emitted_events += 1
            emit(
                "[agx-demo] realtime stable result class={label} confidence={confidence:.3f} snapshot={snapshot}".format(
                    label=result["class"],
                    confidence=float(result["confidence"]),
                    snapshot=result["snapshot_path"],
                )
            )
    finally:
        camera.stop()
        emit("[agx-demo] L515 realtime monitor stopped")

    return emitted_events


def run_agx_l515_voice_demo(
    *,
    model_path: Path | None = None,
    audio_root: Path | None = None,
    audio_device: str | None = None,
    dry_run_audio: bool = False,
    no_audio_delay: bool = False,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    trigger_config: DistanceTriggerConfig | None = None,
    camera_config: L515RGBDCameraConfig | None = None,
    v4l2_config: V4L2RGBDCameraConfig | None = None,
    camera_backend: CameraBackend = "auto",
    q_result: Queue[dict[str, Any]] | None = None,
    camera_frame_sink: FrameSink | None = None,
    loop_mode: Literal["realtime", "depth-trigger"] = "realtime",
    monitor_config: RealtimeMonitorConfig | None = None,
    max_events: int | None = None,
    max_frames: int | None = None,
    poll_delay_sec: float = 0.0,
    trigger_on_invalid_center: bool = True,
    log: LogSink | None = print,
) -> int:
    selected_model = model_path or select_latest_export_model()
    if log is not None:
        log(f"[agx-demo] model={selected_model}")

    camera = create_camera(
        backend=camera_backend,
        camera_config=camera_config,
        v4l2_config=v4l2_config,
        log=log,
    )
    classifier = create_default_classifier(selected_model)
    voice_sink = AgxWavVoiceSink(
        audio_root=audio_root if audio_root is not None else DEFAULT_AUDIO_ROOT,
        audio_device=audio_device,
        dry_run=dry_run_audio,
        no_delay=no_audio_delay,
        log=log,
    )

    if loop_mode == "depth-trigger":
        return run_depth_to_voice_loop(
            camera=camera,
            classifier=classifier,
            voice_sink=voice_sink,
            trigger_config=trigger_config,
            snapshot_dir=snapshot_dir,
            q_result=q_result,
            max_events=max_events,
            max_frames=max_frames,
            poll_delay_sec=poll_delay_sec,
            trigger_on_invalid_center=trigger_on_invalid_center,
            log=log,
        )

    realtime_config = monitor_config or RealtimeMonitorConfig(
        max_events=max_events,
        max_frames=max_frames,
        poll_delay_sec=poll_delay_sec,
    )
    return run_realtime_monitor_loop(
        camera=camera,
        classifier=classifier,
        voice_sink=voice_sink,
        trigger_config=trigger_config,
        snapshot_dir=snapshot_dir,
        q_result=q_result,
        camera_frame_sink=camera_frame_sink,
        monitor_config=realtime_config,
        trigger_on_invalid_center=trigger_on_invalid_center,
        log=log,
    )
