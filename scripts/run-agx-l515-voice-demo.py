#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import threading
import time
from queue import Queue
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agx_audio import DEFAULT_AUDIO_DEVICE, DEFAULT_AUDIO_ROOT
from agx_l515_voice_demo import (
    L515RGBDCameraConfig,
    RealtimeMonitorConfig,
    V4L2RGBDCameraConfig,
    run_agx_l515_voice_demo,
    select_latest_export_model,
)
from firmware_l515.distance_trigger import DistanceTriggerConfig
from runtime import DEFAULT_SNAPSHOT_DIR


def positive_int_or_none(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def start_display_bridge(
    q_result: Queue[dict[str, Any]],
    *,
    host: str,
    port: int,
    log,
) -> tuple[threading.Thread, Any]:
    display_root = ROOT.parent / "display"
    sys.path.insert(0, str(display_root))
    from server import CameraFrameStore, run_display_server

    camera_frames = CameraFrameStore()

    thread = threading.Thread(
        target=run_display_server,
        args=(q_result,),
        kwargs={
            "host": host,
            "port": port,
            "static_root": display_root,
            "camera_frames": camera_frames,
            "audio_enabled": False,
        },
        daemon=True,
    )
    thread.start()
    log(f"[agx-demo] display UI: http://{host}:{port}")
    return thread, camera_frames


def main() -> int:
    default_model = select_latest_export_model()
    parser = argparse.ArgumentParser(
        description=(
            "Run the full AGX flow: L515 realtime RGB-D monitor -> L515 RGB vision model "
            "-> random accept/reject WAV -> AGX HDMI audio."
        )
    )
    parser.add_argument("--model", type=Path, default=default_model, help="ONNX/PT model export to run.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument(
        "--audio-device",
        default=None,
        help=f"AGX audio output device. Default: DISPLAY_AUDIO_DEVICE or {DEFAULT_AUDIO_DEVICE}.",
    )
    parser.add_argument("--dry-run-audio", action="store_true", help="Select the WAV but do not play it.")
    parser.add_argument("--no-audio-delay", action="store_true", help="Skip the cue pre-delay.")
    parser.add_argument("--max-events", type=positive_int_or_none, default=None, help="Stop after N detections. <=0 means forever.")
    parser.add_argument("--max-frames", type=positive_int_or_none, default=None, help="Stop after N depth frames. <=0 means forever.")
    parser.add_argument("--poll-delay-sec", type=float, default=0.0)
    parser.add_argument(
        "--loop-mode",
        choices=("realtime", "depth-trigger"),
        default="realtime",
        help="realtime streams RGB continuously and uses depth as a gate; depth-trigger keeps the older one-shot trigger flow.",
    )
    parser.add_argument("--inference-interval-sec", type=float, default=0.2)
    parser.add_argument("--present-distance-cm", type=float, default=45.0)
    parser.add_argument("--realtime-release-distance-cm", type=float, default=60.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--stable-required", type=int, default=4)
    parser.add_argument("--release-required-frames", type=int, default=3)
    parser.add_argument("--visual-rearm-cooldown-sec", type=float, default=1.0)
    parser.add_argument("--visual-rearm-diff-threshold", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "realsense", "v4l2"),
        default="auto",
        help="Camera backend. auto uses pyrealsense2 only when SDK enumeration works; otherwise V4L2.",
    )
    parser.add_argument("--v4l2-depth-device", default="/dev/video2")
    parser.add_argument("--v4l2-color-device", default="/dev/video6")
    parser.add_argument("--v4l2-depth-scale-m", type=float, default=0.00025)
    parser.add_argument("--v4l2-depth-width", type=int, default=480)
    parser.add_argument("--v4l2-depth-height", type=int, default=640)
    parser.add_argument(
        "--v4l2-color-capture-frames",
        type=int,
        default=30,
        help="Number of V4L2 color frames to stream before using the last one, for auto-exposure warmup.",
    )
    parser.add_argument("--trigger-distance-cm", type=float, default=30.0)
    parser.add_argument("--release-distance-cm", type=float, default=45.0)
    parser.add_argument("--cooldown-sec", type=float, default=2.0)
    parser.add_argument("--required-consecutive-frames", type=int, default=3)
    parser.add_argument(
        "--no-trigger-on-invalid-center",
        action="store_true",
        help="Disable demo fallback that treats invalid center depth as a nearby object.",
    )
    parser.add_argument("--display", action="store_true", help="Start the Display UI and push recognition_result updates into it.")
    parser.add_argument("--display-host", default="0.0.0.0")
    parser.add_argument("--display-port", type=int, default=8080)
    parser.add_argument(
        "--keep-display-open",
        action="store_true",
        help="After max-events completes, keep the display server alive until Ctrl+C.",
    )
    args = parser.parse_args()

    trigger_config = DistanceTriggerConfig(
        trigger_distance_cm=args.trigger_distance_cm,
        release_distance_cm=args.release_distance_cm,
        cooldown_sec=args.cooldown_sec,
        required_consecutive_frames=args.required_consecutive_frames,
    )
    camera_config = L515RGBDCameraConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        timeout_ms=args.timeout_ms,
    )
    v4l2_config = V4L2RGBDCameraConfig(
        depth_device=args.v4l2_depth_device,
        color_device=args.v4l2_color_device,
        depth_width=args.v4l2_depth_width,
        depth_height=args.v4l2_depth_height,
        color_width=args.width,
        color_height=args.height,
        fps=args.fps,
        depth_scale_m=args.v4l2_depth_scale_m,
        warmup_frames=min(args.warmup_frames, 5),
        color_capture_frames=args.v4l2_color_capture_frames,
    )

    def log(message: str) -> None:
        print(message, flush=True)

    q_result: Queue[dict[str, Any]] | None = Queue() if args.display else None
    camera_frame_sink = None
    if q_result is not None:
        _, camera_frames = start_display_bridge(
            q_result,
            host=args.display_host,
            port=args.display_port,
            log=log,
        )
        camera_frame_sink = camera_frames.update_rgb

    monitor_config = RealtimeMonitorConfig(
        inference_interval_sec=args.inference_interval_sec,
        present_distance_cm=args.present_distance_cm,
        release_distance_cm=args.realtime_release_distance_cm,
        confidence_threshold=args.confidence_threshold,
        stable_required=args.stable_required,
        release_required_frames=args.release_required_frames,
        visual_rearm_cooldown_sec=args.visual_rearm_cooldown_sec,
        visual_rearm_diff_threshold=args.visual_rearm_diff_threshold,
        max_events=args.max_events,
        max_frames=args.max_frames,
        poll_delay_sec=args.poll_delay_sec,
    )

    emitted = run_agx_l515_voice_demo(
        model_path=args.model,
        audio_root=args.audio_root,
        audio_device=args.audio_device,
        dry_run_audio=args.dry_run_audio,
        no_audio_delay=args.no_audio_delay,
        snapshot_dir=args.snapshot_dir,
        trigger_config=trigger_config,
        camera_config=camera_config,
        v4l2_config=v4l2_config,
        camera_backend=args.camera_backend,
        q_result=q_result,
        camera_frame_sink=camera_frame_sink,
        loop_mode=args.loop_mode,
        monitor_config=monitor_config,
        max_events=args.max_events,
        max_frames=args.max_frames,
        poll_delay_sec=args.poll_delay_sec,
        trigger_on_invalid_center=not args.no_trigger_on_invalid_center,
        log=log,
    )
    print(f"[agx-demo] emitted_events={emitted}")
    if args.display and args.keep_display_open:
        print("[agx-demo] display server is still running; press Ctrl+C to stop.", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
