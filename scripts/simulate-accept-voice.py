#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
DEFAULT_AUDIO_ROOT = WORKSPACE_ROOT / "display" / "assets" / "audio"
DEFAULT_AUDIO_DEVICE = "alsa_output.platform-3510000.hda.hdmi-stereo"
DEFAULT_AUDIO_DEVICE_ENV = "DISPLAY_AUDIO_DEVICE"
sys.path.insert(0, str(ROOT / "src"))

from voice_feedback import VoiceCueRouter
from vision_contract import build_recognition_result


def local_wav_path(audio_root: Path, cue_audio_path: str) -> Path:
    relative = cue_audio_path.strip().replace("\\", "/")
    prefix = "assets/voice/gpt-sovits/"
    if relative.startswith(prefix):
        relative = relative[len(prefix) :]
    return audio_root / relative


def resolve_audio_device(audio_device: str | None) -> str | None:
    if audio_device is not None:
        return audio_device.strip() or None
    return os.environ.get(DEFAULT_AUDIO_DEVICE_ENV, DEFAULT_AUDIO_DEVICE).strip() or None


def looks_like_pulse_sink(audio_device: str) -> bool:
    return audio_device.startswith(("alsa_output.", "bluez_output.", "auto_null"))


def playable_command(wav_path: Path, *, audio_device: str | None = None) -> list[str]:
    if audio_device:
        if looks_like_pulse_sink(audio_device):
            if shutil.which("paplay"):
                return ["paplay", "--device", audio_device, str(wav_path)]
            if shutil.which("aplay"):
                return ["aplay", "-q", "-D", audio_device, str(wav_path)]
        else:
            if shutil.which("aplay"):
                return ["aplay", "-q", "-D", audio_device, str(wav_path)]
            if shutil.which("paplay"):
                return ["paplay", "--device", audio_device, str(wav_path)]
        raise RuntimeError("找不到支援指定裝置的播放工具，請安裝 paplay 或 aplay。")

    if shutil.which("aplay"):
        return ["aplay", "-q", str(wav_path)]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(wav_path)]
    if shutil.which("play"):
        return ["play", "-q", str(wav_path)]
    if shutil.which("paplay"):
        return ["paplay", str(wav_path)]
    raise RuntimeError("找不到播放工具，請安裝 aplay、ffplay、sox/play 或 paplay。")


def build_accept_cue(seed: int | None = None) -> dict[str, Any]:
    router = VoiceCueRouter(rng=Random(seed) if seed is not None else None)
    result = build_recognition_result(
        predicted_class="accept",
        confidence=0.91,
        snapshot_path="simulate/l515-accept.jpg",
        ts=datetime.now().isoformat(timespec="seconds"),
    )
    return router.route(result).to_payload()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate a vision accept result, randomly choose one recorded positive WAV, and play it locally."
    )
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT, help="Root containing accept/accept-xx.wav")
    parser.add_argument(
        "--audio-device",
        default=None,
        help=(
            "Output device for playback. Default is DISPLAY_AUDIO_DEVICE or "
            f"{DEFAULT_AUDIO_DEVICE}."
        ),
    )
    parser.add_argument("--seed", type=int, help="Optional random seed for repeatable selection")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected cue without playing audio")
    parser.add_argument("--no-delay", action="store_true", help="Skip the cue pre-delay before playback")
    args = parser.parse_args()

    cue = build_accept_cue(seed=args.seed)
    wav_path = local_wav_path(args.audio_root, str(cue["audio_path"]))

    print(f"模擬視覺結果：accept / confidence={cue['source_confidence']}")
    print(f"抽到台詞：{cue['text']}")
    print(f"cue audio_path：{cue['audio_path']}")
    print(f"本機播放檔：{wav_path}")
    audio_device = resolve_audio_device(args.audio_device)
    print(f"播放裝置：{audio_device or 'system default'}")

    if args.dry_run:
        return 0

    if not wav_path.is_file():
        print(f"找不到 WAV：{wav_path}", file=sys.stderr)
        return 1

    if not args.no_delay:
        time.sleep(float(cue.get("pre_delay_sec", 0)))

    subprocess.run(playable_command(wav_path, audio_device=audio_device), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
