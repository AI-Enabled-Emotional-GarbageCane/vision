#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "src"))

from voice_feedback import VoiceCueRouter
from vision_contract import build_recognition_result


def local_wav_path(audio_root: Path, cue_audio_path: str) -> Path:
    relative = cue_audio_path.strip().replace("\\", "/")
    prefix = "assets/voice/gpt-sovits/"
    if relative.startswith(prefix):
        relative = relative[len(prefix) :]
    return audio_root / relative


def playable_command(wav_path: Path) -> list[str]:
    if shutil.which("aplay"):
        return ["aplay", "-q", str(wav_path)]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(wav_path)]
    if shutil.which("play"):
        return ["play", "-q", str(wav_path)]
    if shutil.which("paplay"):
        return ["paplay", str(wav_path)]
    raise RuntimeError("找不到播放工具，請安裝 aplay、ffplay、sox/play 或 paplay。")


def build_reject_cue(seed: int | None = None) -> dict[str, Any]:
    router = VoiceCueRouter(rng=Random(seed) if seed is not None else None)
    result = build_recognition_result(
        predicted_class="reject",
        confidence=0.88,
        snapshot_path="simulate/l515-reject.jpg",
        ts=datetime.now().isoformat(timespec="seconds"),
    )
    return router.route(result).to_payload()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate a vision reject result, randomly choose one recorded reject WAV, and play it locally."
    )
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT, help="Root containing reject/reject-xx.wav")
    parser.add_argument("--seed", type=int, help="Optional random seed for repeatable selection")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected cue without playing audio")
    parser.add_argument("--no-delay", action="store_true", help="Skip the cue pre-delay before playback")
    args = parser.parse_args()

    cue = build_reject_cue(seed=args.seed)
    wav_path = local_wav_path(args.audio_root, str(cue["audio_path"]))
    if not wav_path.is_file():
        print(f"找不到 WAV：{wav_path}", file=sys.stderr)
        return 1

    print(f"模擬視覺結果：reject / confidence={cue['source_confidence']}")
    print(f"抽到台詞：{cue['text']}")
    print(f"cue audio_path：{cue['audio_path']}")
    print(f"本機播放檔：{wav_path}")

    if args.dry_run:
        return 0

    if not args.no_delay:
        time.sleep(float(cue.get("pre_delay_sec", 0)))

    subprocess.run(playable_command(wav_path), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
