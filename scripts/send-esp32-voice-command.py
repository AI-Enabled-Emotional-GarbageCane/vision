#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esp32_serial import Esp32SerialVoiceSink, encode_esp32_voice_command


def load_cue(args: argparse.Namespace) -> dict[str, object]:
    if args.category and args.audio_path:
        return {
            "category": args.category,
            "audio_path": args.audio_path,
            "pre_sfx": args.pre_sfx,
            "pre_delay_sec": args.pre_delay_ms / 1000,
        }

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a voice_feedback_cue to the ESP32-S3 voice player.")
    parser.add_argument("--port", required=True, help="Serial port, for example /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--file", help="JSON file containing a voice_feedback_cue; defaults to stdin")
    parser.add_argument("--category", help="Direct category for manual testing")
    parser.add_argument("--audio-path", help="Direct WAV path for manual testing, e.g. reject/reject-01.wav")
    parser.add_argument("--pre-sfx", default="ding")
    parser.add_argument("--pre-delay-ms", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Print the serial JSON without opening the port")
    args = parser.parse_args()

    cue = load_cue(args)
    payload = encode_esp32_voice_command(cue)
    if args.dry_run:
        print(payload.decode("utf-8"), end="")
        return

    ack = Esp32SerialVoiceSink(args.port, baudrate=args.baudrate).send(cue)
    if ack:
        print(ack)


if __name__ == "__main__":
    main()
