from __future__ import annotations

import json

from esp32_serial import build_esp32_voice_command, encode_esp32_voice_command


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    cue = {
        "event": "voice_feedback_cue",
        "category": "reject",
        "audio_path": "assets/voice/gpt-sovits/reject/reject-01.wav",
        "pre_sfx": "ding",
        "pre_delay_sec": 0.5,
    }

    command = build_esp32_voice_command(cue)
    require(command["category"] == "reject", "category should be preserved")
    require(command["audio_path"] == "reject/reject-01.wav", "AGX asset prefix should be stripped for SD card")
    require(command["pre_sfx"] == "ding", "pre_sfx should be preserved")
    require(command["pre_delay_ms"] == 500, "pre_delay should be converted to milliseconds")

    encoded = encode_esp32_voice_command(cue)
    require(encoded.endswith(b"\n"), "serial command must be newline-delimited")
    decoded = json.loads(encoded.decode("utf-8"))
    require(decoded == command, "encoded command should round-trip as JSON")

    for bad in [
        {**cue, "category": "unknown"},
        {**cue, "audio_path": "../reject/reject-01.wav"},
        {**cue, "audio_path": "reject/reject-01.mp3"},
    ]:
        try:
            build_esp32_voice_command(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad ESP32 command should be rejected: {bad}")

    print("[OK] ESP32 serial command builder creates safe newline-delimited playback JSON")


if __name__ == "__main__":
    main()
