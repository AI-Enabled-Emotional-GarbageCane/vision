from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


ALLOWED_CATEGORIES = {"accept", "reject", "repeat_reject", "low_confidence", "multi_object"}
VOICE_ASSET_PREFIX = "assets/voice/gpt-sovits/"
DEFAULT_BAUDRATE = 115200


def normalize_esp32_audio_path(audio_path: str) -> str:
    normalized = audio_path.strip().replace("\\", "/")
    if normalized.startswith(VOICE_ASSET_PREFIX):
        normalized = normalized[len(VOICE_ASSET_PREFIX) :]
    while normalized.startswith("/"):
        normalized = normalized[1:]
    if not normalized.endswith(".wav"):
        raise ValueError(f"ESP32 audio_path must be a .wav file: {audio_path}")
    if ".." in normalized.split("/"):
        raise ValueError(f"ESP32 audio_path must not traverse directories: {audio_path}")
    return normalized


def build_esp32_voice_command(voice_cue: Mapping[str, Any]) -> dict[str, Any]:
    category = str(voice_cue.get("category", ""))
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"invalid ESP32 voice category: {category}")

    audio_path = normalize_esp32_audio_path(str(voice_cue.get("audio_path", "")))
    pre_sfx = str(voice_cue.get("pre_sfx", "ding"))
    pre_delay_sec = float(voice_cue.get("pre_delay_sec", 0.5))
    if pre_delay_sec < 0:
        raise ValueError(f"pre_delay_sec out of range: {pre_delay_sec}")

    return {
        "category": category,
        "audio_path": audio_path,
        "pre_sfx": pre_sfx,
        "pre_delay_ms": int(round(pre_delay_sec * 1000)),
    }


def encode_esp32_voice_command(voice_cue: Mapping[str, Any]) -> bytes:
    command = build_esp32_voice_command(voice_cue)
    return (json.dumps(command, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")


@dataclass
class Esp32SerialVoiceSink:
    port: str
    baudrate: int = DEFAULT_BAUDRATE
    timeout_sec: float = 1.0

    def send(self, voice_cue: Mapping[str, Any]) -> str:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required to send ESP32 voice commands") from exc

        payload = encode_esp32_voice_command(voice_cue)
        with serial.Serial(self.port, self.baudrate, timeout=self.timeout_sec) as connection:
            connection.write(payload)
            connection.flush()
            ack = connection.readline().decode("utf-8", errors="replace").strip()
        return ack

    def __call__(self, voice_cue: Mapping[str, Any]) -> None:
        self.send(voice_cue)
