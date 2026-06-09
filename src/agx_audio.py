from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


VISION_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = VISION_ROOT.parent
DEFAULT_AUDIO_ROOT = WORKSPACE_ROOT / "display" / "assets" / "audio"
DEFAULT_AUDIO_DEVICE = "alsa_output.platform-3510000.hda.hdmi-stereo"
DEFAULT_AUDIO_DEVICE_ENV = "DISPLAY_AUDIO_DEVICE"
VOICE_ASSET_PREFIX = "assets/voice/gpt-sovits/"


LogSink = Callable[[str], None]


def local_wav_path(audio_root: Path, cue_audio_path: str) -> Path:
    relative = cue_audio_path.strip().replace("\\", "/")
    if relative.startswith(VOICE_ASSET_PREFIX):
        relative = relative[len(VOICE_ASSET_PREFIX) :]
    while relative.startswith("/"):
        relative = relative[1:]
    if not relative.endswith(".wav"):
        raise ValueError(f"voice cue audio_path must be a WAV file: {cue_audio_path}")
    if ".." in relative.split("/"):
        raise ValueError(f"voice cue audio_path must not traverse directories: {cue_audio_path}")
    return audio_root / relative


def resolve_audio_device(audio_device: str | None) -> str | None:
    if audio_device is not None:
        return audio_device.strip() or None
    return os.environ.get(DEFAULT_AUDIO_DEVICE_ENV, DEFAULT_AUDIO_DEVICE).strip() or None


def looks_like_pulse_sink(audio_device: str) -> bool:
    return audio_device.startswith(("alsa_output.", "bluez_output.", "auto_null"))


def playable_command(wav_path: Path, *, audio_device: str | None = None) -> list[str] | None:
    if audio_device:
        if looks_like_pulse_sink(audio_device):
            players = (
                ("paplay", ["--device", audio_device, str(wav_path)]),
                ("aplay", ["-q", "-D", audio_device, str(wav_path)]),
            )
        else:
            players = (
                ("aplay", ["-q", "-D", audio_device, str(wav_path)]),
                ("paplay", ["--device", audio_device, str(wav_path)]),
            )
        for executable, args in players:
            player_path = shutil.which(executable)
            if player_path:
                return [player_path, *args]
        return None

    players = (
        ("paplay", [str(wav_path)]),
        ("aplay", ["-q", str(wav_path)]),
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "error", str(wav_path)]),
        ("play", ["-q", str(wav_path)]),
    )
    for executable, args in players:
        player_path = shutil.which(executable)
        if player_path:
            return [player_path, *args]
    return None


@dataclass
class AgxWavVoiceSink:
    audio_root: Path = DEFAULT_AUDIO_ROOT
    audio_device: str | None = None
    dry_run: bool = False
    no_delay: bool = False
    log: LogSink | None = print

    def __post_init__(self) -> None:
        self.audio_root = Path(self.audio_root)
        self.audio_device = resolve_audio_device(self.audio_device)

    def __call__(self, voice_cue: Mapping[str, Any]) -> None:
        self.play(voice_cue)

    def play(self, voice_cue: Mapping[str, Any]) -> dict[str, Any]:
        wav_path = local_wav_path(self.audio_root, str(voice_cue.get("audio_path", "")))
        status: dict[str, Any] = {
            "category": str(voice_cue.get("category", "")),
            "text": str(voice_cue.get("text", "")),
            "path": str(wav_path),
            "audio_device": self.audio_device,
            "played": False,
        }

        self._log(
            "[agx-audio] cue category={category} text={text} wav={path}".format(
                category=status["category"],
                text=status["text"],
                path=status["path"],
            )
        )

        if not wav_path.is_file():
            status["reason"] = "missing_wav"
            raise FileNotFoundError(f"missing WAV for voice cue: {wav_path}")

        if self.dry_run:
            status["reason"] = "dry_run"
            self._log(f"[agx-audio] dry-run; would play on {self.audio_device or 'system default'}")
            return status

        command = playable_command(wav_path, audio_device=self.audio_device)
        if command is None:
            status["reason"] = "no_audio_player"
            raise RuntimeError("no host audio player found; install paplay, aplay, ffplay, or play")

        if not self.no_delay:
            time.sleep(float(voice_cue.get("pre_delay_sec", 0)))

        subprocess.run(command, check=True)
        status["played"] = True
        status["player"] = Path(command[0]).name
        return status

    def _log(self, message: str) -> None:
        if self.log is not None:
            self.log(message)
