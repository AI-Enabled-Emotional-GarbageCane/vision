from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import agx_audio
from agx_audio import AgxWavVoiceSink, local_wav_path, playable_command


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with TemporaryDirectory() as tmp:
        audio_root = Path(tmp)
        wav_path = audio_root / "accept" / "accept-01.wav"
        wav_path.parent.mkdir(parents=True)
        wav_path.write_bytes(b"RIFF....WAVE")

        resolved = local_wav_path(audio_root, "assets/voice/gpt-sovits/accept/accept-01.wav")
        require(resolved == wav_path, "voice asset path should map to local display audio root")

        logs: list[str] = []
        sink = AgxWavVoiceSink(audio_root=audio_root, dry_run=True, no_delay=True, log=logs.append)
        status = sink.play(
            {
                "category": "accept",
                "text": "ok",
                "audio_path": "assets/voice/gpt-sovits/accept/accept-01.wav",
                "pre_delay_sec": 0.5,
            }
        )
        require(status["played"] is False, "dry-run should not mark audio as played")
        require(status["reason"] == "dry_run", "dry-run should explain that playback was skipped")
        require(any("accept" in message for message in logs), "sink should log the selected cue")

    original_which = agx_audio.shutil.which
    agx_audio.shutil.which = lambda executable: f"/usr/bin/{executable}" if executable == "paplay" else None
    try:
        command = playable_command(Path("/tmp/test.wav"), audio_device="alsa_output.platform-3510000.hda.hdmi-stereo")
    finally:
        agx_audio.shutil.which = original_which

    require(
        command == [
            "/usr/bin/paplay",
            "--device",
            "alsa_output.platform-3510000.hda.hdmi-stereo",
            "/tmp/test.wav",
        ],
        "pulse HDMI sink should use paplay --device",
    )

    print("[OK] AGX audio sink maps voice cues to HDMI WAV playback commands")


if __name__ == "__main__":
    main()
