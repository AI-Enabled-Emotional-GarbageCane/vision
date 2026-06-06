from __future__ import annotations

from vision_contract import build_recognition_result
from voice_feedback import (
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE_LINES,
    PRE_ROAST_DELAY_SEC,
    VoiceCueRouter,
)


class FirstChoice:
    def choice(self, seq):
        return seq[0]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def result(predicted_class: str, confidence: float, ts: str, *, num_objects: int = 1) -> dict[str, object]:
    payload = build_recognition_result(
        predicted_class=predicted_class,
        confidence=confidence,
        snapshot_path="snapshots/test.jpg",
        ts=ts,
    )
    payload["num_objects"] = num_objects
    return payload


def main() -> None:
    router = VoiceCueRouter(rng=FirstChoice())
    require(len(DEFAULT_VOICE_LINES["reject"]) == 30, "reject voice library should include 30 recorded lines")

    accept = router.route(result("accept", 0.91, "2026-06-04T12:00:00")).to_payload()
    require(accept["event"] == "voice_feedback_cue", "voice cue event name drifted")
    require(accept["category"] == "accept", "high-confidence accept should use accept voice")
    require(accept["level"] == "light", "accept voice should be light")
    require(accept["tts_model"] == DEFAULT_TTS_MODEL, "voice cue should declare GPT-SoVITS")
    require(accept["pre_delay_sec"] == PRE_ROAST_DELAY_SEC, "voice cue should preserve roast delay")
    require(
        str(accept["audio_path"]).endswith("assets/voice/gpt-sovits/accept/accept-01.wav"),
        "accept should point at the pre-generated voice asset",
    )

    low = router.route(result("reject", 0.42, "2026-06-04T12:00:10")).to_payload()
    require(low["category"] == "low_confidence", "low confidence should use self-mock voice")
    require(low["level"] == "self-mock", "low confidence should be self-mock level")

    first_reject = router.route(result("reject", 0.88, "2026-06-04T12:00:20")).to_payload()
    second_reject = router.route(result("reject", 0.86, "2026-06-04T12:00:40")).to_payload()
    require(first_reject["category"] == "reject", "first confident reject should be normal reject")
    require(
        str(first_reject["audio_path"]).endswith("assets/voice/gpt-sovits/reject/reject-01.wav"),
        "reject should point at the recorded reject voice assets",
    )
    require(second_reject["category"] == "reject", "every confident reject should use the recorded reject pool")
    require(second_reject["level"] == "medium", "repeat rejects should not escalate away from the 30-line pool")
    require(
        str(second_reject["audio_path"]).endswith("assets/voice/gpt-sovits/reject/reject-01.wav"),
        "repeat rejects should still point at recorded reject voice assets",
    )

    late_reject = router.route(result("reject", 0.93, "2026-06-04T12:02:00")).to_payload()
    require(late_reject["category"] == "reject", "later rejects should also use normal reject")

    multi = router.route(result("accept", 0.8, "2026-06-04T12:02:10", num_objects=2)).to_payload()
    require(multi["category"] == "multi_object", "multi-object payload should use multi-object voice")
    require(multi["source_class"] == "accept", "voice cue should preserve source class for audit")

    print("[OK] voice feedback router maps recognition_result payloads to GPT-SoVITS cues")


if __name__ == "__main__":
    main()
