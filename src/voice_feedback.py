from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from random import Random
from typing import Any, Mapping, Protocol


LOW_CONFIDENCE_THRESHOLD = 0.5
PRE_ROAST_DELAY_SEC = 0.5
DEFAULT_TTS_MODEL = "GPT-SoVITS"
DEFAULT_VOICE_ASSET_ROOT = "assets/voice/gpt-sovits"


class ChoiceLike(Protocol):
    def choice(self, seq: tuple["VoiceLine", ...]) -> "VoiceLine": ...


@dataclass(frozen=True)
class VoiceLine:
    id: str
    category: str
    level: str
    text: str
    relative_audio_path: str


@dataclass(frozen=True)
class VoiceCue:
    category: str
    level: str
    text: str
    audio_path: str
    source_class: str
    source_confidence: float
    source_ts: str
    tts_model: str = DEFAULT_TTS_MODEL
    pre_sfx: str = "ding"
    pre_delay_sec: float = PRE_ROAST_DELAY_SEC
    event: str = "voice_feedback_cue"

    def to_payload(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "category": self.category,
            "level": self.level,
            "text": self.text,
            "audio_path": self.audio_path,
            "source_class": self.source_class,
            "source_confidence": self.source_confidence,
            "source_ts": self.source_ts,
            "tts_model": self.tts_model,
            "pre_sfx": self.pre_sfx,
            "pre_delay_sec": self.pre_delay_sec,
        }


DEFAULT_VOICE_LINES: dict[str, tuple[VoiceLine, ...]] = {
    "accept": (
        VoiceLine("accept-01", "accept", "light", "做得好，這個分類很漂亮。", "accept/accept-01.wav"),
        VoiceLine("accept-02", "accept", "light", "你丟對了，地球幫你按讚。", "accept/accept-02.wav"),
        VoiceLine("accept-03", "accept", "light", "完美命中，這就是正確桶。", "accept/accept-03.wav"),
        VoiceLine("accept-04", "accept", "light", "這次很準，謝謝你的用心。", "accept/accept-04.wav"),
        VoiceLine("accept-05", "accept", "light", "分類成功，今天的環境更乾淨了。", "accept/accept-05.wav"),
        VoiceLine("accept-06", "accept", "light", "好眼力，這個可以放心丟。", "accept/accept-06.wav"),
        VoiceLine("accept-07", "accept", "light", "很棒，垃圾也找到正確的家。", "accept/accept-07.wav"),
        VoiceLine("accept-08", "accept", "light", "你這一丟，回收流程順多了。", "accept/accept-08.wav"),
        VoiceLine("accept-09", "accept", "light", "正確分類，漂亮的一步。", "accept/accept-09.wav"),
        VoiceLine("accept-10", "accept", "light", "這個判斷很可以，繼續保持。", "accept/accept-10.wav"),
        VoiceLine("accept-11", "accept", "light", "謝謝你照顧地球的一小步。", "accept/accept-11.wav"),
        VoiceLine("accept-12", "accept", "light", "丟得精準，分類也到位。", "accept/accept-12.wav"),
        VoiceLine("accept-13", "accept", "light", "做得漂亮，這桶收下了。", "accept/accept-13.wav"),
        VoiceLine("accept-14", "accept", "light", "好選擇，這就是該去的地方。", "accept/accept-14.wav"),
        VoiceLine("accept-15", "accept", "light", "你丟對了，今天加一分。", "accept/accept-15.wav"),
        VoiceLine("accept-16", "accept", "light", "清楚又俐落，分類完成。", "accept/accept-16.wav"),
        VoiceLine("accept-17", "accept", "light", "這次很專業，我喜歡。", "accept/accept-17.wav"),
        VoiceLine("accept-18", "accept", "light", "正確投入，環境感謝你。", "accept/accept-18.wav"),
        VoiceLine("accept-19", "accept", "light", "小小動作，大大加分。", "accept/accept-19.wav"),
        VoiceLine("accept-20", "accept", "light", "完成得很好，下一件也照這樣。", "accept/accept-20.wav"),
    ),
    "reject": (
        VoiceLine("reject-01", "reject", "medium", "垃圾桶都標字了，你還能猜錯。", "reject/reject-01.wav"),
        VoiceLine("reject-02", "reject", "medium", "分類不是占卜，不用靠直覺。", "reject/reject-02.wav"),
        VoiceLine("reject-03", "reject", "medium", "可回收三個字，真的很難嗎？", "reject/reject-03.wav"),
        VoiceLine("reject-04", "reject", "medium", "你丟垃圾的手法，很有創意災難。", "reject/reject-04.wav"),
        VoiceLine("reject-05", "reject", "medium", "垃圾分類，不是給垃圾看的。", "reject/reject-05.wav"),
        VoiceLine("reject-06", "reject", "medium", "這不是投籃，是分類。", "reject/reject-06.wav"),
        VoiceLine("reject-07", "reject", "medium", "你跟標示牌是不是有仇？", "reject/reject-07.wav"),
        VoiceLine("reject-08", "reject", "medium", "地球看到你，應該有點累。", "reject/reject-08.wav"),
        VoiceLine("reject-09", "reject", "medium", "連垃圾都想回家，你卻送錯站。", "reject/reject-09.wav"),
        VoiceLine("reject-10", "reject", "medium", "你的分類方式，像在抽籤。", "reject/reject-10.wav"),
        VoiceLine("reject-11", "reject", "medium", "這桶不是萬用許願池。", "reject/reject-11.wav"),
        VoiceLine("reject-12", "reject", "medium", "廚餘不是紙類，謝謝你的想像力。", "reject/reject-12.wav"),
        VoiceLine("reject-13", "reject", "medium", "瓶罐有家，不要亂寄宿。", "reject/reject-13.wav"),
        VoiceLine("reject-14", "reject", "medium", "你丟錯的瞬間，地球沉默了。", "reject/reject-14.wav"),
        VoiceLine("reject-15", "reject", "medium", "分類桶不是裝飾品，可以看一下。", "reject/reject-15.wav"),
        VoiceLine("reject-16", "reject", "medium", "垃圾都比你更想被分類。", "reject/reject-16.wav"),
        VoiceLine("reject-17", "reject", "medium", "你這一丟，回收員開始懷疑人生。", "reject/reject-17.wav"),
        VoiceLine("reject-18", "reject", "medium", "標籤那麼大，你還能無視。", "reject/reject-18.wav"),
        VoiceLine("reject-19", "reject", "medium", "你的垃圾分類，主打一個自由奔放。", "reject/reject-19.wav"),
        VoiceLine("reject-20", "reject", "medium", "不會分類沒關係，會看字就好。", "reject/reject-20.wav"),
        VoiceLine("reject-21", "reject", "medium", "這不是大雜燴，請尊重垃圾。", "reject/reject-21.wav"),
        VoiceLine("reject-22", "reject", "medium", "你丟得很快，但錯得很穩。", "reject/reject-22.wav"),
        VoiceLine("reject-23", "reject", "medium", "回收桶不是什麼都回收你。", "reject/reject-23.wav"),
        VoiceLine("reject-24", "reject", "medium", "垃圾分類考試，你可能要補考。", "reject/reject-24.wav"),
        VoiceLine("reject-25", "reject", "medium", "你跟正確桶之間，只差一眼。", "reject/reject-25.wav"),
        VoiceLine("reject-26", "reject", "medium", "這操作，連垃圾都想抗議。", "reject/reject-26.wav"),
        VoiceLine("reject-27", "reject", "medium", "分類很簡單，真的不用開天眼。", "reject/reject-27.wav"),
        VoiceLine("reject-28", "reject", "medium", "你丟錯桶的自信，令人佩服。", "reject/reject-28.wav"),
        VoiceLine("reject-29", "reject", "medium", "這不是隨機模式，是垃圾分類。", "reject/reject-29.wav"),
        VoiceLine("reject-30", "reject", "medium", "請給垃圾一個正確的歸宿。", "reject/reject-30.wav"),
    ),
    "repeat_reject": (
        VoiceLine("repeat-01", "repeat_reject", "heavy", "我已經第二次提醒你了。", "repeat_reject/repeat-01.wav"),
        VoiceLine("repeat-02", "repeat_reject", "heavy", "同一件事不要讓我講第三次。", "repeat_reject/repeat-02.wav"),
        VoiceLine("repeat-03", "repeat_reject", "heavy", "你跟分類規則是不是不熟。", "repeat_reject/repeat-03.wav"),
    ),
    "multi_object": (
        VoiceLine("multi-01", "multi_object", "medium-heavy", "一次丟一堆，分一下好嗎。", "multi_object/multi-01.wav"),
        VoiceLine("multi-02", "multi_object", "medium-heavy", "慢慢來，不用整包倒進來。", "multi_object/multi-02.wav"),
        VoiceLine("multi-03", "multi_object", "medium-heavy", "我一次只看一個，謝謝配合。", "multi_object/multi-03.wav"),
    ),
    "low_confidence": (
        VoiceLine("low-01", "low_confidence", "self-mock", "我看不太出來欸，可能是我老花。", "low_confidence/low-01.wav"),
        VoiceLine("low-02", "low_confidence", "self-mock", "這個角度太狠了，我先保留。", "low_confidence/low-02.wav"),
        VoiceLine("low-03", "low_confidence", "self-mock", "我有點猶豫，先不要亂判。", "low_confidence/low-03.wav"),
    ),
}


def _parse_ts(ts: str) -> datetime:
    normalized = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _audio_path(asset_root: str, line: VoiceLine) -> str:
    return f"{asset_root.rstrip('/')}/{line.relative_audio_path}"


class VoiceCueRouter:
    """Maps recognition_result payloads to GPT-SoVITS voice cue payloads.

    This adapter intentionally does not play audio. It prepares a downstream
    cue so display/speaker code can play pre-generated voice-clone assets.
    """

    def __init__(
        self,
        *,
        voice_lines: Mapping[str, tuple[VoiceLine, ...]] = DEFAULT_VOICE_LINES,
        asset_root: str = DEFAULT_VOICE_ASSET_ROOT,
        rng: ChoiceLike | None = None,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        if not 0 <= low_confidence_threshold <= 1:
            raise ValueError(f"low_confidence_threshold out of range: {low_confidence_threshold}")

        self.voice_lines = voice_lines
        self.asset_root = asset_root
        self.rng = rng or Random()
        self.low_confidence_threshold = low_confidence_threshold

    def route(self, recognition_result: Mapping[str, Any]) -> VoiceCue:
        if recognition_result.get("event") != "recognition_result":
            raise ValueError("voice cue routing requires a recognition_result payload")

        source_class = str(recognition_result.get("class"))
        confidence = float(recognition_result.get("confidence", -1))
        num_objects = int(recognition_result.get("num_objects", 1))
        source_ts = str(recognition_result.get("ts", ""))
        if source_class not in {"accept", "reject"}:
            raise ValueError(f"invalid source class for voice cue: {source_class}")
        if not 0 <= confidence <= 1:
            raise ValueError(f"confidence out of range for voice cue: {confidence}")
        if not source_ts:
            raise ValueError("recognition_result ts is required for voice cue routing")

        _parse_ts(source_ts)
        category = self._category(source_class, confidence, num_objects)
        line = self.rng.choice(self.voice_lines[category])

        return VoiceCue(
            category=category,
            level=line.level,
            text=line.text,
            audio_path=_audio_path(self.asset_root, line),
            source_class=source_class,
            source_confidence=confidence,
            source_ts=source_ts,
        )

    def _category(self, source_class: str, confidence: float, num_objects: int) -> str:
        if confidence < self.low_confidence_threshold:
            return "low_confidence"
        if num_objects > 1:
            return "multi_object"
        if source_class == "accept":
            return "accept"
        return "reject"
