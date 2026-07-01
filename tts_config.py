"""Server defaults for TTS (env, CLI, LiveKit text-only requests)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_REF_TEXT = (
    "लहरों से डर कर नौका पार नहीं होती, कोशिश करने वालों की कभी हार नहीं होती। "
    "नन्हीं चींटी जब दाना लेकर चलती है, चढ़ती दीवारों पर सौ बार फिसलती है।"
)


@dataclass
class TTSDefaults:
    ref_audio_path: str = "reference/amitabh_voice.wav"
    ref_text: str = DEFAULT_REF_TEXT
    nfe_step: int = 16
    max_chars: int = 120

    @classmethod
    def from_env(cls) -> "TTSDefaults":
        return cls(
            ref_audio_path=os.getenv("INDICF5_REF_AUDIO", cls.ref_audio_path),
            ref_text=os.getenv("INDICF5_REF_TEXT", DEFAULT_REF_TEXT),
            nfe_step=int(os.getenv("INDICF5_NFE_STEP", str(cls.nfe_step))),
            max_chars=int(os.getenv("INDICF5_MAX_CHARS", str(cls.max_chars))),
        )


server_defaults = TTSDefaults.from_env()


def resolve_ref(ref_audio_path: str | None, ref_text: str | None) -> tuple[str, str]:
    audio = ref_audio_path or server_defaults.ref_audio_path
    text = ref_text if ref_text is not None else server_defaults.ref_text
    return audio, text
