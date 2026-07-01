"""
LiveKit Agents TTS plugin for IndicF5 HTTP server.

Copy this file into your LiveKit agents project, or install with optional deps:

    pip install livekit-agents httpx

Usage:

    from integrations.livekit_indicf5 import IndicF5TTS

    session = AgentSession(
        stt=...,
        llm=...,
        tts=IndicF5TTS(base_url="http://127.0.0.1:8000"),
    )

Server endpoint: POST /v1/tts/synthesize
  - JSON: {"text": "..."}  (ref voice uses server defaults)
  - Response: streaming pcm_s16le @ 24 kHz
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

SAMPLE_RATE = 24000
NUM_CHANNELS = 1


class IndicF5TTS(tts.TTS):
    """Stream speech from an IndicF5 FastAPI server (/v1/tts/synthesize)."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        nfe_step: Optional[int] = None,
        split: bool = False,
        max_chars: int = 120,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._base_url = base_url.rstrip("/")
        self._nfe_step = nfe_step
        self._split = split
        self._max_chars = max_chars
        self._timeout = timeout

    @property
    def model(self) -> str:
        return "indicf5"

    @property
    def provider(self) -> str:
        return "indicf5"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return IndicF5ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )


class IndicF5ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: IndicF5TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: IndicF5TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = f"livekit-{id(self)}"
        payload: dict = {
            "text": self._input_text,
            "split": self._tts._split,
            "max_chars": self._tts._max_chars,
        }
        if self._tts._nfe_step is not None:
            payload["nfe-step"] = self._tts._nfe_step

        timeout = httpx.Timeout(self._tts._timeout, connect=10.0)
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        try:
            async with httpx.AsyncClient(base_url=self._tts._base_url, timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    "/v1/tts/synthesize",
                    json=payload,
                    headers={"X-Request-Id": request_id},
                ) as response:
                    response.raise_for_status()
                    async for pcm_bytes in response.aiter_bytes():
                        if pcm_bytes:
                            output_emitter.push(pcm_bytes)
                        await asyncio.sleep(0)
        except asyncio.CancelledError:
            # LiveKit barge-in: httpx stream context closes → server sees disconnect
            raise
        finally:
            output_emitter.flush()
