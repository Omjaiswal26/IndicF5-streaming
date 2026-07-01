"""FastAPI server for true chunk-level IndicF5 streaming."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from indicf5_streaming import IndicF5Session, stream_audio_bytes

WEB_DIR = Path(__file__).parent / "web"

DEFAULT_NFE_STEP = 32
SAMPLE_RATE = 24000

app = FastAPI(title="IndicF5 Streaming TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
session: Optional[IndicF5Session] = None
default_nfe_step = DEFAULT_NFE_STEP


class TTSRequest(BaseModel):
    text: str
    ref_audio_path: str
    ref_text: str
    max_chars: int = 120
    nfe_step: int = Field(default=DEFAULT_NFE_STEP, alias="nfe-step")

    model_config = {"populate_by_name": True}


def _resolve_nfe_step(query_nfe: Optional[int], body_nfe: Optional[int] = None) -> int:
    if query_nfe is not None:
        return query_nfe
    if body_nfe is not None:
        return body_nfe
    return default_nfe_step


def _stream_response_headers(nfe_step: int) -> dict[str, str]:
    return {
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-NFE-Step": str(nfe_step),
        "X-Chunk-Format": "uint32-le-length-prefix + float32-le-pcm",
    }


def _iter_stream_bytes(
    text: str,
    ref_audio_path: str,
    ref_text: str,
    max_chars: int,
    nfe_step: int,
):
    assert session is not None
    chunks = session.stream(
        text=text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        max_chars=max_chars,
        nfe_step=nfe_step,
    )
    yield from stream_audio_bytes(chunks)


@app.get("/health")
def health():
    return {"status": "ok", "sample_rate": SAMPLE_RATE, "default_nfe_step": default_nfe_step}


@app.post("/tts/stream")
def tts_stream_post(
    body: TTSRequest,
    nfe_step: Optional[int] = Query(None, alias="nfe-step"),
):
    """Stream TTS audio chunks as each is synthesized (binary framed float32 PCM)."""
    resolved_nfe = _resolve_nfe_step(nfe_step, body.nfe_step)

    def generate():
        yield from _iter_stream_bytes(
            body.text,
            body.ref_audio_path,
            body.ref_text,
            body.max_chars,
            resolved_nfe,
        )

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers=_stream_response_headers(resolved_nfe),
    )


@app.get("/tts/stream")
def tts_stream_get(
    text: str,
    ref_audio_path: str,
    ref_text: str,
    max_chars: int = 120,
    nfe_step: Optional[int] = Query(None, alias="nfe-step"),
):
    resolved_nfe = _resolve_nfe_step(nfe_step)

    def generate():
        yield from _iter_stream_bytes(
            text,
            ref_audio_path,
            ref_text,
            max_chars,
            resolved_nfe,
        )

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers=_stream_response_headers(resolved_nfe),
    )


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main():
    global session, default_nfe_step

    parser = argparse.ArgumentParser(description="IndicF5 chunk-level streaming TTS server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default="ai4bharat/IndicF5")
    parser.add_argument(
        "--nfe-step",
        type=int,
        default=DEFAULT_NFE_STEP,
        help="Default NFE steps for CFM sampling (override per request via ?nfe-step=)",
    )
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detect if omitted)")
    args = parser.parse_args()

    default_nfe_step = args.nfe_step
    session = IndicF5Session(model_id=args.model_id, device=args.device)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
