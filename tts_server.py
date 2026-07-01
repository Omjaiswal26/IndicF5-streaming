"""FastAPI server for true chunk-level IndicF5 streaming."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from indicf5_streaming import IndicF5Session

WEB_DIR = Path(__file__).parent / "web"

DEFAULT_NFE_STEP = 16
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


def _log_metric(message: str) -> None:
    print(f"[metrics] {message}", file=sys.stderr, flush=True)


def _stream_response_headers(nfe_step: int, chunk_count: Optional[int] = None) -> dict[str, str]:
    headers = {
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-NFE-Step": str(nfe_step),
        "X-Chunk-Format": "uint32-le-length-prefix + float32-le-pcm",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if chunk_count is not None:
        headers["X-Chunk-Count"] = str(chunk_count)
    return headers


def _iter_stream_bytes(
    text: str,
    ref_audio_path: str,
    ref_text: str,
    max_chars: int,
    nfe_step: int,
    request_id: str,
):
    assert session is not None
    req_t0 = time.perf_counter()
    first_byte_logged = False

    chunks = session.stream(
        text=text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        max_chars=max_chars,
        nfe_step=nfe_step,
    )

    import struct

    for chunk in chunks:
        if not first_byte_logged:
            ttft = time.perf_counter() - req_t0
            _log_metric(
                f"request_id={request_id} ttft={ttft:.3f}s "
                f"chunk=1/{chunk.total} nfe_step={nfe_step} chars={len(chunk.text)}"
            )
            first_byte_logged = True

        samples = chunk.audio.astype("float32", copy=False)
        payload = struct.pack("<I", len(samples)) + samples.tobytes()
        _log_metric(
            f"request_id={request_id} chunk={chunk.index + 1}/{chunk.total} "
            f"gen={chunk.gen_seconds:.3f}s cfm={chunk.cfm_seconds:.3f}s "
            f"voc={chunk.vocoder_seconds:.3f}s bytes={len(payload)} "
            f"audio_s={len(samples) / SAMPLE_RATE:.2f}"
        )
        yield payload

    total_s = time.perf_counter() - req_t0
    _log_metric(f"request_id={request_id} stream_complete total={total_s:.3f}s")


@app.get("/health")
def health():
    return {"status": "ok", "sample_rate": SAMPLE_RATE, "default_nfe_step": default_nfe_step}


@app.post("/tts/stream")
def tts_stream_post(
    request: Request,
    body: TTSRequest,
    nfe_step: Optional[int] = Query(None, alias="nfe-step"),
):
    """Stream TTS audio chunks as each is synthesized (binary framed float32 PCM)."""
    resolved_nfe = _resolve_nfe_step(nfe_step, body.nfe_step)
    request_id = request.headers.get("x-request-id", f"req-{int(time.time() * 1000)}")

    def generate():
        yield from _iter_stream_bytes(
            body.text,
            body.ref_audio_path,
            body.ref_text,
            body.max_chars,
            resolved_nfe,
            request_id,
        )

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers=_stream_response_headers(resolved_nfe),
    )


@app.get("/tts/stream")
def tts_stream_get(
    request: Request,
    text: str,
    ref_audio_path: str,
    ref_text: str,
    max_chars: int = 120,
    nfe_step: Optional[int] = Query(None, alias="nfe-step"),
):
    resolved_nfe = _resolve_nfe_step(nfe_step)
    request_id = request.headers.get("x-request-id", f"req-{int(time.time() * 1000)}")

    def generate():
        yield from _iter_stream_bytes(
            text,
            ref_audio_path,
            ref_text,
            max_chars,
            resolved_nfe,
            request_id,
        )

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers=_stream_response_headers(resolved_nfe),
    )


@app.post("/tts/benchmark")
def tts_benchmark(body: TTSRequest, nfe_step: Optional[int] = Query(None, alias="nfe-step")):
    """Return JSON timing breakdown without streaming audio (for TTFT benchmarking)."""
    resolved_nfe = _resolve_nfe_step(nfe_step, body.nfe_step)
    assert session is not None

    req_t0 = time.perf_counter()
    chunk_metrics = []
    for chunk in session.stream(
        body.text,
        body.ref_audio_path,
        body.ref_text,
        body.max_chars,
        resolved_nfe,
    ):
        chunk_metrics.append(
            {
                "index": chunk.index,
                "total": chunk.total,
                "chars": len(chunk.text),
                "audio_seconds": len(chunk.audio) / SAMPLE_RATE,
                "cfm_seconds": round(chunk.cfm_seconds, 3),
                "vocoder_seconds": round(chunk.vocoder_seconds, 3),
                "gen_seconds": round(chunk.gen_seconds, 3),
                "since_request_start": round(time.perf_counter() - req_t0, 3),
            }
        )

    return {
        "nfe_step": resolved_nfe,
        "chunk_count": len(chunk_metrics),
        "ttft_seconds": chunk_metrics[0]["since_request_start"] if chunk_metrics else None,
        "total_seconds": round(time.perf_counter() - req_t0, 3),
        "total_audio_seconds": round(sum(c["audio_seconds"] for c in chunk_metrics), 3),
        "chunks": chunk_metrics,
    }


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
