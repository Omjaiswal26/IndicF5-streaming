"""FastAPI server for true chunk-level IndicF5 streaming."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from indicf5_streaming import IndicF5Session, StreamChunk
from tts_audio import encode_framed_float32, encode_s16le
from tts_config import TTSDefaults, resolve_ref, server_defaults

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
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    max_chars: int = 120
    split: bool = True
    nfe_step: Optional[int] = Field(default=None, alias="nfe-step")

    model_config = {"populate_by_name": True}


class SynthesizeRequest(BaseModel):
    """LiveKit-friendly: text-only by default, raw PCM out, no sentence split."""

    text: str
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    max_chars: int = 120
    split: bool = False
    nfe_step: Optional[int] = Field(default=None, alias="nfe-step")

    model_config = {"populate_by_name": True}


def _resolve_nfe_step(query_nfe: Optional[int], body_nfe: Optional[int] = None) -> int:
    if query_nfe is not None:
        return query_nfe
    if body_nfe is not None:
        return body_nfe
    return default_nfe_step


def _log_metric(message: str) -> None:
    print(f"[metrics] {message}", file=sys.stderr, flush=True)


def _is_ready() -> bool:
    return session is not None and session.load_report is not None and session.load_report.weights_ok


def _stream_params(body: TTSRequest | SynthesizeRequest, nfe_query: Optional[int]) -> dict:
    ref_audio, ref_text = resolve_ref(body.ref_audio_path, body.ref_text)
    max_chars = body.max_chars if body.max_chars is not None else server_defaults.max_chars
    return {
        "text": body.text,
        "ref_audio_path": ref_audio,
        "ref_text": ref_text,
        "max_chars": max_chars,
        "split": body.split,
        "nfe_step": _resolve_nfe_step(nfe_query, body.nfe_step),
    }


async def _async_stream_chunks(
    request: Request,
    params: dict,
    request_id: str,
) -> AsyncIterator[StreamChunk]:
    """Run sync chunk generation in executor; stop if client disconnects."""
    assert session is not None
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[StreamChunk | None | BaseException] = asyncio.Queue()
    cancelled = asyncio.Event()

    def producer() -> None:
        try:
            for chunk in session.stream(
                text=params["text"],
                ref_audio_path=params["ref_audio_path"],
                ref_text=params["ref_text"],
                max_chars=params["max_chars"],
                nfe_step=params["nfe_step"],
                split=params["split"],
            ):
                if cancelled.is_set():
                    _log_metric(f"request_id={request_id} cancelled during generation")
                    return
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
            loop.call_soon_threadsafe(queue.put_nowait, None)
        except BaseException as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)

    producer_task = loop.run_in_executor(None, producer)
    req_t0 = time.perf_counter()
    first_logged = False

    try:
        while True:
            if await request.is_disconnected():
                cancelled.set()
                _log_metric(f"request_id={request_id} client_disconnected total={time.perf_counter() - req_t0:.3f}s")
                break

            item = await queue.get()
            if isinstance(item, BaseException):
                raise item
            if item is None:
                break

            if not first_logged:
                _log_metric(
                    f"request_id={request_id} ttft={time.perf_counter() - req_t0:.3f}s "
                    f"chunk=1/{item.total} nfe_step={params['nfe_step']} chars={len(item.text)}"
                )
                first_logged = True

            yield item
    finally:
        cancelled.set()
        await producer_task


async def _async_encode_stream(
    request: Request,
    params: dict,
    request_id: str,
    encode_fn: Callable[[StreamChunk], bytes],
    log_label: str,
) -> AsyncIterator[bytes]:
    req_t0 = time.perf_counter()
    chunk_idx = 0
    async for chunk in _async_stream_chunks(request, params, request_id):
        chunk_idx += 1
        payload = encode_fn(chunk)
        _log_metric(
            f"request_id={request_id} chunk={chunk.index + 1}/{chunk.total} "
            f"format={log_label} gen={chunk.gen_seconds:.3f}s cfm={chunk.cfm_seconds:.3f}s "
            f"voc={chunk.vocoder_seconds:.3f}s bytes={len(payload)} "
            f"audio_s={len(chunk.audio) / SAMPLE_RATE:.2f}"
        )
        yield payload

    if chunk_idx:
        _log_metric(f"request_id={request_id} stream_complete total={time.perf_counter() - req_t0:.3f}s")
    else:
        _log_metric(f"request_id={request_id} stream_empty total={time.perf_counter() - req_t0:.3f}s")


def _framed_headers(nfe_step: int) -> dict[str, str]:
    return {
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-NFE-Step": str(nfe_step),
        "X-Chunk-Format": "uint32-le-length-prefix + float32-le-pcm",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }


def _s16le_headers(nfe_step: int) -> dict[str, str]:
    return {
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-NFE-Step": str(nfe_step),
        "X-Audio-Format": "pcm_s16le",
        "Content-Type": "audio/L16;rate=24000;channels=1",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id", f"req-{int(time.time() * 1000)}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sample_rate": SAMPLE_RATE,
        "default_nfe_step": default_nfe_step,
        "default_ref_audio": server_defaults.ref_audio_path,
        "ready": _is_ready(),
    }


@app.get("/ready")
def ready():
    if not _is_ready():
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "message": "Model not loaded or weights failed"},
        )
    return {
        "status": "ready",
        "sample_rate": SAMPLE_RATE,
        "default_nfe_step": default_nfe_step,
        "default_ref_audio": server_defaults.ref_audio_path,
    }


@app.post("/v1/tts/synthesize")
async def v1_tts_synthesize(
    request: Request,
    body: SynthesizeRequest,
    nfe_step: Optional[int] = Query(None, alias="nfe-step"),
):
    """
    LiveKit-oriented endpoint: optional ref fields, no split by default,
    streams raw int16 PCM @ 24 kHz (no length framing).
    """
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    params = _stream_params(body, nfe_step)
    request_id = _request_id(request)

    return StreamingResponse(
        _async_encode_stream(request, params, request_id, encode_s16le, "pcm_s16le"),
        media_type="audio/L16;rate=24000;channels=1",
        headers=_s16le_headers(params["nfe_step"]),
    )


@app.post("/tts/stream")
async def tts_stream_post(
    request: Request,
    body: TTSRequest,
    nfe_step: Optional[int] = Query(None, alias="nfe-step"),
):
    """Stream TTS audio chunks (framed float32 PCM) as each is synthesized."""
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    params = _stream_params(body, nfe_step)
    request_id = _request_id(request)

    return StreamingResponse(
        _async_encode_stream(request, params, request_id, encode_framed_float32, "framed_f32"),
        media_type="application/octet-stream",
        headers=_framed_headers(params["nfe_step"]),
    )


@app.get("/tts/stream")
async def tts_stream_get(
    request: Request,
    text: str,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    max_chars: int = 120,
    split: bool = True,
    nfe_step: Optional[int] = Query(None, alias="nfe-step"),
):
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    body = TTSRequest(
        text=text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        max_chars=max_chars,
        split=split,
        nfe_step=nfe_step,
    )
    return await tts_stream_post(request, body, nfe_step)


@app.post("/tts/benchmark")
def tts_benchmark(body: TTSRequest, nfe_step: Optional[int] = Query(None, alias="nfe-step")):
    """Return JSON timing breakdown without streaming audio."""
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    params = _stream_params(body, nfe_step)
    assert session is not None

    req_t0 = time.perf_counter()
    chunk_metrics = []
    for chunk in session.stream(
        params["text"],
        params["ref_audio_path"],
        params["ref_text"],
        params["max_chars"],
        params["nfe_step"],
        split=params["split"],
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
        "nfe_step": params["nfe_step"],
        "split": params["split"],
        "chunk_count": len(chunk_metrics),
        "ttft_seconds": chunk_metrics[0]["since_request_start"] if chunk_metrics else None,
        "total_seconds": round(time.perf_counter() - req_t0, 3),
        "total_audio_seconds": round(sum(c["audio_seconds"] for c in chunk_metrics), 3),
        "chunks": chunk_metrics,
    }


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def _apply_cli_defaults(args: argparse.Namespace) -> None:
    import tts_config as cfg

    global default_nfe_step
    cfg.server_defaults = TTSDefaults(
        ref_audio_path=args.default_ref_audio,
        ref_text=args.default_ref_text,
        nfe_step=args.nfe_step,
        max_chars=args.default_max_chars,
    )
    default_nfe_step = args.nfe_step


def main():
    global session

    parser = argparse.ArgumentParser(description="IndicF5 chunk-level streaming TTS server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default="ai4bharat/IndicF5")
    parser.add_argument(
        "--nfe-step",
        type=int,
        default=server_defaults.nfe_step,
        help="Default NFE steps when request omits nfe-step",
    )
    parser.add_argument(
        "--default-ref-audio",
        default=server_defaults.ref_audio_path,
        help="Default reference wav when request omits ref_audio_path",
    )
    parser.add_argument(
        "--default-ref-text",
        default=server_defaults.ref_text,
        help="Default reference transcript when request omits ref_text",
    )
    parser.add_argument(
        "--default-max-chars",
        type=int,
        default=server_defaults.max_chars,
        help="Default max_chars for /tts/stream when not specified",
    )
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detect if omitted)")
    args = parser.parse_args()

    _apply_cli_defaults(args)
    session = IndicF5Session(model_id=args.model_id, device=args.device)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
