#!/usr/bin/env python3
"""CLI TTFT benchmark — calls IndicF5Session directly (no HTTP proxy buffering)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from indicf5_streaming import IndicF5Session  # noqa: E402

DEFAULT_REF_AUDIO = "reference/amitabh_voice.wav"
DEFAULT_REF_TEXT = "लहरों से डर कर नौका पार नहीं होती"
DEFAULT_TEXT = """ਜ਼ਿੰਦਗੀ ਵਿੱਚ ਕਾਮਯਾਬੀ ਉਹਨਾਂ ਲੋਕਾਂ ਨੂੰ ਮਿਲਦੀ ਹੈ ਜੋ ਮੁਸ਼ਕਲਾਂ ਤੋਂ ਨਹੀਂ ਡਰਦੇ।
ਹਰ ਨਵਾਂ ਦਿਨ ਇੱਕ ਨਵਾਂ ਮੌਕਾ ਲੈ ਕੇ ਆਉਂਦਾ ਹੈ।
ਜੇ ਤੁਸੀਂ ਆਪਣੇ ਸੁਪਨਿਆਂ 'ਤੇ ਵਿਸ਼ਵਾਸ ਰੱਖਦੇ ਹੋ।
ਅਸਫਲਤਾ ਸਿਰਫ਼ ਇੱਕ ਸਬਕ ਹੈ।"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TTFT for IndicF5 streaming")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nfe-step", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    session = IndicF5Session(device=args.device)
    req_t0 = time.perf_counter()
    chunks = []

    for chunk in session.stream(
        args.text,
        args.ref_audio,
        args.ref_text,
        max_chars=args.max_chars,
        nfe_step=args.nfe_step,
    ):
        chunks.append(
            {
                "index": chunk.index + 1,
                "total": chunk.total,
                "chars": len(chunk.text),
                "audio_seconds": round(len(chunk.audio) / 24000, 3),
                "cfm_seconds": round(chunk.cfm_seconds, 3),
                "vocoder_seconds": round(chunk.vocoder_seconds, 3),
                "gen_seconds": round(chunk.gen_seconds, 3),
                "since_request_start": round(time.perf_counter() - req_t0, 3),
            }
        )

    result = {
        "nfe_step": args.nfe_step,
        "max_chars": args.max_chars,
        "chunk_count": len(chunks),
        "ttft_seconds": chunks[0]["since_request_start"] if chunks else None,
        "total_seconds": round(time.perf_counter() - req_t0, 3),
        "total_audio_seconds": round(sum(c["audio_seconds"] for c in chunks), 3),
        "chunks": chunks,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=== TTFT benchmark ===")
        print(f"nfe_step={result['nfe_step']} max_chars={result['max_chars']}")
        print(f"chunks={result['chunk_count']} ttft={result['ttft_seconds']}s total={result['total_seconds']}s")
        print(f"audio_total={result['total_audio_seconds']}s")
        for c in chunks:
            print(
                f"  chunk {c['index']}/{c['total']}: "
                f"gen={c['gen_seconds']}s cfm={c['cfm_seconds']}s "
                f"audio={c['audio_seconds']}s chars={c['chars']} "
                f"@+{c['since_request_start']}s"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
