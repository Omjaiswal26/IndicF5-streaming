#!/usr/bin/env python3
"""
RunPod / CUDA verification for IndicF5 streaming.

Checks environment, loads weights via indicf5_load (not AutoModel), synthesizes
a short clip, and validates audio statistics.

Usage:
  python scripts/pod_verify.py --device cuda
  python scripts/pod_verify.py --device cuda --output /tmp/pod_verify.wav
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Repo root on sys.path when run as scripts/pod_verify.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from indicf5_load import configure_logging, load_indicf5_models  # noqa: E402
from indicf5_streaming import IndicF5Session  # noqa: E402

import logging

logger = logging.getLogger(__name__)

DEFAULT_REF_AUDIO = "reference/amitabh_voice.wav"
DEFAULT_REF_TEXT = "लहरों से डर कर नौका पार नहीं होती"
DEFAULT_GEN_TEXT = "ਜ਼ਿੰਦਗੀ ਵਿੱਚ ਕਾਮਯਾਬੀ ਉਹਨਾਂ ਲੋਕਾਂ ਨੂੰ ਮਿਲਦੀ ਹੈ।"


def check_transformers_version() -> bool:
    try:
        import transformers

        version = transformers.__version__
        major = int(version.split(".")[0])
        ok = major < 5
        logger.info("transformers version=%s ok=%s (need <5.x)", version, ok)
        return ok
    except ImportError:
        logger.error("transformers not installed")
        return False


def check_torch(device: str) -> bool:
    import torch

    logger.info("torch version=%s", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    logger.info("torch.cuda.is_available()=%s", cuda_ok)

    if device == "cuda" and not cuda_ok:
        logger.error("Requested --device cuda but CUDA is not available")
        return False

    if cuda_ok:
        logger.info("cuda device=%s", torch.cuda.get_device_name(0))
    return True


def check_reference_audio(path: str) -> bool:
    p = Path(path)
    if not p.is_file():
        logger.error("Reference audio missing: %s", p.resolve())
        return False
    logger.info("reference audio ok: %s", p.resolve())
    return True


def validate_audio_stats(audio: np.ndarray) -> dict[str, float | int | bool]:
    finite = np.isfinite(audio)
    stats = {
        "samples": len(audio),
        "nan_count": int(np.isnan(audio).sum()),
        "inf_count": int(np.isinf(audio).sum()),
        "rms": float(np.sqrt(np.mean(audio[finite] ** 2))) if finite.any() else 0.0,
        "peak": float(np.max(np.abs(audio[finite]))) if finite.any() else 0.0,
        "pct_near_silence": float((np.abs(audio) < 0.01).mean() * 100),
        "unique_rounded_10": len(np.unique(np.round(audio[finite], 2))) if finite.any() else 0,
        "all_minus_one": bool(
            finite.all() and len(audio) > 0 and np.all(np.round(audio, 2) == -1.0)
        ),
    }
    stats["ok"] = (
        stats["nan_count"] == 0
        and stats["inf_count"] == 0
        and not stats["all_minus_one"]
        and stats["rms"] < 0.5
        and stats["peak"] <= 1.0
        and stats["unique_rounded_10"] > 3
    )
    return stats


def log_audio_stats(stats: dict, label: str) -> None:
    logger.info("=== audio stats (%s) ===", label)
    for key in (
        "samples",
        "nan_count",
        "inf_count",
        "rms",
        "peak",
        "pct_near_silence",
        "unique_rounded_10",
        "all_minus_one",
        "ok",
    ):
        logger.info("%s=%s", key, stats[key])


def run_synthesis(
    device: str,
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    nfe_step: int,
    output: Path,
) -> dict:
    logger.info("=== synthesis via IndicF5Session ===")
    session = IndicF5Session(device=device)
    if session.load_report and not session.load_report.weights_ok:
        logger.error("Aborting: weights did not load cleanly")
        return {"ok": False, "reason": "weights_not_ok"}

    chunks = list(
        session.stream(
            text=gen_text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            max_chars=120,
            nfe_step=nfe_step,
        )
    )
    if not chunks:
        logger.error("No audio chunks produced")
        return {"ok": False, "reason": "no_chunks"}

    audio = chunks[0].audio
    sf.write(str(output), audio, 24000)
    logger.info("wrote %s (%d samples, %.2fs)", output, len(audio), len(audio) / 24000)

    stats = validate_audio_stats(audio)
    log_audio_stats(stats, "in-memory")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify IndicF5 pod setup")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--output", type=Path, default=Path("/tmp/pod_verify.wav"))
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--text", default=DEFAULT_GEN_TEXT)
    parser.add_argument("--nfe-step", type=int, default=16)
    parser.add_argument("--load-only", action="store_true", help="Only test weight loading")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger.info("=== pod_verify start ===")
    logger.info("repo_root=%s", _REPO_ROOT)

    checks_ok = True
    checks_ok &= check_transformers_version()
    checks_ok &= check_torch(args.device)
    checks_ok &= check_reference_audio(args.ref_audio)

    try:
        logger.info("=== direct load (indicf5_load) ===")
        _, _, report = load_indicf5_models(device=args.device)
        checks_ok &= report.weights_ok
    except Exception:
        logger.exception("load_indicf5_models failed")
        return 1

    if args.load_only:
        logger.info("=== pod_verify done (load-only) ok=%s ===", checks_ok)
        return 0 if checks_ok else 1

    try:
        stats = run_synthesis(
            device=args.device,
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
            gen_text=args.text,
            nfe_step=args.nfe_step,
            output=args.output,
        )
        checks_ok &= bool(stats.get("ok"))
        if args.output.is_file():
            disk_audio, _ = sf.read(str(args.output))
            disk_stats = validate_audio_stats(disk_audio)
            log_audio_stats(disk_stats, "on-disk")
            checks_ok &= bool(disk_stats.get("ok"))
    except Exception:
        logger.exception("synthesis failed")
        return 1

    logger.info("=== pod_verify done ok=%s ===", checks_ok)
    return 0 if checks_ok else 1


if __name__ == "__main__":
    sys.exit(main())
