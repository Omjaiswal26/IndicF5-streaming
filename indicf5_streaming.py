"""Chunk-level streaming wrapper for IndicF5 (no model.py / f5_tts/model changes)."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Generator, Iterator, Optional, Tuple

import numpy as np
import torch
import torchaudio

from indicf5_load import LoadReport, load_indicf5_models
from f5_tts.infer.utils_infer import (
    cfg_strength,
    convert_char_to_pinyin,
    hop_length,
    preprocess_ref_audio_text,
    speed,
    sway_sampling_coef,
    target_rms,
    target_sample_rate,
)
from streaming_indicf5 import split_text

RefCacheKey = Tuple[str, str]


@dataclass(frozen=True)
class RefConditioning:
    audio: torch.Tensor
    ref_text: str
    ref_audio_len: int
    rms: float


@dataclass(frozen=True)
class StreamChunk:
    index: int
    total: int
    text: str
    audio: np.ndarray
    sample_rate: int = target_sample_rate
    cfm_seconds: float = 0.0
    vocoder_seconds: float = 0.0
    gen_seconds: float = 0.0


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _log_timing(message: str) -> None:
    print(f"[timing] {message}", file=sys.stderr, flush=True)


class IndicF5Session:
    """Caches reference conditioning and streams one waveform chunk at a time."""

    def __init__(
        self,
        model_id: str = "ai4bharat/IndicF5",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device or _default_device()
        self._ref_cache: dict[RefCacheKey, RefConditioning] = {}
        self.load_report: Optional[LoadReport] = None

        _log_timing(f"loading model {model_id} on {self.device} (direct checkpoint load)")
        self.ema_model, self.vocoder, self.load_report = load_indicf5_models(
            model_id=model_id,
            device=self.device,
        )
        if self.load_report.weights_ok:
            _log_timing(
                f"model ready in {self.load_report.load_seconds:.3f}s "
                f"ema_keys={self.load_report.ema_keys_loaded}"
            )
        else:
            _log_timing(
                f"model loaded with weight errors in {self.load_report.load_seconds:.3f}s "
                f"missing={len(self.load_report.ema_missing)} "
                f"unexpected={len(self.load_report.ema_unexpected)}"
            )

    def clear_ref_cache(self) -> None:
        self._ref_cache.clear()

    def _cache_key(self, ref_audio_path: str, ref_text: str) -> RefCacheKey:
        return (os.path.abspath(ref_audio_path), ref_text)

    def _get_ref_conditioning(self, ref_audio_path: str, ref_text: str) -> RefConditioning:
        key = self._cache_key(ref_audio_path, ref_text)
        cached = self._ref_cache.get(key)
        if cached is not None:
            _log_timing(f"ref cache hit key={key[0]}")
            return cached

        t0 = time.perf_counter()
        processed_path, processed_text = preprocess_ref_audio_text(
            ref_audio_path,
            ref_text,
            device=self.device,
        )
        audio, sr = torchaudio.load(processed_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio))).item()
        if rms < target_rms:
            audio = audio * (target_rms / rms)

        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)

        audio = audio.to(self.device)

        if len(processed_text[-1].encode("utf-8")) == 1:
            processed_text = processed_text + " "

        ref_audio_len = audio.shape[-1] // hop_length
        conditioning = RefConditioning(
            audio=audio,
            ref_text=processed_text,
            ref_audio_len=ref_audio_len,
            rms=rms,
        )
        self._ref_cache[key] = conditioning
        _log_timing(
            f"ref cache miss preprocess+load+resample={time.perf_counter() - t0:.3f}s "
            f"ref_mel_frames={ref_audio_len}"
        )
        return conditioning

    def stream(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        max_chars: int = 120,
        nfe_step: int = 32,
        split: bool = True,
    ) -> Iterator[StreamChunk]:
        """Yield each synthesized chunk as soon as CFM + vocoder finish for it."""
        ref = self._get_ref_conditioning(ref_audio_path, ref_text)
        if split:
            chunks = split_text(text, max_chars=max_chars)
        else:
            stripped = text.strip()
            chunks = [stripped] if stripped else []
        if not chunks:
            chunks = [text.strip()]
        total = len(chunks)
        stream_t0 = time.perf_counter()

        self.ema_model.to(self.device)
        self.vocoder.to(self.device)

        for index, gen_text in enumerate(chunks):
            chunk_t0 = time.perf_counter()

            text_list = [ref.ref_text + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)

            ref_text_len = len(ref.ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref.ref_audio_len + int(
                ref.ref_audio_len / ref_text_len * gen_text_len / speed
            )

            t_cfm = time.perf_counter()
            with torch.inference_mode():
                generated, _ = self.ema_model.sample(
                    cond=ref.audio,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
            cfm_elapsed = time.perf_counter() - t_cfm

            generated = generated.to(torch.float32)
            generated = generated[:, ref.ref_audio_len :, :]
            generated_mel_spec = generated.permute(0, 2, 1)

            t_voc = time.perf_counter()
            generated_wave = self.vocoder.decode(generated_mel_spec)
            voc_elapsed = time.perf_counter() - t_voc

            if ref.rms < target_rms:
                generated_wave = generated_wave * (ref.rms / target_rms)

            audio_np = generated_wave.squeeze().cpu().numpy().astype(np.float32)
            chunk_elapsed = time.perf_counter() - chunk_t0

            _log_timing(
                f"chunk {index + 1}/{total} "
                f"cfm.sample={cfm_elapsed:.3f}s "
                f"vocoder.decode={voc_elapsed:.3f}s "
                f"total={chunk_elapsed:.3f}s "
                f"since_stream_start={time.perf_counter() - stream_t0:.3f}s "
                f"audio_s={len(audio_np) / target_sample_rate:.2f} "
                f"nfe_step={nfe_step} "
                f"chars={len(gen_text)}"
            )
            if index == 0:
                _log_timing(f"ttft_stream={time.perf_counter() - stream_t0:.3f}s (first chunk ready)")

            yield StreamChunk(
                index=index,
                total=total,
                text=gen_text,
                audio=audio_np,
                cfm_seconds=cfm_elapsed,
                vocoder_seconds=voc_elapsed,
                gen_seconds=chunk_elapsed,
            )


def stream_audio_bytes(chunks: Iterator[StreamChunk]) -> Generator[bytes, None, None]:
    """Encode chunks as [uint32 sample_count][float32 pcm] frames for HTTP streaming."""
    import struct

    for chunk in chunks:
        samples = chunk.audio.astype(np.float32, copy=False)
        yield struct.pack("<I", len(samples)) + samples.tobytes()
