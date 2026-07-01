"""PCM encoding helpers for HTTP streaming."""

from __future__ import annotations

import struct

import numpy as np

from indicf5_streaming import StreamChunk


def float32_to_int16_pcm(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def encode_framed_float32(chunk: StreamChunk) -> bytes:
    samples = chunk.audio.astype(np.float32, copy=False)
    return struct.pack("<I", len(samples)) + samples.tobytes()


def encode_s16le(chunk: StreamChunk) -> bytes:
    return float32_to_int16_pcm(chunk.audio)
