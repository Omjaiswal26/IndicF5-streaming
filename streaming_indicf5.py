import re
import numpy as np


def split_text(text, max_chars=120):
    """Split text into sentence-sized chunks."""

    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > max_chars:
            parts = re.split(r'(?<=[,;:])\s+', sentence)
        else:
            parts = [sentence]

        for part in parts:
            if len(current) + len(part) + 1 <= max_chars:
                current = (current + " " + part).strip()
            else:
                if current:
                    chunks.append(current)
                current = part

    if current:
        chunks.append(current)

    return chunks


def stream_indicf5(
    model,
    text,
    ref_audio_path,
    ref_text,
    max_chars=120,
):
    """Generate one chunk at a time."""

    chunks = split_text(text, max_chars)

    total = len(chunks)

    for i, chunk in enumerate(chunks):

        print(f"Generating chunk {i+1}/{total}")

        audio = model(
            chunk,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
        )

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        yield (
            i,
            total,
            chunk,
            np.asarray(audio, dtype=np.float32),
        )