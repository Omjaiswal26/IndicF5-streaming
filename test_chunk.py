from transformers import AutoModel
import torch
import sounddevice as sd

from streaming_indicf5 import stream_indicf5

# -------------------------
# Device
# -------------------------

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using:", device)

# -------------------------
# Load model
# -------------------------

print("Loading model...")

model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True,
)

model = model.to(device)

print("Model loaded.")

# -------------------------
# Reference voice
# -------------------------

ref_audio_path = "reference/amitabh_voice.wav"

ref_text = (
    "लहरों से डर कर नौका पार नहीं होती, "
    "कोशिश करने वालों की कभी हार नहीं होती। "
    "नन्हीं चींटी जब दाना लेकर चलती है, "
    "चढ़ती दीवारों पर सौ बार फिसलती है।"
)

text = """
ज़ਿੰਦगी ਵਿੱਚ ਕਾਮਯਾਬੀ ਉਹਨਾਂ ਲੋਕਾਂ ਨੂੰ ਮਿਲਦੀ ਹੈ ਜੋ ਮੁਸ਼ਕਲਾਂ ਤੋਂ ਨਹੀਂ ਡਰਦੇ।
ਹਰ ਨਵਾਂ ਦਿਨ ਇੱਕ ਨਵਾਂ ਮੌਕਾ ਲੈ ਕੇ ਆਉਂਦਾ ਹੈ।
ਜੇ ਤੁਸੀਂ ਆਪਣੇ ਸੁਪਨਿਆਂ 'ਤੇ ਵਿਸ਼ਵਾਸ ਰੱਖਦੇ ਹੋ।
"""

# -------------------------
# Generate ONE chunk
# -------------------------

generator = stream_indicf5(
    model,
    text,
    ref_audio_path,
    ref_text,
    max_chars=80,
)

idx, total, chunk, audio = next(generator)

print()
print(f"Chunk {idx+1}/{total}")
print(chunk)
print()

print("Playing...")

sd.play(audio, samplerate=24000)
sd.wait()

print("Finished.")