from transformers import AutoModel
import torch
import sounddevice as sd
from threading import Thread
from queue import Queue

from streaming_indicf5 import stream_indicf5

# ------------------------------------------------
# Device
# ------------------------------------------------

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using:", device)

# ------------------------------------------------
# Load model
# ------------------------------------------------

print("Loading model...")

model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True,
)

model = model.to(device)

print("Model loaded.")

# ------------------------------------------------
# Reference
# ------------------------------------------------

ref_audio_path = "reference/amitabh_voice.wav"

ref_text = (
    "लहरों से डर कर नौका पार नहीं होती, "
    "कोशिश करने वालों की कभी हार नहीं होती। "
    "नन्हीं चींटी जब दाना लेकर चलती है, "
    "चढ़ती दीवारों पर सौ बार फिसलती है।"
)

text = """
ਜ਼ਿੰਦਗੀ ਵਿੱਚ ਕਾਮਯਾਬੀ ਉਹਨਾਂ ਲੋਕਾਂ ਨੂੰ ਮਿਲਦੀ ਹੈ ਜੋ ਮੁਸ਼ਕਲਾਂ ਤੋਂ ਨਹੀਂ ਡਰਦੇ।
ਹਰ ਨਵਾਂ ਦਿਨ ਇੱਕ ਨਵਾਂ ਮੌਕਾ ਲੈ ਕੇ ਆਉਂਦਾ ਹੈ।
ਜੇ ਤੁਸੀਂ ਆਪਣੇ ਸੁਪਨਿਆਂ 'ਤੇ ਵਿਸ਼ਵਾਸ ਰੱਖਦੇ ਹੋ।
ਅਸਫਲਤਾ ਸਿਰਫ਼ ਇੱਕ ਸਬਕ ਹੈ।
"""

# ------------------------------------------------

queue = Queue(maxsize=2)

# ------------------------------------------------

def producer():

    for idx, total, chunk, audio in stream_indicf5(
        model,
        text,
        ref_audio_path,
        ref_text,
        max_chars=80,
    ):

        print(f"\nGenerated {idx+1}/{total}")

        queue.put(audio)

    queue.put(None)

# ------------------------------------------------

def consumer():

    while True:

        audio = queue.get()

        if audio is None:
            break

        print("Playing...")

        sd.play(audio, samplerate=24000)
        sd.wait()

# ------------------------------------------------

producer_thread = Thread(target=producer)
consumer_thread = Thread(target=consumer)

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()

print("\nFinished.")