from transformers import AutoModel
import torch
import sounddevice as sd
from threading import Thread
from queue import Queue
import numpy as np

from streaming_indicf5 import stream_indicf5

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using", device)

model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True,
)

model = model.to(device)

ref_audio_path = "reference/amitabh_voice.wav"

ref_text = (
    "लहरों से डर कर नौका पार नहीं होती, "
    "कोशिश करने वालों की कभी हार नहीं होती। "
    "नन्हीं चींटी जब दाना लेकर चलती है, "
    "चढ़ती दीवारों पर सौ बार फिसलती है।"
)

audio_queue = Queue(maxsize=2)

def play_worker():

    while True:

        audio = audio_queue.get()

        if audio is None:
            break

        sd.play(audio, samplerate=24000)
        sd.wait()

        