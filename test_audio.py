import numpy as np
import sounddevice as sd

fs = 24000

t = np.linspace(0, 1, fs, endpoint=False)

audio = 0.3 * np.sin(2 * np.pi * 440 * t)

print("Playing...")

sd.play(audio.astype(np.float32), fs)

sd.wait()

print("Done")