# **IndicF5: High-Quality Text-to-Speech for Indian Languages**

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/ai4bharat/IndicF5)


We release **IndicF5**, a **near-human polyglot** **Text-to-Speech (TTS)** model trained on **1417 hours** of high-quality speech from **[Rasa](https://huggingface.co/datasets/ai4bharat/Rasa), [IndicTTS](https://www.iitm.ac.in/donlab/indictts/database), [LIMMITS](https://sites.google.com/view/limmits24/), and [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)**.  

IndicF5 supports **11 Indian languages**:  
**Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  

---

## 🚀 Installation
```bash
conda create -n indicf5 python=3.10 -y
conda activate indicf5
pip install git+https://github.com/ai4bharat/IndicF5.git
```


## 🎙 Usage

To generate speech, you need to provide **three inputs**:
1. **Text to synthesize** – The content you want the model to speak.
2. **A reference prompt audio** – An example speech clip that guides the model’s prosody and speaker characteristics.
3. **Text spoken in the reference prompt audio** – The transcript of the reference prompt audio.


```python
from transformers import AutoModel
import numpy as np
import soundfile as sf

# Load INF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Generate speech
audio = model(
    "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.",
    ref_audio_path="prompts/PAN_F_HAPPY_00001.wav",
    ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("samples/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
```

---

## Chunk-level streaming (this fork)

This repo adds true **sentence/chunk-level streaming** without modifying HF `model.py` or `f5_tts/model/`.

| File | Purpose |
|------|---------|
| `indicf5_streaming.py` | `IndicF5Session` — caches ref audio, streams one chunk at a time via `CFM.sample()` + `vocoder.decode()` |
| `tts_server.py` | FastAPI server + web test UI at `/` |
| `web/` | Browser UI for streaming TTS |
| `streaming_indicf5.py` | Sentence splitter used by the session |

### Local quick start

```bash
cd IndicF5
python -m venv venv && source venv/bin/activate
pip install -U pip
pip install 'transformers>=4.40,<4.50'   # important — see pitfalls below
pip install -e .
pip install fastapi "uvicorn[standard]"

python tts_server.py --host 0.0.0.0 --port 8000 --device mps --nfe-step 16
# Mac: use --device mps (or cpu). Linux GPU: --device cuda
```

Open `http://localhost:8000` for the test UI. Default ref audio: `reference/amitabh_voice.wav`.

### API

- `GET /health` — server status
- `POST /tts/stream` — JSON body: `text`, `ref_audio_path`, `ref_text`, `max_chars`, `nfe-step`
- Response: framed **float32 PCM @ 24 kHz** — `[uint32 sample_count][pcm bytes]` per chunk

Query param `?nfe-step=16` overrides the CLI default (32). Lower NFE = faster, slightly lower quality.

### Python streaming API

```python
from indicf5_streaming import IndicF5Session

session = IndicF5Session(device="cuda")  # or mps / cpu
for chunk in session.stream(text, ref_audio_path, ref_text, nfe_step=16):
    print(chunk.index, chunk.text, len(chunk.audio))  # play chunk.audio @ 24000 Hz
```

Timing breakdown is printed to **stderr** (`[timing] ref cache …`, `cfm.sample=…`, `vocoder.decode=…`).

---

## RunPod / cloud GPU deployment

### Pod setup

- Template: **PyTorch** (CUDA, Python 3.10+)
- GPU: RTX 4090 / A100 recommended
- Disk: **≥ 30 GB** (model + HF cache)
- Expose HTTP port: **8000**

### Install on pod

```bash
cd /workspace
apt-get update && apt-get install -y git ffmpeg ca-certificates

git clone https://github.com/Omjaiswal26/IndicF5-streaming.git
cd IndicF5-streaming

python3 -m venv venv && source venv/bin/activate
pip install -U pip

# Pin transformers BEFORE editable install (see pitfalls)
pip install 'transformers>=4.40,<4.50'

pip install -e .
pip install fastapi "uvicorn[standard]"

# PyTorch: match pod CUDA driver (cu124 works on most RunPod images)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # must be True
```

### Run server

```bash
python tts_server.py --host 0.0.0.0 --port 8000 --device cuda --nfe-step 16
```

Open RunPod **Connect → HTTP port 8000** proxy URL in your browser.

Optional: `export HF_HOME=/workspace/hf-cache` to persist model downloads on a volume.

---

## Pitfalls and troubleshooting

### 1. Pin `transformers<4.50` (critical on CUDA pods)

`pip install -e .` can pull **transformers 5.x**, which breaks `AutoModel.from_pretrained("ai4bharat/IndicF5")` on GPU with:

```
RuntimeError: Tensor on device cpu is not on the expected device meta!
```

The crash happens inside **vocoder init** (`MelSpectrogram`) during HF `INF5Model.__init__`, not in your streaming code.

**Fix:** always install `transformers>=4.40,<4.50` (e.g. 4.49.0) before the rest:

```bash
pip install 'transformers>=4.40,<4.50'
```

**Why Mac often works:** typically already on transformers 4.x, and HF `model.py` uses `device=cpu` when CUDA is absent.

---

### 2. PyTorch CUDA version must match the driver

Symptom: `CUDA driver too old` and `torch.cuda.is_available() == False` despite a GPU.

**Fix:** reinstall matching wheels (do not rely on `pip install` saying "already satisfied"):

```bash
pip uninstall -y torch torchaudio triton
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

---

### 3. HF `model.py` uses `torch.compile` on CUDA

The Hugging Face remote `model.py` wraps `ema_model` and `vocoder` in `torch.compile()`. This causes:

- Load-time errors on some torch/CUDA combos (even without compile being the only issue)
- **Weight key mismatch** at load: checkpoint has `ema_model._orig_mod.*` but model expects `ema_model.*` → weights not loaded → **garbage audio**

You may see warnings like `Some weights … were not used` / `newly initialized`.

**Workarounds (without editing HF `model.py`):**

- Use `IndicF5Session` and plan to migrate to direct `f5_tts` weight loading (bypass `AutoModel`)
- On pod, patch cached `model.py` to remove `torch.compile(` (under `~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/…`)
- Listen-test after deploy; do not assume load warnings are harmless

---

### 4. SSL / certificate issues on some pods

Broken pods may MITM HTTPS (`self-signed certificate`) or return 404 for apt/curl.se.

| Symptom | Workaround |
|---------|------------|
| `git clone` SSL fail | `GIT_SSL_NO_VERIFY=1 git clone …` |
| `pip` SSL fail | `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org …` |
| Hugging Face SSL fail | `export HF_HUB_DISABLE_SSL_VERIFICATION=1` (pod-only workaround) |
| Bad `/tmp/cacert.pem` (19 bytes, "404 page not found") | Use `python -c "import certifi; print(certifi.where())"` or `export HF_HUB_DISABLE_SSL_VERIFICATION=1` |

Prefer a **fresh RunPod PyTorch template** over fighting a broken network stack.

---

### 5. `ffmpeg` warning

```
Couldn't find ffmpeg or avconv
```

Install: `apt-get install -y ffmpeg`. Reference `.wav` files may still work without it; pydub preprocessing for some formats needs ffmpeg.

---

### 6. Mac vs RunPod behavior summary

| | Mac (local) | RunPod (CUDA) |
|---|-------------|---------------|
| Device flag | `--device mps` or `cpu` | `--device cuda` |
| transformers | Usually 4.x if pinned | Often 5.x if unpinned → **crash** |
| HF init device | `cpu` (no CUDA in HF model.py) | `cuda` |
| `torch.compile` in HF model | Often inert / different path | Active → weight key issues |
| Typical chunk latency (nfe=16) | ~10–80s on MPS | ~2–15s on RTX 4090 |

Output sample rate is always **24 kHz**.

---

### 7. Diagnostic commands

```bash
# Vocoder alone (should work)
python -c "from f5_tts.infer.utils_infer import load_vocoder; load_vocoder(device='cpu'); print('ok')"

# Full HF load (after pinning transformers)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ai4bharat/IndicF5', trust_remote_code=True); print('ok')"

# CUDA check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## References

We would like to extend our gratitude to the authors of  **[F5-TTS](https://github.com/SWivid/F5-TTS)** for their invaluable contributions and inspiration to this work. Their efforts have played a crucial role in advancing  the field of text-to-speech synthesis.


## 📖 Citation
If you use **IndicF5** in your research or projects, please consider citing it:

### 🔹 BibTeX
```bibtex
@misc{AI4Bharat_IndicF5_2025,
  author       = {Praveen S V and Srija Anand and Soma Siddhartha and Mitesh M. Khapra},
  title        = {IndicF5: High-Quality Text-to-Speech for Indian Languages},
  year         = {2025},
  url          = {https://github.com/AI4Bharat/IndicF5},
}

