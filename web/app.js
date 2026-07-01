const SAMPLE_RATE = 24000;

const els = {
  serverUrl: document.getElementById("serverUrl"),
  text: document.getElementById("text"),
  refText: document.getElementById("refText"),
  refAudioPath: document.getElementById("refAudioPath"),
  nfeStep: document.getElementById("nfeStep"),
  maxChars: document.getElementById("maxChars"),
  generateBtn: document.getElementById("generateBtn"),
  stopBtn: document.getElementById("stopBtn"),
  status: document.getElementById("status"),
  meterFill: document.getElementById("meterFill"),
};

let audioContext = null;
let abortController = null;
let scheduledSources = [];

function setStatus(message, kind = "idle") {
  els.status.textContent = message;
  els.status.className = `status ${kind}`;
}

function setBusy(busy) {
  els.generateBtn.disabled = busy;
  els.stopBtn.disabled = !busy;
}

function resetPlayback() {
  scheduledSources.forEach((src) => {
    try {
      src.stop();
    } catch (_) {
      /* already stopped */
    }
  });
  scheduledSources = [];
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
}

function appendBuffer(existing, chunk) {
  const merged = new Uint8Array(existing.length + chunk.length);
  merged.set(existing);
  merged.set(chunk, existing.length);
  return merged;
}

function parseFrames(buffer) {
  const frames = [];
  let offset = 0;

  while (buffer.length - offset >= 4) {
    const view = new DataView(buffer.buffer, buffer.byteOffset + offset, buffer.length - offset);
    const sampleCount = view.getUint32(0, true);
    const frameSize = 4 + sampleCount * 4;

    if (buffer.length - offset < frameSize) {
      break;
    }

    const floats = new Float32Array(sampleCount);
    for (let i = 0; i < sampleCount; i += 1) {
      floats[i] = view.getFloat32(4 + i * 4, true);
    }
    frames.push(floats);
    offset += frameSize;
  }

  return { frames, remainder: buffer.slice(offset) };
}

function scheduleChunk(floats, startTime) {
  const buffer = audioContext.createBuffer(1, floats.length, SAMPLE_RATE);
  buffer.copyToChannel(floats, 0);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  source.start(startTime);
  scheduledSources.push(source);

  return startTime + buffer.duration;
}

async function streamAndPlay() {
  resetPlayback();
  abortController = new AbortController();

  const baseUrl = els.serverUrl.value.replace(/\/$/, "");
  const payload = {
    text: els.text.value.trim(),
    ref_audio_path: els.refAudioPath.value.trim(),
    ref_text: els.refText.value.trim(),
    max_chars: Number(els.maxChars.value) || 120,
    "nfe-step": Number(els.nfeStep.value) || 32,
  };

  if (!payload.text) {
    setStatus("Please enter text to synthesize.", "error");
    return;
  }

  setBusy(true);
  els.meterFill.style.width = "8%";
  setStatus("Connecting to server…", "busy");

  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioContext.resume();

  const startedAt = performance.now();
  let nextPlayTime = audioContext.currentTime + 0.05;
  let chunkCount = 0;
  let totalSamples = 0;
  let pending = new Uint8Array(0);

  try {
    const response = await fetch(`${baseUrl}/tts/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status} ${response.statusText}`);
    }

    const reader = response.body.getReader();
    setStatus("Generating first chunk… (this can take a minute on CPU/MPS)", "busy");

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      pending = appendBuffer(pending, value);
      const { frames, remainder } = parseFrames(pending);
      pending = remainder;

      for (const frame of frames) {
        chunkCount += 1;
        totalSamples += frame.length;
        nextPlayTime = scheduleChunk(frame, nextPlayTime);

        const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
        const duration = (totalSamples / SAMPLE_RATE).toFixed(1);
        els.meterFill.style.width = `${Math.min(95, 10 + chunkCount * 20)}%`;
        setStatus(
          `Playing chunk ${chunkCount} · ${duration}s audio · ${elapsed}s elapsed`,
          "busy",
        );
      }
    }

    if (chunkCount === 0) {
      throw new Error("No audio chunks received from server.");
    }

    const totalDuration = (totalSamples / SAMPLE_RATE).toFixed(1);
    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
    els.meterFill.style.width = "100%";
    setStatus(`Done · ${chunkCount} chunk(s) · ${totalDuration}s audio · ${elapsed}s total`, "done");
  } catch (error) {
    if (error.name === "AbortError") {
      setStatus("Stopped.", "idle");
      els.meterFill.style.width = "0%";
    } else {
      setStatus(error.message || String(error), "error");
      els.meterFill.style.width = "0%";
    }
  } finally {
    setBusy(false);
    abortController = null;
  }
}

function stopPlayback() {
  if (abortController) {
    abortController.abort();
  }
  resetPlayback();
  els.meterFill.style.width = "0%";
  setStatus("Stopped.", "idle");
}

els.generateBtn.addEventListener("click", streamAndPlay);
els.stopBtn.addEventListener("click", stopPlayback);

setStatus('Ready. Click "Generate & Play" to stream audio from the server.', "idle");
