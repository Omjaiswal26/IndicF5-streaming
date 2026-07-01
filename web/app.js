const SAMPLE_RATE = 24000;
const SCHEDULE_LEAD_S = 0.03;

const LATENCY_PRESETS = {
  low: { nfeStep: 8, maxChars: 60, maxCharsPreset: "60" },
  balanced: { nfeStep: 16, maxChars: 80, maxCharsPreset: "80" },
  quality: { nfeStep: 32, maxChars: 120, maxCharsPreset: "120" },
};

const els = {
  serverUrl: document.getElementById("serverUrl"),
  text: document.getElementById("text"),
  latencyPreset: document.getElementById("latencyPreset"),
  nfeStep: document.getElementById("nfeStep"),
  maxCharsPreset: document.getElementById("maxCharsPreset"),
  maxChars: document.getElementById("maxChars"),
  useServerNfe: document.getElementById("useServerNfe"),
  serverDefaults: document.getElementById("serverDefaults"),
  serverNfeInline: document.getElementById("serverNfeInline"),
  generateBtn: document.getElementById("generateBtn"),
  stopBtn: document.getElementById("stopBtn"),
  status: document.getElementById("status"),
  meterFill: document.getElementById("meterFill"),
};

let serverDefaultNfe = 16;
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

function scheduleChunk(floats, nextPlayTime) {
  const buffer = audioContext.createBuffer(1, floats.length, SAMPLE_RATE);
  buffer.copyToChannel(floats, 0);

  const startTime = Math.max(nextPlayTime, audioContext.currentTime + SCHEDULE_LEAD_S);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  source.start(startTime);
  scheduledSources.push(source);

  return startTime + buffer.duration;
}

function apiBaseUrl() {
  const configured = els.serverUrl.value.trim().replace(/\/$/, "");
  return configured || window.location.origin;
}

function resolvedMaxChars() {
  if (els.maxCharsPreset.value === "custom") {
    return Math.min(300, Math.max(30, Number(els.maxChars.value) || 60));
  }
  return Number(els.maxCharsPreset.value);
}

function syncMaxCharsInputVisibility() {
  const isCustom = els.maxCharsPreset.value === "custom";
  els.maxChars.classList.toggle("hidden", !isCustom);
}

function applyLatencyPreset(presetKey) {
  if (presetKey === "custom") {
    return;
  }
  const preset = LATENCY_PRESETS[presetKey];
  if (!preset) {
    return;
  }
  els.nfeStep.value = String(preset.nfeStep);
  els.maxCharsPreset.value = preset.maxCharsPreset;
  els.maxChars.value = String(preset.maxChars);
  syncMaxCharsInputVisibility();
}

function markCustomPreset() {
  els.latencyPreset.value = "custom";
}

function updateServerNfeDisplay() {
  const label = String(serverDefaultNfe);
  els.serverNfeInline.textContent = label;
  els.serverDefaults.textContent =
    `Server default NFE: ${label} (from --nfe-step or tts_server.py default). ` +
    `Effective NFE for next request: ${effectiveNfeLabel()}.`;
}

function effectiveNfeLabel() {
  if (els.useServerNfe.checked) {
    return `server default (${serverDefaultNfe})`;
  }
  return els.nfeStep.value;
}

function buildPayload() {
  const payload = {
    text: els.text.value.trim(),
    max_chars: resolvedMaxChars(),
    split: true,
  };

  if (!els.useServerNfe.checked) {
    payload["nfe-step"] = Number(els.nfeStep.value);
  }

  return payload;
}

function logClientMetrics(metrics) {
  console.info("[tts-metrics]", metrics);
}

async function fetchServerDefaults() {
  try {
    const res = await fetch(`${apiBaseUrl()}/health`);
    if (!res.ok) {
      return;
    }
    const data = await res.json();
    if (typeof data.default_nfe_step === "number") {
      serverDefaultNfe = data.default_nfe_step;
      updateServerNfeDisplay();
    }
  } catch (_) {
    els.serverDefaults.textContent = "Server default NFE: unknown (could not reach /health)";
    els.serverNfeInline.textContent = "?";
  }
}

async function streamAndPlay() {
  resetPlayback();
  abortController = new AbortController();

  const baseUrl = apiBaseUrl();
  const payload = buildPayload();

  if (!payload.text) {
    setStatus("Please enter text to synthesize.", "error");
    return;
  }

  setBusy(true);
  els.meterFill.style.width = "8%";
  setStatus(
    `Connecting… (nfe=${effectiveNfeLabel()}, max_chars=${payload.max_chars})`,
    "busy",
  );

  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioContext.resume();

  const startedAt = performance.now();
  let nextPlayTime = audioContext.currentTime + SCHEDULE_LEAD_S;
  let chunkCount = 0;
  let totalSamples = 0;
  let pending = new Uint8Array(0);
  let firstChunkAt = null;
  const chunkArrivalTimes = [];

  try {
    const fetchStart = performance.now();
    const response = await fetch(`${baseUrl}/tts/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Request-Id": `web-${Date.now()}`,
      },
      body: JSON.stringify(payload),
      signal: abortController.signal,
    });

    const headersReceivedAt = performance.now();
    const responseNfe = response.headers.get("X-NFE-Step");

    if (!response.ok) {
      throw new Error(`Server returned ${response.status} ${response.statusText}`);
    }

    const reader = response.body.getReader();
    setStatus("Generating first chunk…", "busy");

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      pending = appendBuffer(pending, value);
      const { frames, remainder } = parseFrames(pending);
      pending = remainder;

      for (const frame of frames) {
        const arrivalAt = performance.now();
        if (firstChunkAt === null) {
          firstChunkAt = arrivalAt;
        }
        chunkArrivalTimes.push({
          index: chunkCount + 1,
          arrivalMs: arrivalAt - startedAt,
          samples: frame.length,
          audioSeconds: frame.length / SAMPLE_RATE,
        });

        chunkCount += 1;
        totalSamples += frame.length;
        nextPlayTime = scheduleChunk(frame, nextPlayTime);

        const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
        const duration = (totalSamples / SAMPLE_RATE).toFixed(1);
        const ttft = firstChunkAt ? ((firstChunkAt - startedAt) / 1000).toFixed(1) : "?";
        els.meterFill.style.width = `${Math.min(95, 10 + chunkCount * 20)}%`;
        setStatus(
          `Chunk ${chunkCount} · TTFT ${ttft}s · ${duration}s audio · ${elapsed}s elapsed`,
          "busy",
        );
      }
    }

    if (chunkCount === 0) {
      throw new Error("No audio chunks received from server.");
    }

    const totalDuration = (totalSamples / SAMPLE_RATE).toFixed(1);
    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
    const ttft = ((firstChunkAt - startedAt) / 1000).toFixed(1);
    const ttfb = ((headersReceivedAt - fetchStart) / 1000).toFixed(1);

    els.meterFill.style.width = "100%";
    setStatus(
      `Done · ${chunkCount} chunk(s) · ${totalDuration}s audio · TTFT ${ttft}s · ${elapsed}s total`,
      "done",
    );

    logClientMetrics({
      ttfbSeconds: Number(ttfb),
      ttftSeconds: Number(ttft),
      totalSeconds: Number(elapsed),
      totalAudioSeconds: Number(totalDuration),
      chunkCount,
      maxChars: payload.max_chars,
      nfeStepRequested: payload["nfe-step"] ?? `server-default(${serverDefaultNfe})`,
      nfeStepResponseHeader: responseNfe,
      useServerNfe: els.useServerNfe.checked,
      chunkArrivals: chunkArrivalTimes,
    });
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

els.latencyPreset.addEventListener("change", () => {
  applyLatencyPreset(els.latencyPreset.value);
  updateServerNfeDisplay();
});

els.nfeStep.addEventListener("change", () => {
  markCustomPreset();
  updateServerNfeDisplay();
});

els.maxCharsPreset.addEventListener("change", () => {
  markCustomPreset();
  syncMaxCharsInputVisibility();
  if (els.maxCharsPreset.value !== "custom") {
    els.maxChars.value = els.maxCharsPreset.value;
  }
});

els.maxChars.addEventListener("input", markCustomPreset);
els.useServerNfe.addEventListener("change", updateServerNfeDisplay);

els.generateBtn.addEventListener("click", streamAndPlay);
els.stopBtn.addEventListener("click", stopPlayback);

if (window.location.protocol !== "file:") {
  els.serverUrl.placeholder = window.location.origin;
}

applyLatencyPreset("low");
syncMaxCharsInputVisibility();
fetchServerDefaults();
updateServerNfeDisplay();

setStatus('Ready. "Low latency" preset selected (NFE 8, max 60 chars). Click Generate & Play.', "idle");
