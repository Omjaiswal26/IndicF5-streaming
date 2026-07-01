"""Load IndicF5 weights directly (bypasses HuggingFace AutoModel / _orig_mod mismatch)."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from f5_tts.infer.utils_infer import load_model, load_vocoder
from f5_tts.model import DiT

logger = logging.getLogger(__name__)

INDICF5_MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
EMA_PREFIX = "ema_model."
EMA_STRIP_PREFIX = "ema_model._orig_mod."


@dataclass
class LoadReport:
    model_id: str
    device: str
    vocab_path: str = ""
    checkpoint_path: str = ""
    checkpoint_keys: int = 0
    ema_keys_loaded: int = 0
    ema_missing: list[str] = field(default_factory=list)
    ema_unexpected: list[str] = field(default_factory=list)
    vocoder_source: str = "charactr/vocos-mel-24khz"
    load_seconds: float = 0.0

    @property
    def weights_ok(self) -> bool:
        return len(self.ema_missing) == 0 and len(self.ema_unexpected) == 0

    def log_summary(self) -> None:
        logger.info("=== IndicF5 load report ===")
        logger.info("model_id=%s device=%s", self.model_id, self.device)
        logger.info("vocab_path=%s", self.vocab_path)
        logger.info("checkpoint_path=%s", self.checkpoint_path)
        logger.info("checkpoint_keys=%d ema_keys_loaded=%d", self.checkpoint_keys, self.ema_keys_loaded)
        logger.info(
            "ema_missing=%d ema_unexpected=%d weights_ok=%s",
            len(self.ema_missing),
            len(self.ema_unexpected),
            self.weights_ok,
        )
        logger.info("vocoder_source=%s load_seconds=%.3f", self.vocoder_source, self.load_seconds)
        if self.ema_missing:
            logger.warning("ema_missing sample: %s", self.ema_missing[:5])
        if self.ema_unexpected:
            logger.warning("ema_unexpected sample: %s", self.ema_unexpected[:5])


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_ema_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map checkpoint keys ema_model._orig_mod.* -> CFM submodule keys."""
    ema_sd: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith(EMA_PREFIX):
            continue
        if key.startswith(EMA_STRIP_PREFIX):
            ema_sd[key[len(EMA_STRIP_PREFIX) :]] = value
        else:
            ema_sd[key[len(EMA_PREFIX) :]] = value
    return ema_sd


def load_indicf5_models(
    model_id: str = "ai4bharat/IndicF5",
    device: Optional[str] = None,
) -> tuple[torch.nn.Module, torch.nn.Module, LoadReport]:
    """
    Load ema_model (CFM) and vocoder with correct IndicF5 checkpoint weights.

    Vocoder weights come from charactr/vocos-mel-24khz, not the IndicF5 safetensors file.
    """
    resolved_device = device or _default_device()
    report = LoadReport(model_id=model_id, device=resolved_device)
    if not logging.getLogger().handlers:
        configure_logging()
    t0 = time.perf_counter()

    logger.info("Loading vocoder on %s", resolved_device)
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=resolved_device)

    logger.info("Downloading vocab.txt from %s", model_id)
    report.vocab_path = hf_hub_download(model_id, filename="checkpoints/vocab.txt")

    logger.info("Building empty CFM (DiT)")
    ema_model = load_model(
        DiT,
        INDICF5_MODEL_CFG,
        mel_spec_type="vocos",
        vocab_file=report.vocab_path,
        device=resolved_device,
    )

    logger.info("Downloading model.safetensors from %s", model_id)
    report.checkpoint_path = hf_hub_download(model_id, filename="model.safetensors")
    state_dict = load_file(report.checkpoint_path, device=resolved_device)
    report.checkpoint_keys = len(state_dict)

    ema_sd = _extract_ema_state_dict(state_dict)
    report.ema_keys_loaded = len(ema_sd)
    logger.info("Loading %d ema_model keys into CFM", report.ema_keys_loaded)

    missing, unexpected = ema_model.load_state_dict(ema_sd, strict=False)
    report.ema_missing = list(missing)
    report.ema_unexpected = list(unexpected)
    report.load_seconds = time.perf_counter() - t0

    report.log_summary()
    if not report.weights_ok:
        logger.error(
            "Weight load incomplete — audio will be garbage. "
            "Do not use AutoModel.from_pretrained for IndicF5 on CUDA."
        )

    return ema_model, vocoder, report


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
