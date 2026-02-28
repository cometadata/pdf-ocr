# pdf_ocr/gpu.py
"""GPU detection and vLLM parameter tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

LOGGER = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    vram_mb: int


def detect_gpus() -> List[GPUInfo]:
    """Detect available CUDA GPUs and their VRAM."""
    try:
        if torch is None or not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                index=i,
                name=str(props.name),
                vram_mb=props.total_mem // (1024 * 1024),
            ))
        return gpus
    except Exception:
        LOGGER.debug("GPU detection failed", exc_info=True)
        return []
