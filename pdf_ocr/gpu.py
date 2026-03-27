from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

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


def _gpu_count_from_env() -> int:
    """Fallback GPU count from CUDA_VISIBLE_DEVICES or torch.cuda.device_count()."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        devices = [d.strip() for d in cvd.split(",") if d.strip()]
        if devices:
            return len(devices)
    if torch is not None:
        try:
            return torch.cuda.device_count()
        except Exception:
            pass
    return 0


def detect_gpus() -> List[GPUInfo]:
    if torch is None or not torch.cuda.is_available():
        return []
    try:
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                index=i,
                name=str(props.name),
                vram_mb=props.total_memory // (1024 * 1024),
            ))
        return gpus
    except Exception:
        LOGGER.warning("GPU property detection failed, falling back to device count", exc_info=True)
        count = _gpu_count_from_env()
        return [GPUInfo(index=i, name="unknown", vram_mb=0) for i in range(count)]


def get_physical_gpu_ids() -> List[str]:
    """Return physical GPU IDs from CUDA_VISIBLE_DEVICES or torch device count."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        devices = [d.strip() for d in cvd.split(",") if d.strip()]
        if devices:
            return devices
    if torch is not None:
        try:
            count = torch.cuda.device_count()
            return [str(i) for i in range(count)]
        except Exception:
            pass
    return []


def recommend_engine_kwargs(gpus: List[GPUInfo]) -> Dict[str, Any]:
    """Return default vLLM engine kwargs; user YAML config takes precedence."""
    if not gpus:
        return {}

    kwargs: Dict[str, Any] = {}

    vram_mb = gpus[0].vram_mb
    if vram_mb > 0:
        if vram_mb >= 131072:  # 128 GB+ (H200, etc.)
            kwargs["max_num_batched_tokens"] = 16384
            kwargs["gpu_memory_utilization"] = 0.95
        elif vram_mb >= 65536:  # 64 GB+
            kwargs["max_num_batched_tokens"] = 16384
            kwargs["gpu_memory_utilization"] = 0.90
        elif vram_mb >= 30720:  # 30 GB+
            kwargs["max_num_batched_tokens"] = 8192
            kwargs["gpu_memory_utilization"] = 0.90
        else:
            kwargs["max_num_batched_tokens"] = 4096
            kwargs["gpu_memory_utilization"] = 0.85

    return kwargs
