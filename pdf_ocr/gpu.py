from __future__ import annotations

import logging
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


def detect_gpus() -> List[GPUInfo]:
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


def recommend_engine_kwargs(gpus: List[GPUInfo]) -> Dict[str, Any]:
    """Return default vLLM engine kwargs; user YAML config takes precedence."""
    if not gpus:
        return {}

    vram_mb = gpus[0].vram_mb

    if vram_mb >= 65536:  # 64 GB+
        max_num_batched_tokens = 16384
        gpu_memory_utilization = 0.90
    elif vram_mb >= 30720:  # 30 GB+
        max_num_batched_tokens = 8192
        gpu_memory_utilization = 0.90
    else:
        max_num_batched_tokens = 4096
        gpu_memory_utilization = 0.85

    kwargs = {
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
    }

    if len(gpus) > 1:
        kwargs["data_parallel_size"] = len(gpus)

    return kwargs
