# pdf_ocr/engine_factory.py
"""Factory for creating the appropriate offline inference engine."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from .config import ModelConfig
from .gpu import detect_gpus
from .offline import VLLMOfflineEngine
from .parallel import DataParallelEngine

if TYPE_CHECKING:
    from .convert import InferenceEngine

LOGGER = logging.getLogger(__name__)


def create_offline_engine(config: ModelConfig) -> "InferenceEngine":
    """Create the best offline engine for the current hardware.

    Decision logic:
    1. If user set tensor-parallel-size in YAML, respect it (single engine).
    2. If auto_parallel is False, use single engine.
    3. If multiple GPUs detected and model fits on one GPU, use DataParallelEngine.
    4. Otherwise, use single VLLMOfflineEngine.
    """
    gpus = detect_gpus()
    gpu_count = len(gpus)

    # Check if user explicitly set tensor parallelism
    user_tp = config.vllm_args.get("tensor-parallel-size")
    if user_tp is not None and int(user_tp) > 1:
        LOGGER.info(
            "User set tensor-parallel-size=%s; using single engine", user_tp
        )
        return VLLMOfflineEngine(config)

    # Check if auto parallelism is disabled
    if not config.inference.auto_parallel:
        LOGGER.info("auto_parallel disabled; using single engine")
        return VLLMOfflineEngine(config)

    # Multi-GPU data parallelism
    if gpu_count > 1:
        LOGGER.info(
            "Detected %d GPUs; creating DataParallelEngine with %d replicas",
            gpu_count, gpu_count,
        )
        replicas = []
        for gpu in gpus:
            # Pin each replica to a specific GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
            replica = VLLMOfflineEngine(config)
            replicas.append(replica)
        # Restore visibility to all GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(g.index) for g in gpus
        )
        return DataParallelEngine(replicas)

    # Single GPU or no GPU
    return VLLMOfflineEngine(config)
