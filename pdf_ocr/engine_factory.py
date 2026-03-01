from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import ModelConfig
from .offline import VLLMOfflineEngine

if TYPE_CHECKING:
    from .convert import InferenceEngine

LOGGER = logging.getLogger(__name__)


def create_offline_engine(config: ModelConfig) -> "InferenceEngine":
    return VLLMOfflineEngine(config)
