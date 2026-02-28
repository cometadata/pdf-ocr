from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:
    from PIL import Image

LOGGER = logging.getLogger(__name__)


class DataParallelEngine:
    """Each replica is pinned to a separate GPU.

    Implements the same ``infer_batch`` interface as ``VLLMOfflineEngine``.
    """

    def __init__(self, replicas: list) -> None:
        if not replicas:
            raise ValueError("DataParallelEngine requires at least one replica")
        self._replicas = replicas
        self._next_replica = 0
        LOGGER.info("DataParallelEngine initialized with %d replicas", len(replicas))

    def infer_batch(self, images: Sequence["Image.Image"]) -> List[str]:
        if not images:
            return []
        replica = self._replicas[self._next_replica]
        self._next_replica = (self._next_replica + 1) % len(self._replicas)
        return replica.infer_batch(images)
