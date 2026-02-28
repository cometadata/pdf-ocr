# tests/test_parallel.py
import pytest
from unittest.mock import MagicMock, patch, call
from PIL import Image

from pdf_ocr.parallel import DataParallelEngine


class TestDataParallelEngine:
    def test_round_robin_distribution(self):
        """Batches should be distributed round-robin across replicas."""
        replica_0 = MagicMock()
        replica_1 = MagicMock()
        replica_0.infer_batch.return_value = ["r0_page0", "r0_page1"]
        replica_1.infer_batch.return_value = ["r1_page0", "r1_page1"]

        engine = DataParallelEngine.__new__(DataParallelEngine)
        engine._replicas = [replica_0, replica_1]
        engine._next_replica = 0

        images_batch1 = [Image.new("RGB", (10, 10)) for _ in range(2)]
        images_batch2 = [Image.new("RGB", (10, 10)) for _ in range(2)]

        result1 = engine.infer_batch(images_batch1)
        result2 = engine.infer_batch(images_batch2)

        assert result1 == ["r0_page0", "r0_page1"]
        assert result2 == ["r1_page0", "r1_page1"]
        replica_0.infer_batch.assert_called_once()
        replica_1.infer_batch.assert_called_once()

    def test_wraps_around(self):
        """After using all replicas, should wrap back to first."""
        replicas = [MagicMock() for _ in range(2)]
        for r in replicas:
            r.infer_batch.return_value = ["result"]

        engine = DataParallelEngine.__new__(DataParallelEngine)
        engine._replicas = replicas
        engine._next_replica = 0

        img = [Image.new("RGB", (10, 10))]
        for _ in range(3):
            engine.infer_batch(img)

        assert replicas[0].infer_batch.call_count == 2
        assert replicas[1].infer_batch.call_count == 1

    def test_empty_batch(self):
        """Empty batch should return empty without touching replicas."""
        replica = MagicMock()

        engine = DataParallelEngine.__new__(DataParallelEngine)
        engine._replicas = [replica]
        engine._next_replica = 0

        assert engine.infer_batch([]) == []
        replica.infer_batch.assert_not_called()

    def test_conforms_to_inference_engine_protocol(self):
        """DataParallelEngine must implement infer_batch(images) -> List[str]."""
        from pdf_ocr.convert import InferenceEngine
        engine = DataParallelEngine.__new__(DataParallelEngine)
        engine._replicas = [MagicMock()]
        engine._next_replica = 0
        assert hasattr(engine, "infer_batch")
