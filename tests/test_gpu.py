# tests/test_gpu.py
import pytest
from unittest.mock import patch, MagicMock

from pdf_ocr.gpu import detect_gpus, GPUInfo


class TestDetectGpus:
    @patch("pdf_ocr.gpu.torch")
    def test_no_cuda_returns_empty(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        result = detect_gpus()
        assert result == []

    @patch("pdf_ocr.gpu.torch")
    def test_single_gpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            name="NVIDIA A100", total_mem=85899345920  # 80 GB
        )
        result = detect_gpus()
        assert len(result) == 1
        assert result[0].index == 0
        assert result[0].vram_mb == 81920

    @patch("pdf_ocr.gpu.torch")
    def test_multi_gpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            name="NVIDIA L4", total_mem=25769803776  # 24 GB
        )
        result = detect_gpus()
        assert len(result) == 4

    @patch("pdf_ocr.gpu.torch")
    def test_import_error_returns_empty(self, mock_torch):
        mock_torch.cuda.is_available.side_effect = RuntimeError("no CUDA")
        result = detect_gpus()
        assert result == []
