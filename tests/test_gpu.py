import pytest
from unittest.mock import patch, MagicMock

from pdf_ocr.gpu import detect_gpus, GPUInfo, recommend_engine_kwargs


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


class TestRecommendEngineKwargs:
    def test_small_gpu_24gb(self):
        gpus = [GPUInfo(index=0, name="L4", vram_mb=24576)]
        kwargs = recommend_engine_kwargs(gpus)
        assert kwargs["max_num_batched_tokens"] == 4096
        assert kwargs["gpu_memory_utilization"] == 0.85

    def test_medium_gpu_48gb(self):
        gpus = [GPUInfo(index=0, name="A40", vram_mb=49152)]
        kwargs = recommend_engine_kwargs(gpus)
        assert kwargs["max_num_batched_tokens"] == 8192
        assert kwargs["gpu_memory_utilization"] == 0.90

    def test_large_gpu_80gb(self):
        gpus = [GPUInfo(index=0, name="A100", vram_mb=81920)]
        kwargs = recommend_engine_kwargs(gpus)
        assert kwargs["max_num_batched_tokens"] == 16384
        assert kwargs["gpu_memory_utilization"] == 0.90

    def test_no_gpus_returns_empty(self):
        kwargs = recommend_engine_kwargs([])
        assert kwargs == {}

    def test_single_gpu_no_data_parallel(self):
        gpus = [GPUInfo(index=0, name="A100", vram_mb=81920)]
        kwargs = recommend_engine_kwargs(gpus)
        assert "data_parallel_size" not in kwargs

    def test_multi_gpu_uses_first_gpu_vram(self):
        gpus = [
            GPUInfo(index=0, name="A100", vram_mb=81920),
            GPUInfo(index=1, name="A100", vram_mb=81920),
        ]
        kwargs = recommend_engine_kwargs(gpus)
        assert kwargs["max_num_batched_tokens"] == 16384
        assert kwargs["data_parallel_size"] == 2
