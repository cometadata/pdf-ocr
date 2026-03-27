import pytest
from unittest.mock import patch, MagicMock

from pdf_ocr.gpu import detect_gpus, GPUInfo, recommend_engine_kwargs, _gpu_count_from_env, get_physical_gpu_ids


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
            name="NVIDIA A100", total_memory=85899345920  # 80 GB
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
            name="NVIDIA L4", total_memory=25769803776  # 24 GB
        )
        result = detect_gpus()
        assert len(result) == 4

    @patch("pdf_ocr.gpu.torch")
    def test_properties_failure_falls_back_to_count(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        mock_torch.cuda.get_device_properties.side_effect = RuntimeError("CUDA error")
        result = detect_gpus()
        assert len(result) == 4
        assert all(g.name == "unknown" for g in result)
        assert all(g.vram_mb == 0 for g in result)

    @patch("pdf_ocr.gpu.torch", None)
    def test_no_torch_returns_empty(self):
        result = detect_gpus()
        assert result == []


class TestGpuCountFromEnv:
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    def test_reads_cuda_visible_devices(self):
        assert _gpu_count_from_env() == 4

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"})
    def test_single_device(self):
        assert _gpu_count_from_env() == 1

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""})
    @patch("pdf_ocr.gpu.torch")
    def test_empty_env_falls_back_to_torch(self, mock_torch):
        mock_torch.cuda.device_count.return_value = 2
        assert _gpu_count_from_env() == 2

    @patch.dict("os.environ", {}, clear=True)
    @patch("pdf_ocr.gpu.torch", None)
    def test_no_env_no_torch_returns_zero(self):
        # Remove CUDA_VISIBLE_DEVICES if present
        import os
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        assert _gpu_count_from_env() == 0


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

    def test_very_large_gpu_141gb(self):
        gpus = [GPUInfo(index=0, name="H200", vram_mb=144384)]
        kwargs = recommend_engine_kwargs(gpus)
        assert kwargs["max_num_batched_tokens"] == 16384
        assert kwargs["gpu_memory_utilization"] == 0.95

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
        assert "data_parallel_size" not in kwargs

    def test_unknown_vram_skips_batched_tokens(self):
        gpus = [
            GPUInfo(index=0, name="unknown", vram_mb=0),
            GPUInfo(index=1, name="unknown", vram_mb=0),
        ]
        kwargs = recommend_engine_kwargs(gpus)
        assert "max_num_batched_tokens" not in kwargs
        assert "gpu_memory_utilization" not in kwargs
        assert "data_parallel_size" not in kwargs


class TestGetPhysicalGpuIds:
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    def test_reads_env_var(self):
        assert get_physical_gpu_ids() == ["0", "1", "2", "3"]

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,5,7"})
    def test_non_contiguous_ids(self):
        assert get_physical_gpu_ids() == ["2", "5", "7"]

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-8aa8f89a-1234"})
    def test_uuid_style_ids(self):
        assert get_physical_gpu_ids() == ["GPU-8aa8f89a-1234"]

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""})
    @patch("pdf_ocr.gpu.torch")
    def test_empty_env_falls_back_to_torch(self, mock_torch):
        mock_torch.cuda.device_count.return_value = 2
        assert get_physical_gpu_ids() == ["0", "1"]

    @patch.dict("os.environ", {}, clear=True)
    @patch("pdf_ocr.gpu.torch", None)
    def test_no_env_no_torch_returns_empty(self):
        import os
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        assert get_physical_gpu_ids() == []
