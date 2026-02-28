import pytest
from unittest.mock import patch, MagicMock

from pdf_ocr.config import ModelConfig, InferenceConfig
from pdf_ocr.gpu import GPUInfo


class TestCreateOfflineEngine:
    def _make_config(self, auto_parallel=True):
        return ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            vllm_args={"trust-remote-code": True},
            inference=InferenceConfig(auto_parallel=auto_parallel),
        )

    @patch("pdf_ocr.engine_factory.detect_gpus")
    @patch("pdf_ocr.engine_factory.VLLMOfflineEngine")
    def test_single_gpu_returns_plain_engine(self, MockEngine, mock_detect):
        from pdf_ocr.engine_factory import create_offline_engine

        mock_detect.return_value = [GPUInfo(index=0, name="A100", vram_mb=81920)]
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance

        engine = create_offline_engine(self._make_config())
        assert engine is mock_instance
        MockEngine.assert_called_once()

    @patch("pdf_ocr.engine_factory.detect_gpus")
    @patch("pdf_ocr.engine_factory.VLLMOfflineEngine")
    def test_no_gpu_returns_plain_engine(self, MockEngine, mock_detect):
        from pdf_ocr.engine_factory import create_offline_engine

        mock_detect.return_value = []
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance

        engine = create_offline_engine(self._make_config())
        assert engine is mock_instance

    @patch("pdf_ocr.engine_factory.detect_gpus")
    @patch("pdf_ocr.engine_factory.VLLMOfflineEngine")
    @patch("pdf_ocr.engine_factory.DataParallelEngine")
    def test_multi_gpu_returns_data_parallel(self, MockDP, MockEngine, mock_detect):
        from pdf_ocr.engine_factory import create_offline_engine

        mock_detect.return_value = [
            GPUInfo(index=0, name="L4", vram_mb=24576),
            GPUInfo(index=1, name="L4", vram_mb=24576),
        ]
        mock_engine_0 = MagicMock()
        mock_engine_1 = MagicMock()
        MockEngine.side_effect = [mock_engine_0, mock_engine_1]

        mock_dp = MagicMock()
        MockDP.return_value = mock_dp

        engine = create_offline_engine(self._make_config())
        assert engine is mock_dp
        assert MockEngine.call_count == 2
        MockDP.assert_called_once()

    @patch("pdf_ocr.engine_factory.detect_gpus")
    @patch("pdf_ocr.engine_factory.VLLMOfflineEngine")
    def test_auto_parallel_disabled_returns_plain(self, MockEngine, mock_detect):
        from pdf_ocr.engine_factory import create_offline_engine

        mock_detect.return_value = [
            GPUInfo(index=0, name="L4", vram_mb=24576),
            GPUInfo(index=1, name="L4", vram_mb=24576),
        ]
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance

        engine = create_offline_engine(self._make_config(auto_parallel=False))
        assert engine is mock_instance
        MockEngine.assert_called_once()

    @patch("pdf_ocr.engine_factory.detect_gpus")
    @patch("pdf_ocr.engine_factory.VLLMOfflineEngine")
    def test_yaml_tensor_parallel_skips_data_parallel(self, MockEngine, mock_detect):
        from pdf_ocr.engine_factory import create_offline_engine

        mock_detect.return_value = [
            GPUInfo(index=0, name="A100", vram_mb=81920),
            GPUInfo(index=1, name="A100", vram_mb=81920),
        ]
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            vllm_args={"trust-remote-code": True, "tensor-parallel-size": 2},
        )
        engine = create_offline_engine(config)
        assert engine is mock_instance
        MockEngine.assert_called_once()
