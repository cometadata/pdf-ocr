from unittest.mock import patch, MagicMock

from pdf_ocr.config import ModelConfig


class TestCreateOfflineEngine:
    def _make_config(self):
        return ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            vllm_args={"trust-remote-code": True},
        )

    @patch("pdf_ocr.engine_factory.VLLMOfflineEngine")
    def test_returns_vllm_offline_engine(self, MockEngine):
        from pdf_ocr.engine_factory import create_offline_engine

        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance

        engine = create_offline_engine(self._make_config())
        assert engine is mock_instance
        MockEngine.assert_called_once()
