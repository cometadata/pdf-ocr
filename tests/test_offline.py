import pytest
from unittest.mock import patch, MagicMock

from pdf_ocr.config import ModelConfig, InferenceConfig
from pdf_ocr.offline import (
    _cli_flag_to_kwarg,
    _coerce_value,
    build_engine_kwargs,
    VLLMOfflineEngine,
)


class TestCliFlagToKwarg:
    def test_simple_dash(self):
        assert _cli_flag_to_kwarg("trust-remote-code") == "trust_remote_code"

    def test_no_dashes(self):
        assert _cli_flag_to_kwarg("model") == "model"

    def test_multiple_dashes(self):
        assert _cli_flag_to_kwarg("gpu-memory-utilization") == "gpu_memory_utilization"


class TestCoerceValue:
    def test_json_dict_string(self):
        assert _coerce_value('{"image": 1}') == {"image": 1}

    def test_json_list_string(self):
        assert _coerce_value("[1, 2, 3]") == [1, 2, 3]

    def test_plain_string_passthrough(self):
        assert _coerce_value("hello") == "hello"

    def test_int_passthrough(self):
        assert _coerce_value(42) == 42

    def test_float_passthrough(self):
        assert _coerce_value(0.9) == 0.9

    def test_bool_passthrough(self):
        assert _coerce_value(True) is True

    def test_none_passthrough(self):
        assert _coerce_value(None) is None


class TestBuildEngineKwargs:
    def _make_config(self, vllm_args=None):
        return ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            vllm_args=vllm_args or {},
        )

    def test_basic_conversion(self):
        config = self._make_config({
            "trust-remote-code": True,
            "max-model-len": 4096,
            "gpu-memory-utilization": 0.90,
        })
        kwargs = build_engine_kwargs(config)
        assert kwargs["model"] == "test/model"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["max_model_len"] == 4096
        assert kwargs["gpu_memory_utilization"] == 0.90

    def test_json_string_parsed(self):
        config = self._make_config({
            "limit-mm-per-prompt": '{"image": 1}',
        })
        kwargs = build_engine_kwargs(config)
        assert kwargs["limit_mm_per_prompt"] == {"image": 1}

    def test_server_only_args_filtered(self):
        config = self._make_config({
            "served-model-name": "my-model",
            "host": "0.0.0.0",
            "port": 8000,
            "trust-remote-code": True,
        })
        kwargs = build_engine_kwargs(config)
        assert "served_model_name" not in kwargs
        assert "host" not in kwargs
        assert "port" not in kwargs
        assert kwargs["trust_remote_code"] is True

    def test_negation_prefix(self):
        config = self._make_config({
            "no-enable-prefix-caching": True,
        })
        kwargs = build_engine_kwargs(config)
        assert kwargs["enable_prefix_caching"] is False

    def test_negation_prefix_false(self):
        config = self._make_config({
            "no-enable-prefix-caching": False,
        })
        kwargs = build_engine_kwargs(config)
        assert kwargs["enable_prefix_caching"] is True

    def test_full_lighton_config(self):
        """Mirrors the current lighton_ocr_2_1b.yaml after optimization."""
        config = self._make_config({
            "limit-mm-per-prompt": '{"image": 1}',
            "mm-processor-cache-gb": 4,
            "enable-chunked-prefill": True,
            "max-model-len": 4096,
            "max-num-batched-tokens": 16384,
            "gpu-memory-utilization": 0.90,
            "trust-remote-code": True,
        })
        kwargs = build_engine_kwargs(config)
        assert kwargs["model"] == "test/model"
        assert kwargs["limit_mm_per_prompt"] == {"image": 1}
        assert kwargs["mm_processor_cache_gb"] == 4
        assert kwargs["enable_chunked_prefill"] is True
        assert kwargs["max_model_len"] == 4096
        assert kwargs["max_num_batched_tokens"] == 16384
        assert kwargs["gpu_memory_utilization"] == 0.90
        assert kwargs["trust_remote_code"] is True
        # prefix caching no longer explicitly disabled
        assert "enable_prefix_caching" not in kwargs


class TestVLLMOfflineEngine:
    @patch("pdf_ocr.offline.LLM", create=True)
    @patch("pdf_ocr.offline.SamplingParams", create=True)
    def test_init_passes_correct_kwargs(self, MockSamplingParams, MockLLM):
        mock_llm_instance = MagicMock()
        mock_sp_instance = MagicMock()

        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            import sys
            mock_vllm = sys.modules["vllm"]
            mock_vllm.LLM = MockLLM
            mock_vllm.SamplingParams = MockSamplingParams
            MockLLM.return_value = mock_llm_instance
            MockSamplingParams.return_value = mock_sp_instance

            config = ModelConfig(
                model_id="test/model",
                served_model_name="test-model",
                vllm_args={
                    "trust-remote-code": True,
                    "max-model-len": 4096,
                },
            )
            engine = VLLMOfflineEngine(config)

            MockLLM.assert_called_once_with(
                model="test/model",
                trust_remote_code=True,
                max_model_len=4096,
            )
            MockSamplingParams.assert_called_once_with(
                max_tokens=4000,
                temperature=0.2,
                top_p=0.9,
            )

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_infer_batch_returns_text(self):
        from PIL import Image
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_llm_instance = MagicMock()
        mock_vllm.LLM.return_value = mock_llm_instance

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="# Hello World")]
        mock_llm_instance.chat.return_value = [mock_output]

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            vllm_args={"trust-remote-code": True},
        )
        engine = VLLMOfflineEngine(config)

        img = Image.new("RGB", (100, 100))
        results = engine.infer_batch([img])

        assert results == ["# Hello World"]
        mock_llm_instance.chat.assert_called_once()

        call_args = mock_llm_instance.chat.call_args
        messages_list = call_args[1].get("messages") or call_args[0][0]
        image_url = messages_list[0][0]["content"][0]["image_url"]["url"]
        assert isinstance(image_url, str) and image_url.startswith("data:image/png;base64,")

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_infer_batch_empty(self):
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_vllm.LLM.return_value = MagicMock()

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
        )
        engine = VLLMOfflineEngine(config)

        results = engine.infer_batch([])
        assert results == []

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_extra_body_warns(self):
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_vllm.LLM.return_value = MagicMock()

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            inference=InferenceConfig(extra_body={"skip_special_tokens": False}),
        )

        with patch("pdf_ocr.offline.LOGGER") as mock_logger:
            engine = VLLMOfflineEngine(config)
            mock_logger.warning.assert_called_once()
