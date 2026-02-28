import pytest
from unittest.mock import patch, MagicMock

from pdf_ocr.config import ModelConfig, InferenceConfig
from pdf_ocr.offline import (
    _cli_flag_to_kwarg,
    _coerce_value,
    build_engine_kwargs,
    VLLMOfflineEngine,
)
from pdf_ocr.gpu import GPUInfo

# Force-import gpu module (and transitively torch) at module level so that
# subsequent sys.modules patches inside tests don't trigger a partial torch
# re-import which fails with a docstring RuntimeError.
try:
    import pdf_ocr.gpu  # noqa: F401
except Exception:
    pass


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
            "no-enable-prefix-caching": True,
            "mm-processor-cache-gb": 0,
            "enable-chunked-prefill": True,
            "max-model-len": 4096,
            "gpu-memory-utilization": 0.90,
            "trust-remote-code": True,
        })
        kwargs = build_engine_kwargs(config)
        assert kwargs["model"] == "test/model"
        assert kwargs["limit_mm_per_prompt"] == {"image": 1}
        assert kwargs["enable_prefix_caching"] is False
        assert kwargs["mm_processor_cache_gb"] == 0
        assert kwargs["enable_chunked_prefill"] is True
        assert kwargs["max_model_len"] == 4096
        assert kwargs["gpu_memory_utilization"] == 0.90
        assert kwargs["trust_remote_code"] is True
        assert "max_num_batched_tokens" not in kwargs


class TestBuildEngineKwargsAutoDefaults:
    def _make_config(self, vllm_args=None):
        return ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            vllm_args=vllm_args or {},
        )

    def test_auto_kwargs_applied_when_no_yaml_override(self):
        config = self._make_config({"trust-remote-code": True})
        auto = {"max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.90}
        kwargs = build_engine_kwargs(config, auto_kwargs=auto)
        assert kwargs["max_num_batched_tokens"] == 8192
        assert kwargs["gpu_memory_utilization"] == 0.90
        assert kwargs["trust_remote_code"] is True

    def test_yaml_overrides_auto_kwargs(self):
        config = self._make_config({
            "gpu-memory-utilization": 0.85,
            "trust-remote-code": True,
        })
        auto = {"max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.90}
        kwargs = build_engine_kwargs(config, auto_kwargs=auto)
        # YAML wins over auto
        assert kwargs["gpu_memory_utilization"] == 0.85
        # Auto still applies where no YAML override
        assert kwargs["max_num_batched_tokens"] == 8192

    def test_no_auto_kwargs_same_as_before(self):
        config = self._make_config({"trust-remote-code": True})
        kwargs = build_engine_kwargs(config)
        assert kwargs == {"model": "test/model", "trust_remote_code": True}


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
        assert isinstance(image_url, str) and image_url.startswith("data:image/jpeg;base64,")

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

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_thread_pool_uses_max_encode_workers(self):
        """ThreadPoolExecutor should receive max_workers from config."""
        from PIL import Image
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_llm_instance = MagicMock()
        mock_vllm.LLM.return_value = mock_llm_instance

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="# Result")]
        mock_llm_instance.chat.return_value = [mock_output]

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            vllm_args={"trust-remote-code": True},
            inference=InferenceConfig(max_encode_workers=4),
        )
        engine = VLLMOfflineEngine(config)

        with patch("pdf_ocr.offline.ThreadPoolExecutor") as MockPool:
            mock_pool_instance = MagicMock()
            mock_pool_instance.__enter__ = MagicMock(return_value=mock_pool_instance)
            mock_pool_instance.__exit__ = MagicMock(return_value=False)
            mock_pool_instance.map.return_value = [[{"role": "user", "content": []}]]
            MockPool.return_value = mock_pool_instance

            img = Image.new("RGB", (100, 100))
            engine.infer_batch([img])

            MockPool.assert_called_once_with(max_workers=4)

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_default_max_encode_workers_is_8(self):
        """Default config should use max_encode_workers=8."""
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_vllm.LLM.return_value = MagicMock()

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
        )
        assert config.inference.max_encode_workers == 8


class TestAsyncPipeline:
    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_infer_batch_still_returns_correct_results(self):
        """Pipeline refactor must not change the return values."""
        from PIL import Image
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_llm_instance = MagicMock()
        mock_vllm.LLM.return_value = mock_llm_instance

        mock_outputs = [
            MagicMock(outputs=[MagicMock(text="# Page 0")]),
            MagicMock(outputs=[MagicMock(text="# Page 1")]),
        ]
        mock_llm_instance.chat.return_value = mock_outputs

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            vllm_args={"trust-remote-code": True},
        )
        engine = VLLMOfflineEngine(config)

        images = [Image.new("RGB", (100, 100)) for _ in range(2)]
        results = engine.infer_batch(images)

        assert results == ["# Page 0", "# Page 1"]

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_infer_batch_encodes_as_jpeg_base64(self):
        """Images must still be encoded as JPEG base64 data URIs."""
        from PIL import Image
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_llm_instance = MagicMock()
        mock_vllm.LLM.return_value = mock_llm_instance

        mock_output = MagicMock(outputs=[MagicMock(text="result")])
        mock_llm_instance.chat.return_value = [mock_output]

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
        )
        engine = VLLMOfflineEngine(config)

        img = Image.new("RGB", (100, 100))
        engine.infer_batch([img])

        call_args = mock_llm_instance.chat.call_args
        messages_list = call_args[1].get("messages") or call_args[0][0]
        url = messages_list[0][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_multiple_batches_all_succeed(self):
        """Calling infer_batch multiple times must work correctly."""
        from PIL import Image
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_llm_instance = MagicMock()
        mock_vllm.LLM.return_value = mock_llm_instance

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
        )
        engine = VLLMOfflineEngine(config)

        for batch_idx in range(3):
            mock_llm_instance.chat.return_value = [
                MagicMock(outputs=[MagicMock(text=f"result_{batch_idx}")])
            ]
            images = [Image.new("RGB", (50, 50))]
            results = engine.infer_batch(images)
            assert results == [f"result_{batch_idx}"]

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_infer_batch_empty_still_works(self):
        """Empty batch must return empty list without touching GPU."""
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_llm_instance = MagicMock()
        mock_vllm.LLM.return_value = mock_llm_instance

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
        )
        engine = VLLMOfflineEngine(config)
        assert engine.infer_batch([]) == []
        mock_llm_instance.chat.assert_not_called()


class TestFactoryIntegration:
    @patch("pdf_ocr.engine_factory.detect_gpus")
    @patch("pdf_ocr.engine_factory.VLLMOfflineEngine")
    def test_init_module_uses_factory(self, MockEngine, mock_detect):
        """pdf_ocr.convert() with backend='offline' should use create_offline_engine."""
        mock_detect.return_value = [GPUInfo(index=0, name="L4", vram_mb=24576)]
        mock_instance = MagicMock()
        mock_instance.infer_batch.return_value = ["# result"]
        MockEngine.return_value = mock_instance

        # This tests that the import path works; the actual convert call
        # is too complex to test here without more mocking.
        from pdf_ocr.engine_factory import create_offline_engine
        engine = create_offline_engine(ModelConfig(
            model_id="test/model",
            served_model_name="test",
        ))
        assert engine is mock_instance
