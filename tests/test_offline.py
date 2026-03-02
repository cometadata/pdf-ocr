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
        assert kwargs["gpu_memory_utilization"] == 0.85
        assert kwargs["max_num_batched_tokens"] == 8192

    def test_no_auto_kwargs_same_as_before(self):
        config = self._make_config({"trust-remote-code": True})
        kwargs = build_engine_kwargs(config)
        assert kwargs == {"model": "test/model", "trust_remote_code": True}


class TestVLLMOfflineEngine:
    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_init_passes_correct_kwargs(self):
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_llm_instance = MagicMock()
        mock_sp_instance = MagicMock()
        mock_vllm.LLM.return_value = mock_llm_instance
        mock_vllm.SamplingParams.return_value = mock_sp_instance

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            vllm_args={
                "trust-remote-code": True,
                "max-model-len": 4096,
            },
        )
        engine = VLLMOfflineEngine(config)

        mock_vllm.LLM.assert_called_once_with(
            model="test/model",
            trust_remote_code=True,
            max_model_len=4096,
        )
        mock_vllm.SamplingParams.assert_called_once_with(
            max_tokens=4000,
            temperature=0.2,
            top_p=0.9,
        )

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_infer_batch_passes_pil_images_directly(self):
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
        image_value = messages_list[0][0]["content"][0]["image_pil"]
        assert isinstance(image_value, Image.Image)

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
    def test_no_base64_encoding(self):
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
        image_value = messages_list[0][0]["content"][0]["image_pil"]
        assert not isinstance(image_value, str)
        assert isinstance(image_value, Image.Image)

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_multiple_batches_all_succeed(self):
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
    def test_no_start_next_prep_method(self):
        import sys

        mock_vllm = sys.modules["vllm"]
        mock_vllm.LLM.return_value = MagicMock()

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
        )
        engine = VLLMOfflineEngine(config)
        assert not hasattr(engine, "start_next_prep")


class TestDataParallelInit:
    @patch.dict("sys.modules", {"vllm": MagicMock()})
    @patch("pdf_ocr.gpu.get_physical_gpu_ids", return_value=["0"])
    def test_single_gpu_uses_single_path(self, mock_gpu_ids):
        import sys
        mock_vllm = sys.modules["vllm"]
        mock_vllm.LLM.return_value = MagicMock()

        config = ModelConfig(model_id="test/model", served_model_name="test-model")
        engine = VLLMOfflineEngine(config)

        assert engine._llm is not None
        assert engine._workers == []
        assert engine._input_queues == []

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    @patch("pdf_ocr.gpu.get_physical_gpu_ids", return_value=["0", "1", "2", "3"])
    @patch("pdf_ocr.offline.mp")
    def test_multi_gpu_spawns_workers(self, mock_mp, mock_gpu_ids):
        mock_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_ctx
        mock_process = MagicMock()
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        config = ModelConfig(model_id="test/model", served_model_name="test-model")
        engine = VLLMOfflineEngine(config)

        assert engine._llm is None
        assert len(engine._workers) == 4
        assert len(engine._input_queues) == 4
        assert mock_process.start.call_count == 4

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    @patch("pdf_ocr.gpu.get_physical_gpu_ids", return_value=[])
    def test_no_gpus_uses_single_path(self, mock_gpu_ids):
        import sys
        mock_vllm = sys.modules["vllm"]
        mock_vllm.LLM.return_value = MagicMock()

        config = ModelConfig(model_id="test/model", served_model_name="test-model")
        engine = VLLMOfflineEngine(config)

        assert engine._llm is not None
        assert engine._workers == []


class TestDataParallelInference:
    def _make_dp_engine(self, n_workers=2):
        """Create a VLLMOfflineEngine with mocked DP workers."""
        engine = object.__new__(VLLMOfflineEngine)
        engine.config = ModelConfig(model_id="test/model", served_model_name="test-model")
        engine._llm = None
        engine._sampling_params = None

        engine._workers = [MagicMock(is_alive=MagicMock(return_value=True)) for _ in range(n_workers)]
        engine._input_queues = [MagicMock() for _ in range(n_workers)]
        engine._output_queue = MagicMock()

        return engine

    def test_interleaved_split_and_reassembly(self):
        from PIL import Image

        engine = self._make_dp_engine(n_workers=2)

        images = [Image.new("RGB", (10, 10)) for _ in range(5)]
        # Worker 0 gets indices [0, 2, 4], Worker 1 gets [1, 3]

        def mock_get(timeout=None):
            if not hasattr(mock_get, 'call_count'):
                mock_get.call_count = 0
            mock_get.call_count += 1
            if mock_get.call_count == 1:
                return (0, ["r0", "r2", "r4"], None)
            else:
                return (1, ["r1", "r3"], None)

        engine._output_queue.get = mock_get

        results = engine.infer_batch(images)
        assert results == ["r0", "r1", "r2", "r3", "r4"]

    def test_error_propagation_from_worker(self):
        from PIL import Image

        engine = self._make_dp_engine(n_workers=2)
        images = [Image.new("RGB", (10, 10)) for _ in range(4)]

        engine._output_queue.get.return_value = (0, None, ValueError("CUDA OOM"))

        with pytest.raises(ValueError, match="CUDA OOM"):
            engine.infer_batch(images)

    def test_fewer_images_than_workers(self):
        from PIL import Image

        engine = self._make_dp_engine(n_workers=4)
        images = [Image.new("RGB", (10, 10)) for _ in range(2)]
        # Only workers 0 and 1 get work

        def mock_get(timeout=None):
            if not hasattr(mock_get, 'call_count'):
                mock_get.call_count = 0
            mock_get.call_count += 1
            if mock_get.call_count == 1:
                return (0, ["r0"], None)
            else:
                return (1, ["r1"], None)

        engine._output_queue.get = mock_get

        results = engine.infer_batch(images)
        assert results == ["r0", "r1"]

        # Workers 2 and 3 should not have received any work
        engine._input_queues[2].put.assert_not_called()
        engine._input_queues[3].put.assert_not_called()

    def test_empty_batch_returns_empty(self):
        engine = self._make_dp_engine(n_workers=2)
        results = engine.infer_batch([])
        assert results == []


class TestShutdown:
    def test_shutdown_sends_poison_pills_and_joins(self):
        engine = object.__new__(VLLMOfflineEngine)
        engine._workers = [MagicMock() for _ in range(3)]
        engine._input_queues = [MagicMock() for _ in range(3)]
        engine._output_queue = MagicMock()

        for w in engine._workers:
            w.is_alive.return_value = False

        engine.shutdown()

        for q in engine._input_queues:
            q.put.assert_called_once_with(None)
        for w in engine._workers:
            w.join.assert_called_once_with(timeout=10)

    def test_shutdown_terminates_stuck_workers(self):
        engine = object.__new__(VLLMOfflineEngine)
        stuck_worker = MagicMock()
        stuck_worker.is_alive.return_value = True

        engine._workers = [stuck_worker]
        engine._input_queues = [MagicMock()]
        engine._output_queue = MagicMock()

        engine.shutdown()

        stuck_worker.terminate.assert_called_once()

    def test_shutdown_no_workers_is_noop(self):
        engine = object.__new__(VLLMOfflineEngine)
        engine._workers = []
        engine._input_queues = []
        engine._output_queue = None

        engine.shutdown()  # Should not raise

    def test_shutdown_clears_worker_lists(self):
        engine = object.__new__(VLLMOfflineEngine)
        engine._workers = [MagicMock()]
        engine._input_queues = [MagicMock()]
        engine._output_queue = MagicMock()

        engine._workers[0].is_alive.return_value = False

        engine.shutdown()

        assert engine._workers == []
        assert engine._input_queues == []


class TestWorkersHealthy:
    def test_single_gpu_always_healthy(self):
        engine = object.__new__(VLLMOfflineEngine)
        engine._workers = []
        assert engine.workers_healthy() is True

    def test_all_alive_returns_true(self):
        engine = object.__new__(VLLMOfflineEngine)
        engine._workers = [
            MagicMock(is_alive=MagicMock(return_value=True)),
            MagicMock(is_alive=MagicMock(return_value=True)),
        ]
        assert engine.workers_healthy() is True

    def test_one_dead_returns_false(self):
        engine = object.__new__(VLLMOfflineEngine)
        engine._workers = [
            MagicMock(is_alive=MagicMock(return_value=True)),
            MagicMock(is_alive=MagicMock(return_value=False)),
        ]
        assert engine.workers_healthy() is False

    def test_all_dead_returns_false(self):
        engine = object.__new__(VLLMOfflineEngine)
        engine._workers = [
            MagicMock(is_alive=MagicMock(return_value=False)),
            MagicMock(is_alive=MagicMock(return_value=False)),
        ]
        assert engine.workers_healthy() is False


class TestRestartWorker:
    def test_restart_replaces_dead_worker(self):
        engine = object.__new__(VLLMOfflineEngine)
        engine.config = ModelConfig(model_id="test/model", served_model_name="test-model")

        old_worker = MagicMock()
        old_worker.is_alive.return_value = False

        mock_ctx = MagicMock()
        new_process = MagicMock()
        new_process.pid = 12345
        mock_ctx.Process.return_value = new_process
        new_queue = MagicMock()
        mock_ctx.Queue.return_value = new_queue

        engine._workers = [old_worker]
        engine._input_queues = [MagicMock()]
        engine._output_queue = MagicMock()
        engine._ctx = mock_ctx
        engine._gpu_ids = ["0"]
        engine._engine_kwargs = {"model": "test/model"}

        engine._restart_worker(0)

        assert engine._workers[0] is new_process
        assert engine._input_queues[0] is new_queue
        new_process.start.assert_called_once()

    def test_restart_terminates_alive_worker(self):
        engine = object.__new__(VLLMOfflineEngine)
        engine.config = ModelConfig(model_id="test/model", served_model_name="test-model")

        old_worker = MagicMock()
        # is_alive returns True on first call (triggers terminate), then False
        old_worker.is_alive.side_effect = [True, False]

        mock_ctx = MagicMock()
        new_process = MagicMock()
        new_process.pid = 99
        mock_ctx.Process.return_value = new_process
        mock_ctx.Queue.return_value = MagicMock()

        engine._workers = [old_worker]
        engine._input_queues = [MagicMock()]
        engine._output_queue = MagicMock()
        engine._ctx = mock_ctx
        engine._gpu_ids = ["0"]
        engine._engine_kwargs = {"model": "test/model"}

        engine._restart_worker(0)

        old_worker.terminate.assert_called_once()
        assert engine._workers[0] is new_process


class TestDataParallelDeadWorkerRestart:
    def test_dead_worker_restarted_before_dispatch(self):
        from PIL import Image

        engine = object.__new__(VLLMOfflineEngine)
        engine.config = ModelConfig(model_id="test/model", served_model_name="test-model")
        engine._engine_kwargs = {"model": "test/model"}
        engine._gpu_ids = ["0", "1"]

        mock_ctx = MagicMock()
        new_process = MagicMock()
        new_process.pid = 999
        new_process.is_alive.return_value = True
        mock_ctx.Process.return_value = new_process
        new_queue = MagicMock()
        mock_ctx.Queue.return_value = new_queue
        engine._ctx = mock_ctx

        alive_worker = MagicMock()
        alive_worker.is_alive.return_value = True
        dead_worker = MagicMock()
        dead_worker.is_alive.return_value = False

        engine._workers = [alive_worker, dead_worker]
        engine._input_queues = [MagicMock(), MagicMock()]
        engine._output_queue = MagicMock()

        # After restart, the dead worker is replaced; both return results
        call_count = [0]
        def mock_get(timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return (0, ["r0"], None)
            else:
                return (1, ["r1"], None)

        engine._output_queue.get = mock_get

        images = [Image.new("RGB", (10, 10)) for _ in range(2)]
        results = engine._infer_data_parallel(images)

        # Worker 1 was replaced
        assert engine._workers[1] is new_process
        assert results == ["r0", "r1"]


class TestMicroBatchInit:
    @patch.dict("sys.modules", {"vllm": MagicMock()})
    @patch("pdf_ocr.gpu.get_physical_gpu_ids", return_value=["0", "1"])
    @patch("pdf_ocr.offline.mp")
    def test_worker_batch_size_passed_to_worker(self, mock_mp, mock_gpu_ids):
        mock_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_ctx
        mock_process = MagicMock()
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        config = ModelConfig(
            model_id="test/model",
            served_model_name="test-model",
            inference=InferenceConfig(worker_batch_size=16),
        )
        engine = VLLMOfflineEngine(config)

        # Check that worker_batch_size=16 was passed as the last positional arg
        for call in mock_ctx.Process.call_args_list:
            args = call[1]["args"]
            assert args[-1] == 16  # worker_batch_size is the last arg


class TestFactoryIntegration:
    @patch("pdf_ocr.engine_factory.VLLMOfflineEngine")
    def test_init_module_uses_factory(self, MockEngine):
        mock_instance = MagicMock()
        mock_instance.infer_batch.return_value = ["# result"]
        MockEngine.return_value = mock_instance

        from pdf_ocr.engine_factory import create_offline_engine
        engine = create_offline_engine(ModelConfig(
            model_id="test/model",
            served_model_name="test",
        ))
        assert engine is mock_instance
