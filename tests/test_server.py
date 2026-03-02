import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pdf_ocr.server import build_vllm_command, launch_vllm, VLLMClient
from pdf_ocr.config import ModelConfig, InferenceConfig, PdfRenderingConfig


def test_build_vllm_command_basic():
    config = ModelConfig(
        model_id="lightonai/LightOnOCR-2-1B",
        served_model_name="lighton-ocr",
        vllm_args={
            "trust-remote-code": True,
            "max-model-len": 4096,
            "gpu-memory-utilization": 0.90,
        },
    )
    cmd = build_vllm_command(config, port=8000, host="0.0.0.0")

    assert cmd[0] == "vllm"
    assert cmd[1] == "serve"
    assert "lightonai/LightOnOCR-2-1B" in cmd
    assert "--served-model-name" in cmd
    assert "lighton-ocr" in cmd
    assert "--trust-remote-code" in cmd
    assert "--max-model-len" in cmd
    assert "4096" in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.90" in cmd
    assert "--port" in cmd
    assert "8000" in cmd


def test_build_vllm_command_boolean_flags():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
        vllm_args={"no-enable-prefix-caching": True, "trust-remote-code": True},
    )
    cmd = build_vllm_command(config, port=8000, host="0.0.0.0")

    idx = cmd.index("--no-enable-prefix-caching")
    if idx + 1 < len(cmd):
        assert cmd[idx + 1].startswith("--") or cmd[idx + 1] == "8000"


def test_build_vllm_command_string_values():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
        vllm_args={"limit-mm-per-prompt": '{"image": 1}'},
    )
    cmd = build_vllm_command(config, port=8000, host="0.0.0.0")

    idx = cmd.index("--limit-mm-per-prompt")
    assert cmd[idx + 1] == '{"image": 1}'


def test_build_vllm_command_chunked_prefill():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
        vllm_args={"enable-chunked-prefill": True, "trust-remote-code": True},
    )
    cmd = build_vllm_command(config, port=8000, host="0.0.0.0")
    assert "--enable-chunked-prefill" in cmd
    idx = cmd.index("--enable-chunked-prefill")
    if idx + 1 < len(cmd):
        assert cmd[idx + 1].startswith("--") or cmd[idx + 1] in ("8000", "0.0.0.0")


def test_vllm_client_stores_retry_config():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
        inference=InferenceConfig(max_retries=5, retry_backoff=3.0),
    )
    client = VLLMClient(base_url="http://localhost:8000", config=config)
    assert client.max_retries == 5
    assert client.retry_backoff == 3.0


def test_async_completion_retries_on_failure():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
        inference=InferenceConfig(max_retries=3, retry_backoff=0.01),
    )
    client = VLLMClient(base_url="http://localhost:8000", config=config)

    mock_message = MagicMock()
    mock_message.content = "# OCR Result"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    call_count = 0

    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Server unavailable")
        return mock_response

    client._client = MagicMock()
    client._client.chat.completions.create = mock_create

    payload = {
        "model": "test",
        "messages": [],
        "max_tokens": 100,
        "temperature": 0.2,
    }

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(client._async_completion(payload))
    finally:
        loop.close()

    assert result == "# OCR Result"
    assert call_count == 3


def test_async_completion_raises_after_max_retries():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
        inference=InferenceConfig(max_retries=2, retry_backoff=0.01),
    )
    client = VLLMClient(base_url="http://localhost:8000", config=config)

    async def mock_create(**kwargs):
        raise ConnectionError("Server unavailable")

    client._client = MagicMock()
    client._client.chat.completions.create = mock_create

    payload = {
        "model": "test",
        "messages": [],
        "max_tokens": 100,
        "temperature": 0.2,
    }

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ConnectionError, match="Server unavailable"):
            loop.run_until_complete(client._async_completion(payload))
    finally:
        loop.close()


def test_prepare_payload_includes_extra_body():
    from PIL import Image

    extra = {"skip_special_tokens": False, "vllm_xargs": {"ngram_size": 30}}
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
        inference=InferenceConfig(extra_body=extra),
    )
    client = VLLMClient(base_url="http://localhost:8000", config=config)

    img = Image.new("RGB", (10, 10))
    payload = client._prepare_payload(img)

    assert "extra_body" in payload
    assert payload["extra_body"]["skip_special_tokens"] is False
    assert payload["extra_body"]["vllm_xargs"]["ngram_size"] == 30
    img.close()


def test_prepare_payload_omits_extra_body_when_empty():
    from PIL import Image

    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
    )
    client = VLLMClient(base_url="http://localhost:8000", config=config)

    img = Image.new("RGB", (10, 10))
    payload = client._prepare_payload(img)

    assert "extra_body" not in payload
    img.close()


def test_async_completion_passes_extra_body():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
        inference=InferenceConfig(max_retries=1, extra_body={"skip_special_tokens": False}),
    )
    client = VLLMClient(base_url="http://localhost:8000", config=config)

    mock_message = MagicMock()
    mock_message.content = "result"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    captured_kwargs = {}

    async def mock_create(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_response

    client._client = MagicMock()
    client._client.chat.completions.create = mock_create

    payload = {
        "model": "test",
        "messages": [],
        "max_tokens": 100,
        "temperature": 0.2,
        "extra_body": {"skip_special_tokens": False},
    }

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(client._async_completion(payload))
    finally:
        loop.close()

    assert "extra_body" in captured_kwargs
    assert captured_kwargs["extra_body"]["skip_special_tokens"] is False


def test_build_vllm_command_data_parallel_size():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
        vllm_args={"trust-remote-code": True},
    )
    cmd = build_vllm_command(config, data_parallel_size=4)
    assert "--data-parallel-size" in cmd
    idx = cmd.index("--data-parallel-size")
    assert cmd[idx + 1] == "4"


def test_build_vllm_command_data_parallel_size_one_omitted():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
    )
    cmd = build_vllm_command(config, data_parallel_size=1)
    assert "--data-parallel-size" not in cmd


def test_build_vllm_command_data_parallel_size_none_omitted():
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
    )
    cmd = build_vllm_command(config, data_parallel_size=None)
    assert "--data-parallel-size" not in cmd


@patch("pdf_ocr.server.subprocess.Popen")
def test_launch_vllm_passes_data_parallel_size(mock_popen):
    mock_popen.return_value = MagicMock()
    mock_popen.return_value.stdout = None
    mock_popen.return_value.stderr = None
    config = ModelConfig(
        model_id="test/model",
        served_model_name="test",
    )
    launch_vllm(config, data_parallel_size=4)
    cmd = mock_popen.call_args[0][0]
    assert "--data-parallel-size" in cmd
    idx = cmd.index("--data-parallel-size")
    assert cmd[idx + 1] == "4"
