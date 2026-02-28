import pytest
from pdf_ocr.server import build_vllm_command
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
