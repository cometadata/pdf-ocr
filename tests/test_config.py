import pytest
from pathlib import Path

from pdf_ocr.config import ModelConfig, load_config


def test_load_bundled_config_by_name():
    config = load_config("lighton_ocr_2_1b")
    assert config.model_id == "lightonai/LightOnOCR-2-1B"
    assert config.served_model_name == "lighton-ocr"
    assert config.inference.max_tokens == 4000
    assert config.inference.temperature == 0.2
    assert config.pdf_rendering.dpi == 200
    assert config.pdf_rendering.max_dimension == 1540


def test_load_bundled_config_by_model_id():
    config = load_config("lightonai/LightOnOCR-2-1B")
    assert config.model_id == "lightonai/LightOnOCR-2-1B"


def test_load_config_from_yaml_path(tmp_path):
    yaml_content = """
model_id: custom/MyModel
served_model_name: my-model
vllm_args:
  trust-remote-code: true
inference:
  max_tokens: 2048
  temperature: 0.5
  batch_size: 8
pdf_rendering:
  dpi: 150
  max_dimension: 1024
"""
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml_content)

    config = load_config(str(config_path))
    assert config.model_id == "custom/MyModel"
    assert config.inference.max_tokens == 2048
    assert config.pdf_rendering.dpi == 150


def test_load_config_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        load_config("nonexistent_model")


def test_config_with_overrides():
    config = load_config("lighton_ocr_2_1b")
    config = config.with_overrides(batch_size=16, max_tokens=8192)
    assert config.inference.batch_size == 16
    assert config.inference.max_tokens == 8192
    assert config.inference.temperature == 0.2


def test_extra_body_default_empty():
    config = load_config("lighton_ocr_2_1b")
    assert config.inference.extra_body == {}


def test_extra_body_from_yaml(tmp_path):
    yaml_content = """
model_id: custom/MyModel
served_model_name: my-model
inference:
  max_tokens: 2048
  extra_body:
    skip_special_tokens: false
    vllm_xargs:
      ngram_size: 30
"""
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml_content)

    config = load_config(str(config_path))
    assert config.inference.extra_body == {
        "skip_special_tokens": False,
        "vllm_xargs": {"ngram_size": 30},
    }


def test_chunked_prefill_in_bundled_config():
    config = load_config("lighton_ocr_2_1b")
    assert config.vllm_args.get("enable-chunked-prefill") is True


def test_backend_default_server():
    config = load_config("lighton_ocr_2_1b")
    assert config.backend == "server"


def test_backend_from_yaml(tmp_path):
    yaml_content = """
model_id: custom/MyModel
served_model_name: my-model
backend: offline
"""
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml_content)

    config = load_config(str(config_path))
    assert config.backend == "offline"


def test_backend_with_overrides():
    config = load_config("lighton_ocr_2_1b")
    config = config.with_overrides(backend="offline")
    assert config.backend == "offline"


def test_new_inference_fields_defaults():
    config = load_config("lighton_ocr_2_1b")
    assert config.inference.offline_batch_size == 128
    assert config.inference.render_workers == 4
    assert config.inference.max_retry_depth == 3
    assert config.inference.flush_every == 10


def test_new_inference_fields_with_overrides():
    config = load_config("lighton_ocr_2_1b")
    config = config.with_overrides(offline_batch_size=64, render_workers=8)
    assert config.inference.offline_batch_size == 64
    assert config.inference.render_workers == 8


