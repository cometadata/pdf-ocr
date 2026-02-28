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
