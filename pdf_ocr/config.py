from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict

import yaml

LOGGER = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"


@dataclass(frozen=True)
class InferenceConfig:
    max_tokens: int = 4000
    temperature: float = 0.2
    top_p: float = 0.9
    batch_size: int = 4
    max_concurrency: int = 4
    request_timeout: int = 120
    max_retries: int = 3
    retry_backoff: float = 2.0
    extra_body: Dict[str, Any] = field(default_factory=dict)
    offline_batch_size: int = 128
    render_workers: int = 4
    max_retry_depth: int = 3
    flush_every: int = 10


@dataclass(frozen=True)
class PdfRenderingConfig:
    dpi: int = 200
    max_dimension: int = 1540


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    served_model_name: str
    backend: str = "server"
    vllm_args: Dict[str, Any] = field(default_factory=dict)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    pdf_rendering: PdfRenderingConfig = field(default_factory=PdfRenderingConfig)

    def with_overrides(self, **kwargs) -> ModelConfig:
        inference_fields = {f.name for f in InferenceConfig.__dataclass_fields__.values()}
        pdf_fields = {f.name for f in PdfRenderingConfig.__dataclass_fields__.values()}

        inference_overrides = {}
        pdf_overrides = {}
        top_overrides = {}

        for k, v in kwargs.items():
            if v is None:
                continue
            if k in inference_fields:
                inference_overrides[k] = v
            elif k in pdf_fields:
                pdf_overrides[k] = v
            else:
                top_overrides[k] = v

        new_inference = replace(self.inference, **inference_overrides) if inference_overrides else self.inference
        new_pdf = replace(self.pdf_rendering, **pdf_overrides) if pdf_overrides else self.pdf_rendering

        return replace(self, inference=new_inference, pdf_rendering=new_pdf, **top_overrides)


def _parse_yaml(raw: Dict[str, Any]) -> ModelConfig:
    inference_data = raw.get("inference", {})
    pdf_data = raw.get("pdf_rendering", {})
    return ModelConfig(
        model_id=raw["model_id"],
        served_model_name=raw.get("served_model_name", raw["model_id"].split("/")[-1]),
        backend=raw.get("backend", "server"),
        vllm_args=raw.get("vllm_args", {}),
        inference=InferenceConfig(**{k: v for k, v in inference_data.items() if k in InferenceConfig.__dataclass_fields__}),
        pdf_rendering=PdfRenderingConfig(**{k: v for k, v in pdf_data.items() if k in PdfRenderingConfig.__dataclass_fields__}),
    )


def _list_bundled_configs() -> Dict[str, Path]:
    configs = {}
    if MODELS_DIR.is_dir():
        for p in MODELS_DIR.glob("*.yaml"):
            configs[p.stem] = p
    return configs


def load_config(model: str) -> ModelConfig:
    path = Path(model)
    if path.suffix in {".yaml", ".yml"} and path.is_file():
        with open(path) as f:
            return _parse_yaml(yaml.safe_load(f))

    bundled = _list_bundled_configs()
    if model in bundled:
        with open(bundled[model]) as f:
            return _parse_yaml(yaml.safe_load(f))

    for name, config_path in bundled.items():
        with open(config_path) as f:
            data = yaml.safe_load(f)
            if data.get("model_id") == model:
                return _parse_yaml(data)

    available = list(bundled.keys())
    raise ValueError(
        f"Unknown model: {model!r}. "
        f"Available bundled configs: {available}. "
        f"Or pass a path to a .yaml file."
    )
