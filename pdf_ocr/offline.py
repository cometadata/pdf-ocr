from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from .config import ModelConfig

if TYPE_CHECKING:
    from PIL import Image

LOGGER = logging.getLogger(__name__)

_SERVER_ONLY_ARGS = frozenset({
    "served-model-name",
    "host",
    "port",
    "api-key",
    "ssl-keyfile",
    "ssl-certfile",
    "root-path",
    "response-role",
    "chat-template",
})

_NEGATION_FLAGS = frozenset({
    "no-enable-prefix-caching",
})


def _cli_flag_to_kwarg(flag: str) -> str:
    return flag.replace("-", "_")


def _coerce_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
    return value


def build_engine_kwargs(config: ModelConfig, auto_kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": config.model_id,
    }

    # Apply auto-detected defaults first (lowest priority)
    if auto_kwargs:
        kwargs.update(auto_kwargs)

    for flag, value in config.vllm_args.items():
        if flag in _SERVER_ONLY_ARGS:
            continue

        if flag in _NEGATION_FLAGS:
            real_flag = flag[3:]
            kwarg_name = _cli_flag_to_kwarg(real_flag)
            kwargs[kwarg_name] = not bool(value)
            continue

        kwarg_name = _cli_flag_to_kwarg(flag)
        kwargs[kwarg_name] = _coerce_value(value)

    return kwargs


class VLLMOfflineEngine:

    def __init__(self, config: ModelConfig) -> None:
        from vllm import LLM, SamplingParams

        self.config = config

        if config.inference.extra_body:
            LOGGER.warning(
                "extra_body is set but is only applicable in server mode; "
                "it will be ignored for offline inference"
            )

        from .gpu import detect_gpus, recommend_engine_kwargs

        gpus = detect_gpus()
        auto_kwargs = recommend_engine_kwargs(gpus)
        if auto_kwargs:
            LOGGER.info("Auto-detected GPU config: %s", auto_kwargs)

        engine_kwargs = build_engine_kwargs(config, auto_kwargs=auto_kwargs)
        LOGGER.info("Initializing offline vLLM engine: %s", engine_kwargs)
        t0 = time.monotonic()
        self._llm = LLM(**engine_kwargs)
        LOGGER.info("vLLM engine initialized in %.2fs", time.monotonic() - t0)
        self._sampling_params = SamplingParams(
            max_tokens=config.inference.max_tokens,
            temperature=config.inference.temperature,
            top_p=config.inference.top_p,
        )

    def infer_batch(self, images: Sequence["Image.Image"]) -> List[str]:
        if not images:
            return []

        t0 = time.monotonic()

        messages_list = [
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image}},
            ]}]
            for image in images
        ]

        outputs = self._llm.chat(
            messages=messages_list,
            sampling_params=self._sampling_params,
        )
        t_infer = time.monotonic()

        results = [o.outputs[0].text if o.outputs else "" for o in outputs]

        LOGGER.info(
            "Batch inference: %d images | %.2fs | %.2f pages/s",
            len(images),
            t_infer - t0,
            len(images) / (t_infer - t0),
        )
        return results
