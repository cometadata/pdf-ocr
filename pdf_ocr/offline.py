from __future__ import annotations

import base64
import json
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from .config import ModelConfig

if TYPE_CHECKING:
    from PIL import Image

LOGGER = logging.getLogger(__name__)

# vllm_args keys that only apply to the server CLI; skipped for offline use
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

# "no-" prefixed flags negate the underlying kwarg
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


def build_engine_kwargs(config: ModelConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": config.model_id,
    }

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


def encode_image(image: "Image.Image") -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class VLLMOfflineEngine:

    def __init__(self, config: ModelConfig) -> None:
        from vllm import LLM, SamplingParams

        self.config = config

        if config.inference.extra_body:
            LOGGER.warning(
                "extra_body is set but is only applicable in server mode; "
                "it will be ignored for offline inference"
            )

        engine_kwargs = build_engine_kwargs(config)
        LOGGER.info("Initializing offline vLLM engine: %s", engine_kwargs)
        self._llm = LLM(**engine_kwargs)
        self._sampling_params = SamplingParams(
            max_tokens=config.inference.max_tokens,
            temperature=config.inference.temperature,
            top_p=config.inference.top_p,
        )

    def infer_batch(self, images: Sequence["Image.Image"]) -> List[str]:
        if not images:
            return []

        messages_list = []
        for image in images:
            b64 = encode_image(image)
            messages_list.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                            },
                        },
                    ],
                }
            ])

        outputs = self._llm.chat(
            messages=messages_list,
            sampling_params=self._sampling_params,
        )

        results: List[str] = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            results.append(text)
        return results
