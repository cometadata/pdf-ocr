from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
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


def encode_image(image: "Image.Image") -> str:
    """Encode a PIL image to a base64 JPEG string for offline inference."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
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

        # Pipeline state: pre-encoded messages from a background thread
        self._prep_result: queue.Queue[list | None] = queue.Queue(maxsize=1)
        self._prep_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Image encoding helpers
    # ------------------------------------------------------------------

    def _encode_images(self, images: Sequence["Image.Image"]) -> list:
        """Encode a sequence of PIL images to vLLM chat messages (CPU-bound)."""
        def _encode_one(image: "Image.Image") -> list:
            b64 = encode_image(image)
            return [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}]

        with ThreadPoolExecutor(max_workers=self.config.inference.max_encode_workers) as pool:
            return list(pool.map(_encode_one, images))

    def _start_prep(self, images: Sequence["Image.Image"]) -> None:
        """Start encoding the next batch in a background thread."""
        def _worker():
            try:
                result = self._encode_images(images)
                self._prep_result.put(result)
            except Exception:
                LOGGER.exception("Background image encoding failed")
                self._prep_result.put(None)

        self._prep_thread = threading.Thread(target=_worker, daemon=True)
        self._prep_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_batch(self, images: Sequence["Image.Image"]) -> List[str]:
        if not images:
            return []

        t0 = time.monotonic()

        # Check if we have pre-encoded messages from a prior _start_prep call
        if self._prep_thread is not None:
            self._prep_thread.join()
            self._prep_thread = None
            pre_encoded = self._prep_result.get_nowait()
        else:
            pre_encoded = None

        if pre_encoded is not None and len(pre_encoded) == len(images):
            messages_list = pre_encoded
        else:
            # No valid pre-encoded batch (first call, or size mismatch) — encode now
            messages_list = self._encode_images(images)

        t_prep = time.monotonic()

        outputs = self._llm.chat(
            messages=messages_list,
            sampling_params=self._sampling_params,
        )
        t_infer = time.monotonic()

        results = [o.outputs[0].text if o.outputs else "" for o in outputs]

        LOGGER.info(
            "Batch inference: %d images | prep=%.2fs | vllm_chat=%.2fs | total=%.2fs | %.2f pages/s",
            len(images),
            t_prep - t0,
            t_infer - t_prep,
            t_infer - t0,
            len(images) / (t_infer - t0),
        )
        return results

    def start_next_prep(self, images: Sequence["Image.Image"]) -> None:
        """Pre-encode the next batch while current inference results are being processed.

        Called by the orchestration layer (convert_pages_streaming) after
        infer_batch returns but before the next batch's images are needed.
        """
        self._start_prep(images)
