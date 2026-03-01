"""vLLM server management and async inference client."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import signal
import subprocess
import threading
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any, Awaitable, Dict, List, Sequence

import requests
from openai import AsyncOpenAI

from .config import ModelConfig

if TYPE_CHECKING:
    from PIL import Image

LOGGER = logging.getLogger(__name__)


def encode_image(image: "Image.Image") -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _format_arg(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_vllm_command(config: ModelConfig, port: int = 8000, host: str = "0.0.0.0") -> List[str]:
    cmd = [
        "vllm", "serve", config.model_id,
        "--served-model-name", config.served_model_name,
        "--port", str(port),
        "--host", host,
    ]
    for flag, value in config.vllm_args.items():
        cmd.append(f"--{flag}")
        if value is not True:
            cmd.append(_format_arg(value))
    return cmd


def _stream_output(pipe, prefix: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            print(f"[{prefix}] {line.rstrip()}", flush=True)
    finally:
        pipe.close()


def launch_vllm(config: ModelConfig, port: int = 8000, host: str = "0.0.0.0") -> subprocess.Popen:
    cmd = build_vllm_command(config, port=port, host=host)
    LOGGER.info("Launching vLLM server: %s", " ".join(cmd))

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )

    threads = []
    for name, pipe in [("STDOUT", process.stdout), ("STDERR", process.stderr)]:
        if pipe:
            t = threading.Thread(
                target=_stream_output, args=(pipe, f"vLLM {name}"), daemon=True
            )
            t.start()
            threads.append(t)

    process._log_threads = threads  # type: ignore
    return process


def shutdown_server(server_process: subprocess.Popen) -> None:
    LOGGER.info("Shutting down vLLM server")
    server_process.send_signal(signal.SIGTERM)
    try:
        server_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        LOGGER.warning("Server did not exit in time, sending SIGKILL")
        server_process.kill()
    for thread in getattr(server_process, "_log_threads", []):
        thread.join(timeout=1)


def wait_for_server(url: str, timeout_s: int = 600, interval_s: int = 5) -> bool:
    start = time.time()
    LOGGER.info("Waiting for vLLM server at %s ...", url)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).ok:
                elapsed = time.time() - start
                LOGGER.info("vLLM server ready in %.0fs", elapsed)
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    LOGGER.error("vLLM server failed to start after %ds", timeout_s)
    return False


class VLLMClient:
    """Async batch inference client for VLM models via vLLM."""

    def __init__(self, base_url: str, config: ModelConfig) -> None:
        self.base_url = base_url.rstrip("/")
        self.config = config
        self.max_retries = max(0, config.inference.max_retries)
        self.retry_backoff = max(0.0, config.inference.retry_backoff)
        self._client = AsyncOpenAI(api_key="vllm", base_url=f"{self.base_url}/v1")

    def _prepare_payload(self, image: "Image.Image") -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_image(image)}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.config.inference.max_tokens,
            "temperature": self.config.inference.temperature,
            "top_p": self.config.inference.top_p,
        }
        if self.config.inference.extra_body:
            payload["extra_body"] = self.config.inference.extra_body
        return payload

    async def _async_completion(self, payload: Dict[str, Any]) -> str:
        timeout = self.config.inference.request_timeout
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=payload["model"],
                    messages=payload["messages"],
                    max_tokens=payload["max_tokens"],
                    temperature=payload["temperature"],
                    top_p=payload.get("top_p"),
                    timeout=timeout,
                )
                extra_body = payload.get("extra_body")
                if extra_body:
                    kwargs["extra_body"] = extra_body

                response = await self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = min(self.retry_backoff * 2 ** (attempt - 1), 60.0)
                    LOGGER.warning(
                        "vLLM request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, self.max_retries, exc, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                LOGGER.error(
                    "vLLM request failed after %d attempts: %s", self.max_retries, exc
                )
                raise
            if not response.choices:
                return ""
            return getattr(response.choices[0].message, "content", "") or ""

        assert last_exc is not None
        raise last_exc

    def infer_batch(self, images: Sequence["Image.Image"]) -> List[str]:
        if not images:
            return []
        payloads = [self._prepare_payload(img) for img in images]
        return self._run_async(self._async_infer_batch(payloads))

    async def _async_infer_batch(self, payloads: Sequence[Dict[str, Any]]) -> List[str]:
        tasks = [asyncio.create_task(self._async_completion(p)) for p in payloads]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _run_async(coro: Awaitable[Any]) -> Any:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.run_until_complete(loop.shutdown_asyncgens())
            return result
        finally:
            asyncio.set_event_loop(None)
            loop.close()
