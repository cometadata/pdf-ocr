from __future__ import annotations

import atexit
import json
import logging
import multiprocessing as mp
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

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


def _dp_worker(
    gpu_id: str,
    engine_kwargs: Dict[str, Any],
    max_tokens: int,
    temperature: float,
    top_p: float,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    worker_batch_size: int = 8,
) -> None:
    """Worker process for data-parallel inference on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    from vllm import LLM, SamplingParams

    llm = LLM(**engine_kwargs)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    while True:
        item = input_queue.get()
        if item is None:
            break

        task_id, images = item
        try:
            all_results: List[str] = []
            for start in range(0, len(images), worker_batch_size):
                micro_batch = images[start:start + worker_batch_size]
                messages_list = [
                    [{"role": "user", "content": [
                        {"type": "image_pil", "image_pil": image},
                    ]}]
                    for image in micro_batch
                ]
                outputs = llm.chat(
                    messages=messages_list,
                    sampling_params=sampling_params,
                )
                all_results.extend(
                    o.outputs[0].text if o.outputs else "" for o in outputs
                )
            output_queue.put((task_id, all_results, None))
        except Exception as exc:
            output_queue.put((task_id, None, exc))


class VLLMOfflineEngine:

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._workers: List[mp.Process] = []
        self._input_queues: List[mp.Queue] = []
        self._output_queue: mp.Queue | None = None

        if config.inference.extra_body:
            LOGGER.warning(
                "extra_body is set but is only applicable in server mode; "
                "it will be ignored for offline inference"
            )

        from .gpu import detect_gpus, get_physical_gpu_ids, recommend_engine_kwargs

        gpus = detect_gpus()
        auto_kwargs = recommend_engine_kwargs(gpus)
        if auto_kwargs:
            LOGGER.info("Auto-detected GPU config: %s", auto_kwargs)

        engine_kwargs = build_engine_kwargs(config, auto_kwargs=auto_kwargs)
        gpu_ids = get_physical_gpu_ids()

        if len(gpu_ids) > 1:
            self._init_data_parallel(engine_kwargs, gpu_ids)
        else:
            self._init_single(engine_kwargs)

    def _init_single(self, engine_kwargs: Dict[str, Any]) -> None:
        from vllm import LLM, SamplingParams

        LOGGER.info("Initializing offline vLLM engine (single GPU): %s", engine_kwargs)
        t0 = time.monotonic()
        self._llm = LLM(**engine_kwargs)
        LOGGER.info("vLLM engine initialized in %.2fs", time.monotonic() - t0)
        self._sampling_params = SamplingParams(
            max_tokens=self.config.inference.max_tokens,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
        )

    def _init_data_parallel(self, engine_kwargs: Dict[str, Any], gpu_ids: List[str]) -> None:
        LOGGER.info(
            "Initializing data-parallel vLLM across %d GPUs: %s | kwargs: %s",
            len(gpu_ids), gpu_ids, engine_kwargs,
        )
        self._llm = None
        self._sampling_params = None
        self._engine_kwargs = engine_kwargs
        self._gpu_ids = gpu_ids

        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._output_queue = ctx.Queue()

        worker_batch_size = self.config.inference.worker_batch_size

        t0 = time.monotonic()
        for gpu_id in gpu_ids:
            input_q = ctx.Queue()
            self._input_queues.append(input_q)
            p = ctx.Process(
                target=_dp_worker,
                args=(
                    gpu_id,
                    engine_kwargs,
                    self.config.inference.max_tokens,
                    self.config.inference.temperature,
                    self.config.inference.top_p,
                    input_q,
                    self._output_queue,
                    worker_batch_size,
                ),
            )
            p.start()
            self._workers.append(p)

        LOGGER.info(
            "Spawned %d data-parallel workers in %.2fs",
            len(self._workers), time.monotonic() - t0,
        )
        atexit.register(self.shutdown)

    def workers_healthy(self) -> bool:
        """Return True if all data-parallel workers are alive (or single-GPU mode)."""
        if not self._workers:
            return True
        return all(w.is_alive() for w in self._workers)

    def _restart_worker(self, worker_idx: int) -> None:
        """Terminate a dead worker and spawn a replacement."""
        old = self._workers[worker_idx]
        gpu_id = self._gpu_ids[worker_idx]

        if old.is_alive():
            old.terminate()
            old.join(timeout=5)
        if old.is_alive():
            old.kill()
            old.join(timeout=5)

        new_input_q = self._ctx.Queue()
        self._input_queues[worker_idx] = new_input_q

        p = self._ctx.Process(
            target=_dp_worker,
            args=(
                gpu_id,
                self._engine_kwargs,
                self.config.inference.max_tokens,
                self.config.inference.temperature,
                self.config.inference.top_p,
                new_input_q,
                self._output_queue,
                self.config.inference.worker_batch_size,
            ),
        )
        p.start()
        self._workers[worker_idx] = p
        LOGGER.warning(
            "Restarted worker %d (GPU %s) with PID %d",
            worker_idx, gpu_id, p.pid,
        )

    def infer_batch(self, images: Sequence["Image.Image"]) -> List[str]:
        if not images:
            return []

        if self._workers:
            return self._infer_data_parallel(images)
        return self._infer_single(images)

    def _infer_single(self, images: Sequence["Image.Image"]) -> List[str]:
        t0 = time.monotonic()

        messages_list = [
            [{"role": "user", "content": [
                {"type": "image_pil", "image_pil": image},
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

    def _infer_data_parallel(self, images: Sequence["Image.Image"]) -> List[str]:
        t0 = time.monotonic()
        n_workers = len(self._workers)
        images_list = list(images)

        # Restart any dead workers before dispatching
        for i, w in enumerate(self._workers):
            if not w.is_alive():
                LOGGER.warning("Worker %d found dead before dispatch, restarting", i)
                self._restart_worker(i)

        # Interleaved split for even distribution
        chunks: List[List] = [[] for _ in range(n_workers)]
        index_maps: List[List[int]] = [[] for _ in range(n_workers)]
        for i, img in enumerate(images_list):
            worker_idx = i % n_workers
            chunks[worker_idx].append(img)
            index_maps[worker_idx].append(i)

        # Dispatch to workers
        tasks_sent = 0
        for worker_idx, chunk in enumerate(chunks):
            if chunk:
                self._input_queues[worker_idx].put((worker_idx, chunk))
                tasks_sent += 1

        # Collect results
        results = [""] * len(images_list)
        for _ in range(tasks_sent):
            try:
                task_id, worker_results, error = self._output_queue.get(timeout=300)
            except Exception:
                # Check for dead workers and restart them
                dead = [i for i, w in enumerate(self._workers) if not w.is_alive()]
                if dead:
                    for i in dead:
                        self._restart_worker(i)
                    raise RuntimeError(f"Worker(s) {dead} died during inference (restarted)")
                raise

            if error is not None:
                raise error

            for local_idx, global_idx in enumerate(index_maps[task_id]):
                results[global_idx] = worker_results[local_idx]

        t_infer = time.monotonic()
        LOGGER.info(
            "Data-parallel inference: %d images across %d GPUs | %.2fs | %.2f pages/s",
            len(images_list), n_workers,
            t_infer - t0,
            len(images_list) / (t_infer - t0),
        )
        return results

    def shutdown(self) -> None:
        """Shut down worker processes cleanly."""
        if not self._workers:
            return

        for q in self._input_queues:
            try:
                q.put(None)
            except Exception:
                pass

        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                LOGGER.warning("Worker %s did not exit in time, terminating", p.pid)
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                LOGGER.warning("Worker %s still alive after terminate, killing", p.pid)
                p.kill()
                p.join(timeout=5)

        self._workers.clear()
        self._input_queues.clear()
        LOGGER.info("All data-parallel workers shut down")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
