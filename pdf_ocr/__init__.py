from pdf_ocr.config import ModelConfig, load_config
from pdf_ocr.convert import ConversionResult, PageResult

import logging
import os
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger(__name__)


def convert(
    source: str,
    model: str = "lighton_ocr_2_1b",
    *,
    output: Optional[str] = None,
    private: bool = False,
    base_url: Optional[str] = None,
    backend: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_pages: Optional[int] = None,
    pdf_column: Optional[str] = None,
    split: str = "train",
    token: Optional[str] = None,
    port: int = 8000,
    host: str = "0.0.0.0",
) -> List[ConversionResult]:
    from .config import load_config
    from .pdf_input import load_pdfs

    config = load_config(model)
    config = config.with_overrides(batch_size=batch_size, max_tokens=None)

    effective_backend = backend or config.backend
    if batch_size is not None:
        effective_batch_size = batch_size
    elif effective_backend == "offline":
        effective_batch_size = config.inference.offline_batch_size
    else:
        effective_batch_size = config.inference.batch_size
    hf_token = token or os.environ.get("HF_TOKEN")

    if base_url is not None and effective_backend == "offline":
        LOGGER.warning("base_url provided; ignoring backend='offline' and using server mode")
        effective_backend = "server"

    checkpoint_dir = None
    is_local_output = False
    if output and not ("/" in output and not output.startswith(".") and not output.startswith("/")):
        is_local_output = True
        checkpoint_dir = Path(output)

    server_process = None

    if effective_backend == "offline":
        from .engine_factory import create_offline_engine
        client = create_offline_engine(config)
    else:
        from .server import VLLMClient, launch_vllm, shutdown_server, wait_for_server
        data_parallel_size = None
        if base_url is None:
            from .gpu import get_physical_gpu_ids
            gpu_ids = get_physical_gpu_ids()
            has_tp = "tensor-parallel-size" in config.vllm_args
            has_dp = "data-parallel-size" in config.vllm_args
            if len(gpu_ids) > 1 and not has_tp and not has_dp:
                data_parallel_size = len(gpu_ids)
                LOGGER.info(
                    "Auto-detected %d GPUs, using data-parallel-size=%d",
                    len(gpu_ids), data_parallel_size,
                )
            base_url = f"http://127.0.0.1:{port}"
            server_process = launch_vllm(
                config, port=port, host=host,
                data_parallel_size=data_parallel_size,
            )
            health_url = f"{base_url}/health"
            if not wait_for_server(health_url):
                shutdown_server(server_process)
                raise RuntimeError("vLLM server did not become ready in time")
        client = VLLMClient(base_url=base_url, config=config)

        if data_parallel_size and data_parallel_size > 1 and batch_size is None:
            effective_batch_size = min(
                effective_batch_size * data_parallel_size,
                config.inference.offline_batch_size,
            )
            LOGGER.info(
                "Scaled batch size to %d for %d-way data parallelism",
                effective_batch_size, data_parallel_size,
            )

    try:
        from .convert import convert_pages_streaming, _group_by_document

        pages = load_pdfs(source, config, pdf_column=pdf_column, split=split, token=hf_token)

        all_results = []
        batch_count = 0
        flush_every = config.inference.flush_every
        hub_shard_index = 0
        pending_hub_rows = []

        for batch_results in convert_pages_streaming(
            pages,
            client,
            batch_size=effective_batch_size,
            max_pages=max_pages,
            max_retry_depth=config.inference.max_retry_depth,
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=checkpoint_dir is not None,
        ):
            all_results.extend(batch_results)
            batch_count += 1

            if output and is_local_output:
                from .storage import save_batch_incremental
                save_batch_incremental(batch_results, Path(output))
            elif output and not is_local_output:
                pending_hub_rows.extend(batch_results)
                if batch_count % flush_every == 0 and pending_hub_rows:
                    from .storage import push_batch_to_hub
                    push_batch_to_hub(
                        pending_hub_rows, repo_id=output,
                        shard_index=hub_shard_index,
                        token=hf_token, private=private,
                    )
                    hub_shard_index += 1
                    pending_hub_rows = []

        if output and not is_local_output and pending_hub_rows:
            from .storage import push_batch_to_hub
            push_batch_to_hub(
                pending_hub_rows, repo_id=output,
                shard_index=hub_shard_index,
                token=hf_token, private=private,
            )

    finally:
        if server_process is not None:
            from .server import shutdown_server
            shutdown_server(server_process)

    if checkpoint_dir is not None:
        from .storage import clear_checkpoints
        clear_checkpoints(checkpoint_dir)

    return _group_by_document(all_results)
