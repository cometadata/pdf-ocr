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
    from .convert import convert_pages
    from .pdf_input import load_pdfs

    config = load_config(model)
    config = config.with_overrides(batch_size=batch_size, max_tokens=None)

    effective_backend = backend or config.backend
    if batch_size is not None:
        effective_batch_size = batch_size
    elif effective_backend == "offline":
        effective_batch_size = 512
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
        from .offline import VLLMOfflineEngine
        client = VLLMOfflineEngine(config)
    else:
        from .server import VLLMClient, launch_vllm, shutdown_server, wait_for_server
        if base_url is None:
            base_url = f"http://127.0.0.1:{port}"
            server_process = launch_vllm(config, port=port, host=host)
            health_url = f"{base_url}/health"
            if not wait_for_server(health_url):
                shutdown_server(server_process)
                raise RuntimeError("vLLM server did not become ready in time")
        client = VLLMClient(base_url=base_url, config=config)

    try:
        pages = load_pdfs(source, config, pdf_column=pdf_column, split=split, token=hf_token)
        results = convert_pages(
            pages,
            client,
            batch_size=effective_batch_size,
            max_pages=max_pages,
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=checkpoint_dir is not None,
        )
    finally:
        if server_process is not None:
            from .server import shutdown_server
            shutdown_server(server_process)

    if output:
        if not is_local_output:
            from .storage import push_to_hub
            push_to_hub(results, repo_id=output, private=private, token=hf_token)
        else:
            from .storage import save_local
            save_local(results, output)

    if checkpoint_dir is not None:
        from .storage import clear_checkpoints
        clear_checkpoints(checkpoint_dir)

    return results
