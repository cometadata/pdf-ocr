"""pdf_ocr: Convert PDFs to Markdown using VLM models via vLLM."""

from pdf_ocr.config import ModelConfig, load_config
from pdf_ocr.convert import ConversionResult, PageResult

import logging
import os
from typing import List, Optional

LOGGER = logging.getLogger(__name__)


def convert(
    source: str,
    model: str = "lighton_ocr_2_1b",
    *,
    output: Optional[str] = None,
    private: bool = False,
    base_url: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_pages: Optional[int] = None,
    pdf_column: Optional[str] = None,
    split: str = "train",
    token: Optional[str] = None,
    port: int = 8000,
    host: str = "0.0.0.0",
) -> List[ConversionResult]:
    """Convert PDFs to markdown using a VLM model via vLLM.

    Args:
        source: PDF file path, directory, or HF dataset reference.
        model: Model config name, HF model ID, or path to YAML config.
        output: Local output directory path or HF repo ID for results.
        private: If True, create private HF dataset (only with HF output).
        base_url: URL of existing vLLM server. If provided, skips server launch.
        batch_size: Override config batch size.
        max_pages: Limit total pages processed.
        pdf_column: Column name for PDFs in HF datasets.
        split: Dataset split to use for HF dataset input.
        token: HuggingFace token.
        port: Port for vLLM server (when launching).
        host: Host for vLLM server (when launching).

    Returns:
        List of ConversionResult, one per input document.
    """
    from .config import load_config
    from .convert import convert_pages
    from .pdf_input import load_pdfs
    from .server import VLLMClient, launch_vllm, shutdown_server, wait_for_server

    config = load_config(model)
    config = config.with_overrides(batch_size=batch_size, max_tokens=None)

    effective_batch_size = batch_size or config.inference.batch_size
    hf_token = token or os.environ.get("HF_TOKEN")

    # Server lifecycle
    server_process = None
    if base_url is None:
        base_url = f"http://127.0.0.1:{port}"
        server_process = launch_vllm(config, port=port, host=host)
        health_url = f"{base_url}/health"
        if not wait_for_server(health_url):
            shutdown_server(server_process)
            raise RuntimeError("vLLM server did not become ready in time")

    try:
        client = VLLMClient(base_url=base_url, config=config)
        pages = load_pdfs(source, config, pdf_column=pdf_column, split=split, token=hf_token)
        results = convert_pages(pages, client, batch_size=effective_batch_size, max_pages=max_pages)
    finally:
        if server_process is not None:
            shutdown_server(server_process)

    # Output handling
    if output:
        if "/" in output and not output.startswith(".") and not output.startswith("/"):
            # Looks like an HF repo ID
            from .storage import push_to_hub
            push_to_hub(results, repo_id=output, private=private, token=hf_token)
        else:
            from .storage import save_local
            save_local(results, output)

    return results
