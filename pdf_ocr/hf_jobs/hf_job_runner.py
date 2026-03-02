# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pdf-ocr @ https://github.com/cometadata/pdf-ocr/archive/main.tar.gz",
#     "huggingface-hub",
#     "datasets>=4.0.0",
#     "pyarrow>=12.0.0",
#     "pillow",
#     "pypdfium2",
#     "requests",
#     "openai",
#     "pyyaml",
# ]
# ///

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _log_memory():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    LOGGER.info("Memory: RSS = %.1f MB", rss_kb / 1024)
                    return
    except Exception:
        pass



def ensure_code_checkout() -> Path:
    repo_id = os.environ.get("JOB_CODE_REPO")
    if not repo_id:
        return Path(".")

    from huggingface_hub import snapshot_download

    repo_type = os.environ.get("JOB_CODE_REPO_TYPE", "dataset")
    revision = os.environ.get("JOB_CODE_REVISION")
    local_dir = Path(os.environ.get("JOB_CODE_LOCAL_DIR", "/tmp/pdf-ocr-job-code"))
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
    )
    return local_dir


def main() -> None:
    import faulthandler
    faulthandler.enable()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    def _sigterm_handler(signum, frame):
        LOGGER.info("Received SIGTERM, initiating graceful shutdown")
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    code_dir = ensure_code_checkout()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from pdf_ocr.config import load_config
    from pdf_ocr.convert import convert_pages_streaming
    from pdf_ocr.pdf_input import load_pdfs
    from pdf_ocr.storage import push_batch_to_hub, save_batch_incremental

    source = os.environ.get("INPUT_SOURCE")
    if not source:
        raise RuntimeError("INPUT_SOURCE environment variable must be set")

    model_config_name = os.environ.get("MODEL_CONFIG", "lighton_ocr_2_1b")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    hf_token = os.environ.get("HF_TOKEN")
    private = os.environ.get("PRIVATE", "false").lower() in {"true", "1", "yes"}
    batch_size = int(os.environ["BATCH_SIZE"]) if os.environ.get("BATCH_SIZE") else None
    max_pages = int(os.environ["MAX_PAGES"]) if os.environ.get("MAX_PAGES") else None
    port = int(os.environ.get("PORT", "8000"))
    backend = os.environ.get("BACKEND")

    LOGGER.info("Starting pdf_ocr job: source=%s model=%s backend=%s batch_size=%s", source, model_config_name, backend, batch_size)

    config = load_config(model_config_name)
    config = config.with_overrides(batch_size=batch_size, max_tokens=None)

    effective_backend = backend or config.backend
    if batch_size is not None:
        effective_batch_size = batch_size
    elif effective_backend == "offline":
        effective_batch_size = config.inference.offline_batch_size
    else:
        effective_batch_size = config.inference.batch_size

    if effective_backend == "offline":
        from pdf_ocr.engine_factory import create_offline_engine
        client = create_offline_engine(config)
    else:
        from pdf_ocr.server import VLLMClient, launch_vllm, shutdown_server, wait_for_server
        base_url = f"http://127.0.0.1:{port}"
        server_process = launch_vllm(config, port=port, host="0.0.0.0")
        health_url = f"{base_url}/health"
        if not wait_for_server(health_url):
            shutdown_server(server_process)
            raise RuntimeError("vLLM server did not become ready in time")
        client = VLLMClient(base_url=base_url, config=config)

    job_start = time.monotonic()

    no_resume = os.environ.get("NO_RESUME", "false").lower() in {"true", "1", "yes"}

    total_pages = 0
    shard_index = 0
    completed_pages: set = set()
    if hf_repo_id and not no_resume:
        from pdf_ocr.storage import load_hub_progress
        shard_index, completed_pages = load_hub_progress(hf_repo_id, token=hf_token)
    elif hf_repo_id and no_resume:
        LOGGER.info("Resume disabled (NO_RESUME); starting fresh")

    pages = load_pdfs(source, config, token=hf_token, completed_pages=completed_pages)

    flush_every = int(os.environ.get("FLUSH_EVERY", "3"))
    batch_count = 0
    pending_hub_rows = []
    output_dir = os.environ.get("OUTPUT_DIR", "./outputs")

    try:
        for batch_results in convert_pages_streaming(
            pages,
            client,
            batch_size=effective_batch_size,
            max_pages=max_pages,
            max_retry_depth=config.inference.max_retry_depth,
        ):
            total_pages += len(batch_results)
            batch_count += 1
            _log_memory()

            if hf_repo_id:
                pending_hub_rows.extend(batch_results)
                if batch_count % flush_every == 0 and pending_hub_rows:
                    push_batch_to_hub(
                        pending_hub_rows, repo_id=hf_repo_id,
                        shard_index=shard_index,
                        token=hf_token, private=private,
                    )
                    shard_index += 1
                    pending_hub_rows = []
            else:
                save_batch_incremental(batch_results, Path(output_dir))
    finally:
        if hf_repo_id and pending_hub_rows:
            try:
                LOGGER.info("Flushing %d pending rows on shutdown", len(pending_hub_rows))
                push_batch_to_hub(
                    pending_hub_rows, repo_id=hf_repo_id,
                    shard_index=shard_index,
                    token=hf_token, private=private,
                )
            except Exception:
                LOGGER.exception("Failed to flush pending rows on shutdown")

        if effective_backend != "offline":
            shutdown_server(server_process)

    elapsed = time.monotonic() - job_start
    LOGGER.info(
        "Performance summary: %d pages | %.1fs total | %.2f pages/s | backend=%s | batch_size=%s",
        total_pages, elapsed,
        total_pages / elapsed if elapsed > 0 else 0,
        backend, effective_batch_size,
    )


if __name__ == "__main__":
    main()
