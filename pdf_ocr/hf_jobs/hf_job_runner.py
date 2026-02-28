# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pdf-ocr[offline] @ https://github.com/cometadata/pdf-ocr/archive/main.tar.gz",
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

"""
Entrypoint for HuggingFace Jobs.

Installs pdf_ocr from GitHub and runs the conversion pipeline.
All configuration is via environment variables.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def ensure_code_checkout() -> Path:
    """Download supplementary job code from HF Hub if JOB_CODE_REPO is set.

    The primary pdf_ocr package is installed from GitHub via PEP 723
    dependencies. This function is only needed if extra code is hosted
    in a HF repo.
    """
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
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Download code if running in HF Jobs
    code_dir = ensure_code_checkout()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from pdf_ocr import convert
    from pdf_ocr.storage import push_to_hub

    source = os.environ.get("INPUT_SOURCE")
    if not source:
        raise RuntimeError("INPUT_SOURCE environment variable must be set")

    model_config = os.environ.get("MODEL_CONFIG", "lighton_ocr_2_1b")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    hf_token = os.environ.get("HF_TOKEN")
    private = os.environ.get("PRIVATE", "false").lower() in {"true", "1", "yes"}
    batch_size = int(os.environ["BATCH_SIZE"]) if os.environ.get("BATCH_SIZE") else None
    max_pages = int(os.environ["MAX_PAGES"]) if os.environ.get("MAX_PAGES") else None
    port = int(os.environ.get("PORT", "8000"))
    backend = os.environ.get("BACKEND")

    LOGGER.info("Starting pdf_ocr job: source=%s model=%s backend=%s", source, model_config, backend)

    results = convert(
        source=source,
        model=model_config,
        backend=backend,
        batch_size=batch_size,
        max_pages=max_pages,
        token=hf_token,
        port=port,
    )

    if hf_repo_id:
        push_to_hub(
            results,
            repo_id=hf_repo_id,
            private=private,
            token=hf_token,
        )
        LOGGER.info("Results pushed to %s", hf_repo_id)
    else:
        # Save locally as fallback
        from pdf_ocr.storage import save_local
        output_dir = os.environ.get("OUTPUT_DIR", "./outputs")
        save_local(results, output_dir)
        LOGGER.info("Results saved to %s", output_dir)

    total_pages = sum(len(r.pages) for r in results)
    LOGGER.info("Job complete: %d documents, %d pages", len(results), total_pages)


if __name__ == "__main__":
    main()
