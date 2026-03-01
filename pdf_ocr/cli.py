from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_ocr",
        description="Convert PDFs to Markdown using VLM models via vLLM.",
    )
    parser.add_argument(
        "source",
        help="PDF file, directory, or HuggingFace dataset reference",
    )
    parser.add_argument(
        "--model", default="lighton_ocr_2_1b",
        help="Model config name, HF model ID, or path to YAML config (default: lighton_ocr_2_1b)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Local output directory for markdown files",
    )
    parser.add_argument(
        "--hf-repo", default=None,
        help="HuggingFace repo ID to push results as a dataset",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create private HF dataset (only with --hf-repo)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="URL of existing vLLM server (skips server launch)",
    )
    parser.add_argument(
        "--backend", default=None,
        choices=["server", "offline"],
        help="Inference backend: 'server' (default) or 'offline' (vLLM LLM class)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Limit total pages processed",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override inference batch size from model config",
    )
    parser.add_argument(
        "--pdf-column", default=None,
        help="Column name containing PDF data in HF dataset",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split for HF dataset input (default: train)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for vLLM server when launching (default: 8000)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    from pdf_ocr import convert

    results = convert(
        source=args.source,
        model=args.model,
        output=args.output,
        base_url=args.base_url,
        backend=args.backend,
        batch_size=args.batch_size,
        max_pages=args.max_pages,
        pdf_column=args.pdf_column,
        split=args.split,
        port=args.port,
        token=os.environ.get("HF_TOKEN"),
    )

    if args.hf_repo:
        from pdf_ocr.storage import push_to_hub
        push_to_hub(
            results,
            repo_id=args.hf_repo,
            private=args.private,
            token=os.environ.get("HF_TOKEN"),
        )

    if args.output is None and args.hf_repo is None:
        for result in results:
            for page in result.pages:
                print(page.markdown)
                print()

    total_pages = sum(len(r.pages) for r in results)
    logging.getLogger(__name__).info(
        "Done: %d documents, %d pages", len(results), total_pages,
    )
