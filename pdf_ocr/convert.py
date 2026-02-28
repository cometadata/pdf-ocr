"""Core conversion pipeline: PDF pages to markdown via vLLM."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from .pdf_input import PageImage
from .server import VLLMClient

LOGGER = logging.getLogger(__name__)


@dataclass
class PageResult:
    page_index: int
    markdown: str


@dataclass
class ConversionResult:
    doc_id: str
    source: str
    pages: list[PageResult]


def _batch_pages(pages: Iterator[PageImage], batch_size: int) -> Iterator[List[PageImage]]:
    batch: List[PageImage] = []
    for page in pages:
        batch.append(page)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _group_by_document(
    results: List[Tuple[str, str, PageResult]],
) -> List[ConversionResult]:
    docs: OrderedDict[str, ConversionResult] = OrderedDict()
    for doc_id, source, page_result in results:
        if doc_id not in docs:
            docs[doc_id] = ConversionResult(doc_id=doc_id, source=source, pages=[])
        docs[doc_id].pages.append(page_result)
    for doc in docs.values():
        doc.pages.sort(key=lambda p: p.page_index)
    return list(docs.values())


def convert_pages(
    pages: Iterator[PageImage],
    client: VLLMClient,
    batch_size: int = 4,
    max_pages: int | None = None,
    checkpoint_dir: Optional[Path] = None,
    resume_from_checkpoint: bool = False,
) -> List[ConversionResult]:
    """Convert page images to markdown via vLLM."""
    from .storage import load_checkpoints, save_batch_checkpoint

    start = time.time()
    all_results: List[Tuple[str, str, PageResult]] = []
    page_count = 0
    batch_count = 0
    skip_pages = 0

    if resume_from_checkpoint and checkpoint_dir is not None:
        previous = load_checkpoints(checkpoint_dir)
        if previous:
            all_results.extend(previous)
            skip_pages = len(previous)
            LOGGER.info("Resuming from checkpoint: skipping %d already-processed pages", skip_pages)

    def limited_pages():
        nonlocal page_count
        skipped = 0
        for page in pages:
            if skipped < skip_pages:
                skipped += 1
                page.image.close()
                continue
            if max_pages and page_count >= max_pages:
                break
            page_count += 1
            yield page

    for batch in _batch_pages(limited_pages(), batch_size):
        batch_count += 1
        images = [p.image for p in batch]

        LOGGER.info("Processing batch %d (%d pages)", batch_count, len(images))
        try:
            markdowns = client.infer_batch(images)
        except Exception:
            LOGGER.exception("Batch %d failed", batch_count)
            markdowns = [""] * len(batch)

        batch_results: List[Tuple[str, str, PageResult]] = []
        for page, md in zip(batch, markdowns):
            result = (
                page.doc_id,
                page.source,
                PageResult(page_index=page.page_index, markdown=md.strip()),
            )
            batch_results.append(result)
            page.image.close()

        all_results.extend(batch_results)

        if checkpoint_dir is not None:
            save_batch_checkpoint(batch_results, checkpoint_dir, batch_count - 1 + (skip_pages // max(batch_size, 1)))

    elapsed = time.time() - start
    total_pages = page_count + skip_pages
    results = _group_by_document(all_results)
    LOGGER.info(
        "Conversion complete: %d pages, %d documents, %.1fs",
        total_pages, len(results), elapsed,
    )
    return results
