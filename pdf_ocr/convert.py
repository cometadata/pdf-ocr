"""Core conversion pipeline: PDF pages to markdown via vLLM."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

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
    """Group page images into batches."""
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
    """Group page results by document, preserving order."""
    docs: OrderedDict[str, ConversionResult] = OrderedDict()
    for doc_id, source, page_result in results:
        if doc_id not in docs:
            docs[doc_id] = ConversionResult(doc_id=doc_id, source=source, pages=[])
        docs[doc_id].pages.append(page_result)
    # Sort pages within each document
    for doc in docs.values():
        doc.pages.sort(key=lambda p: p.page_index)
    return list(docs.values())


def convert_pages(
    pages: Iterator[PageImage],
    client: VLLMClient,
    batch_size: int = 4,
    max_pages: int | None = None,
) -> List[ConversionResult]:
    """Convert page images to markdown via vLLM.

    Args:
        pages: Iterator of PageImage objects.
        client: VLLMClient connected to a running vLLM server.
        batch_size: Number of pages per inference batch.
        max_pages: Optional limit on total pages processed.

    Returns:
        List of ConversionResult, one per document.
    """
    start = time.time()
    all_results: List[Tuple[str, str, PageResult]] = []
    page_count = 0
    batch_count = 0

    # Apply max_pages limit
    def limited_pages():
        nonlocal page_count
        for page in pages:
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

        for page, md in zip(batch, markdowns):
            all_results.append((
                page.doc_id,
                page.source,
                PageResult(page_index=page.page_index, markdown=md.strip()),
            ))
            page.image.close()

    elapsed = time.time() - start
    results = _group_by_document(all_results)
    LOGGER.info(
        "Conversion complete: %d pages, %d documents, %.1fs",
        page_count, len(results), elapsed,
    )
    return results
