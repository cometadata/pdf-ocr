from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Protocol, Sequence, Tuple

from .pdf_input import PageImage

if TYPE_CHECKING:
    from PIL import Image


class InferenceEngine(Protocol):
    def infer_batch(self, images: Sequence["Image.Image"]) -> List[str]: ...

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


def _infer_with_retry(
    client: InferenceEngine,
    batch: List[PageImage],
    max_depth: int = 3,
    subdivision: int = 2,
) -> List[str]:
    """Infer a batch, subdividing and retrying on failure.

    When infer_batch raises, the batch is split in half and each sub-batch
    is retried recursively up to *max_depth* levels. Pages that still fail
    at max depth get empty strings, isolating single bad pages instead of
    blanking entire batches.
    """
    images = [p.image for p in batch]
    try:
        return client.infer_batch(images)
    except Exception:
        if max_depth <= 0 or len(batch) <= 1:
            LOGGER.error("Batch of %d failed at max depth", len(batch))
            return [""] * len(batch)
        LOGGER.warning(
            "Batch of %d failed; subdividing (depth=%d)",
            len(batch), max_depth - 1,
        )
        sub_size = max(1, len(batch) // subdivision)
        sub_batches = [batch[i:i + sub_size] for i in range(0, len(batch), sub_size)]
        results: List[str] = []
        for sb in sub_batches:
            results.extend(_infer_with_retry(client, sb, max_depth - 1, subdivision))
        return results


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


def convert_pages_streaming(
    pages: Iterator[PageImage],
    client: InferenceEngine,
    batch_size: int = 4,
    max_pages: int | None = None,
    max_retry_depth: int = 3,
    checkpoint_dir: Optional[Path] = None,
    resume_from_checkpoint: bool = False,
) -> Iterator[List[Tuple[str, str, PageResult]]]:
    """Process pages in batches, yielding results per batch instead of accumulating.

    Each yield is a list of (doc_id, source, PageResult) tuples for one batch.
    """
    from .storage import load_checkpoints, save_batch_checkpoint

    start = time.time()
    page_count = 0
    batch_count = 0
    skip_pages = 0

    if resume_from_checkpoint and checkpoint_dir is not None:
        previous = load_checkpoints(checkpoint_dir)
        if previous:
            skip_pages = len(previous)
            LOGGER.info("Resuming from checkpoint: skipping %d already-processed pages", skip_pages)
            yield previous

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

    batches_iter = _batch_pages(limited_pages(), batch_size)
    # Peek-ahead for pipeline overlap
    current_batch = next(batches_iter, None)

    while current_batch is not None:
        batch = current_batch
        current_batch = next(batches_iter, None)  # peek next

        batch_count += 1
        batch_start = time.time()

        LOGGER.info("Processing batch %d (%d pages)", batch_count, len(batch))
        markdowns = _infer_with_retry(client, batch, max_depth=max_retry_depth)

        # Kick off pre-encoding of next batch while we process results
        if current_batch is not None:
            prep = getattr(client, "start_next_prep", None)
            if prep is not None:
                prep([p.image for p in current_batch])

        batch_results: List[Tuple[str, str, PageResult]] = []
        for page, md in zip(batch, markdowns):
            result = (
                page.doc_id,
                page.source,
                PageResult(page_index=page.page_index, markdown=md.strip()),
            )
            batch_results.append(result)
            page.image.close()

        if checkpoint_dir is not None:
            ckpt_start = time.time()
            save_batch_checkpoint(batch_results, checkpoint_dir, batch_count - 1 + (skip_pages // max(batch_size, 1)))
            LOGGER.info("Checkpoint saved in %.2fs", time.time() - ckpt_start)

        batch_elapsed = time.time() - batch_start
        LOGGER.info(
            "Batch %d complete: %d pages in %.2fs (%.2f pages/s) | cumulative: %d pages in %.2fs (%.2f pages/s)",
            batch_count, len(batch), batch_elapsed,
            len(batch) / batch_elapsed if batch_elapsed > 0 else 0,
            page_count, time.time() - start,
            page_count / (time.time() - start) if (time.time() - start) > 0 else 0,
        )

        yield batch_results

    elapsed = time.time() - start
    total_pages = page_count + skip_pages
    LOGGER.info(
        "Conversion complete: %d pages in %.1fs",
        total_pages, elapsed,
    )


def convert_pages(
    pages: Iterator[PageImage],
    client: InferenceEngine,
    batch_size: int = 4,
    max_pages: int | None = None,
    max_retry_depth: int = 3,
    checkpoint_dir: Optional[Path] = None,
    resume_from_checkpoint: bool = False,
) -> List[ConversionResult]:
    """Process pages and return grouped results. Wraps convert_pages_streaming."""
    all_results: List[Tuple[str, str, PageResult]] = []
    for batch_results in convert_pages_streaming(
        pages,
        client,
        batch_size=batch_size,
        max_pages=max_pages,
        max_retry_depth=max_retry_depth,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    ):
        all_results.extend(batch_results)
    return _group_by_document(all_results)
