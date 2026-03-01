from __future__ import annotations

import logging
import queue
import threading
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


def _skip_pages(source: Iterator[PageImage], n: int) -> Iterator[PageImage]:
    skipped = 0
    for page in source:
        if skipped < n:
            skipped += 1
            page.image.close()
            continue
        yield page


class _EMPTY:
    pass


class Pipeline:
    def __init__(
        self,
        source: Iterator[PageImage],
        batch_size: int,
        stop_event: threading.Event,
    ) -> None:
        self._page_queue: queue.Queue[PageImage | None] = queue.Queue(
            maxsize=batch_size,
        )
        self._batch_queue: queue.Queue[List[PageImage] | None] = queue.Queue(
            maxsize=2,
        )
        self._stop_event = stop_event
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._peeked: List[PageImage] | None | type[_EMPTY] = _EMPTY
        self._renderer = threading.Thread(
            target=self._render, args=(source,), daemon=True,
        )
        self._batcher = threading.Thread(
            target=self._assemble, args=(batch_size,), daemon=True,
        )

    def start(self) -> None:
        self._renderer.start()
        self._batcher.start()

    def _set_error(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc

    def _check_error(self) -> None:
        with self._lock:
            if self._error is not None:
                raise self._error

    def _safe_put(self, q: queue.Queue, item: object) -> bool:
        while not self._stop_event.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _safe_get(self, q: queue.Queue) -> object | None:
        while not self._stop_event.is_set():
            try:
                return q.get(timeout=0.5)
            except queue.Empty:
                continue
        return None

    def _render(self, source: Iterator[PageImage]) -> None:
        try:
            for page in source:
                if self._stop_event.is_set():
                    return
                if not self._safe_put(self._page_queue, page):
                    return
        except Exception as exc:
            self._set_error(exc)
        finally:
            self._safe_put(self._page_queue, None)

    def _assemble(self, batch_size: int) -> None:
        try:
            batch: List[PageImage] = []
            while True:
                item = self._safe_get(self._page_queue)
                if item is None:
                    break
                batch.append(item)
                if len(batch) >= batch_size:
                    if not self._safe_put(self._batch_queue, batch):
                        return
                    batch = []
            if batch:
                if not self._safe_put(self._batch_queue, batch):
                    return
        except Exception as exc:
            self._set_error(exc)
        finally:
            self._safe_put(self._batch_queue, None)

    def __iter__(self) -> "Pipeline":
        return self

    def __next__(self) -> List[PageImage]:
        if self._peeked is not _EMPTY:
            val = self._peeked
            self._peeked = _EMPTY
            if val is None:
                self._check_error()
                raise StopIteration
            return val

        item = self._safe_get(self._batch_queue)
        if item is None:
            self._check_error()
            raise StopIteration
        self._check_error()
        return item

    def try_peek_next(self) -> List[PageImage] | None:
        if self._peeked is not _EMPTY:
            return self._peeked if self._peeked is not None else None
        try:
            item = self._batch_queue.get_nowait()
        except queue.Empty:
            return None
        self._peeked = item
        if item is None:
            return None
        return item

    def close(self) -> None:
        self._stop_event.set()
        for t in (self._renderer, self._batcher):
            t.join(timeout=5.0)
        for q in (self._page_queue, self._batch_queue):
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    continue
                if isinstance(item, list):
                    for page in item:
                        page.image.close()
                else:
                    item.image.close()

    def __enter__(self) -> "Pipeline":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _infer_with_retry(
    client: InferenceEngine,
    batch: List[PageImage],
    max_depth: int = 3,
    subdivision: int = 2,
) -> List[str]:
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

    from .storage import load_checkpoints, save_batch_checkpoint

    stop_event = threading.Event()
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

    source = _skip_pages(pages, skip_pages)

    with Pipeline(source, batch_size, stop_event) as pipeline:
        for batch in pipeline:
            page_count += len(batch)
            batch_count += 1
            batch_start = time.time()

            LOGGER.info("Processing batch %d (%d pages)", batch_count, len(batch))
            markdowns = _infer_with_retry(client, batch, max_depth=max_retry_depth)

            next_batch = pipeline.try_peek_next()
            if next_batch is not None:
                prep = getattr(client, "start_next_prep", None)
                if prep is not None:
                    prep([p.image for p in next_batch])

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

            if max_pages and page_count >= max_pages:
                break

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
