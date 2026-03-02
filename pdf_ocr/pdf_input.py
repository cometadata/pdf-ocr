from __future__ import annotations

import enum
import logging
import multiprocessing
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set

import pypdfium2 as pdfium
from PIL import Image

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:  # pragma: no cover
    HfApi = None  # type: ignore[assignment,misc]
    hf_hub_download = None  # type: ignore[assignment]

from .config import ModelConfig, PdfRenderingConfig

LOGGER = logging.getLogger(__name__)


class InputType(enum.Enum):
    PDF_FILE = "pdf_file"
    DIRECTORY = "directory"
    HF_DATASET = "hf_dataset"


@dataclass
class PageImage:
    doc_id: str
    source: str
    page_index: int
    image: Image.Image


def render_page(page, cfg: PdfRenderingConfig) -> Image.Image:
    scale = cfg.dpi / 72
    image = page.render(scale=scale).to_pil()
    w, h = image.size
    if max(w, h) > cfg.max_dimension:
        ratio = cfg.max_dimension / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def render_pdf(source: str | Path, cfg: PdfRenderingConfig, doc_id: Optional[str] = None) -> Iterator[PageImage]:
    t0 = time.monotonic()
    path = Path(source)
    if doc_id is None:
        doc_id = path.stem

    pdf = pdfium.PdfDocument(str(path))
    n_pages = len(pdf)
    for i in range(n_pages):
        image = render_page(pdf[i], cfg)
        yield PageImage(doc_id=doc_id, source=str(path), page_index=i, image=image)
    LOGGER.info("Rendered %d pages from %s in %.2fs", n_pages, doc_id, time.monotonic() - t0)


def render_pdf_bytes(data: bytes, cfg: PdfRenderingConfig, doc_id: str, source: str = "") -> Iterator[PageImage]:
    t0 = time.monotonic()
    pdf = pdfium.PdfDocument(data)
    n_pages = len(pdf)
    for i in range(n_pages):
        image = render_page(pdf[i], cfg)
        yield PageImage(doc_id=doc_id, source=source or doc_id, page_index=i, image=image)
    LOGGER.info("Rendered %d pages from %s in %.2fs", n_pages, doc_id, time.monotonic() - t0)


_RENDER_CHILD_TIMEOUT = 600


def _render_one_process(path_str, doc_id, cfg, completed_set, result_queue):
    """Target for multiprocessing.Process — renders one PDF in an isolated child."""
    import faulthandler
    faulthandler.enable()
    try:
        pdf = pdfium.PdfDocument(path_str)
        n_pages = len(pdf)
        if len(completed_set) >= n_pages:
            return
        for i in range(n_pages):
            if i in completed_set:
                continue
            image = render_page(pdf[i], cfg)
            result_queue.put(PageImage(doc_id=doc_id, source=path_str, page_index=i, image=image))
    except Exception:
        LOGGER.warning("Failed to render %s in subprocess", doc_id, exc_info=True)


def _render_one_isolated(item, cfg, completed_pages, out_queue, mp_ctx):
    """Thread-level wrapper: spawns a subprocess for the pdfium work."""
    path, doc_id = item
    doc_completed = completed_pages.get(doc_id, set())
    LOGGER.debug("Starting render of %s", doc_id)
    t0 = time.monotonic()

    result_queue = mp_ctx.Queue()
    proc = mp_ctx.Process(
        target=_render_one_process,
        args=(str(path), doc_id, cfg, doc_completed, result_queue),
    )
    proc.start()

    rendered = 0
    timed_out = False
    deadline = time.monotonic() + _RENDER_CHILD_TIMEOUT

    while proc.is_alive():
        if time.monotonic() > deadline:
            timed_out = True
            LOGGER.warning("Render of %s timed out after %ds, killing child", doc_id, _RENDER_CHILD_TIMEOUT)
            proc.kill()
            proc.join(timeout=5)
            break
        try:
            page_image = result_queue.get(timeout=0.5)
            out_queue.put(page_image)
            rendered += 1
        except queue.Empty:
            continue

    while True:
        try:
            page_image = result_queue.get_nowait()
            out_queue.put(page_image)
            rendered += 1
        except queue.Empty:
            break

    if not timed_out:
        if proc.exitcode and proc.exitcode < 0:
            import signal as _signal
            sig = -proc.exitcode
            sig_name = _signal.Signals(sig).name if sig in _signal.Signals._value2member_map_ else str(sig)
            LOGGER.warning(
                "Render of %s crashed (signal %s), recovered %d pages",
                doc_id, sig_name, rendered,
            )
        elif proc.exitcode and proc.exitcode > 0:
            LOGGER.warning("Render of %s exited with code %d, recovered %d pages", doc_id, proc.exitcode, rendered)
        else:
            LOGGER.info(
                "Rendered %d pages from %s in %.2fs (skipped %d completed)",
                rendered, doc_id, time.monotonic() - t0, len(doc_completed),
            )

    result_queue.close()
    result_queue.join_thread()


def parallel_render(
    pdf_items: Iterable[tuple[Path, str]],
    cfg: PdfRenderingConfig,
    completed_pages: Dict[str, Set[int]],
    num_workers: int = 4,
    queue_size: int = 256,
    _use_processes: bool = True,
) -> Iterator[PageImage]:
    out_queue: queue.Queue[PageImage | None] = queue.Queue(maxsize=queue_size)
    error: list[BaseException] = []
    error_lock = threading.Lock()

    if _use_processes:
        mp_ctx = multiprocessing.get_context("spawn")

        def _producer() -> None:
            try:
                with ThreadPoolExecutor(max_workers=num_workers) as pool:
                    futures = []
                    for item in pdf_items:
                        futures.append(
                            pool.submit(_render_one_isolated, item, cfg, completed_pages, out_queue, mp_ctx)
                        )
                    for f in futures:
                        f.result()
            except Exception as exc:
                with error_lock:
                    error.append(exc)
            finally:
                out_queue.put(None)
    else:
        def _render_one(item: tuple[Path, str]) -> None:
            path, doc_id = item
            try:
                pdf = pdfium.PdfDocument(str(path))
                n_pages = len(pdf)
                doc_completed = completed_pages.get(doc_id, set())
                if len(doc_completed) >= n_pages:
                    LOGGER.debug("Skipping fully completed doc %s (%d pages)", doc_id, n_pages)
                    return
                t0 = time.monotonic()
                rendered = 0
                for i in range(n_pages):
                    if i in doc_completed:
                        continue
                    image = render_page(pdf[i], cfg)
                    out_queue.put(PageImage(
                        doc_id=doc_id, source=str(path), page_index=i, image=image,
                    ))
                    rendered += 1
                LOGGER.info(
                    "Rendered %d pages from %s in %.2fs (skipped %d completed)",
                    rendered, doc_id, time.monotonic() - t0, len(doc_completed),
                )
            except Exception:
                LOGGER.warning("Failed to render %s, skipping document", doc_id, exc_info=True)

        def _producer() -> None:
            try:
                with ThreadPoolExecutor(max_workers=num_workers) as pool:
                    futures = []
                    for item in pdf_items:
                        futures.append(pool.submit(_render_one, item))
                    for f in futures:
                        f.result()
            except Exception as exc:
                with error_lock:
                    error.append(exc)
            finally:
                out_queue.put(None)

    producer = threading.Thread(target=_producer, daemon=True)
    producer.start()

    while True:
        item = out_queue.get()
        if item is None:
            break
        yield item

    producer.join(timeout=5.0)

    with error_lock:
        if error:
            raise error[0]


def detect_input_type(source: str) -> InputType:
    if source.startswith("hf://"):
        return InputType.HF_DATASET
    path = Path(source)
    if path.is_file() and path.suffix.lower() == ".pdf":
        return InputType.PDF_FILE
    if path.is_dir():
        return InputType.DIRECTORY
    if "/" in source and not path.exists():
        return InputType.HF_DATASET
    raise ValueError(f"Cannot determine input type for: {source!r}")


def _load_single_pdf(source: str, cfg: PdfRenderingConfig) -> Iterator[PageImage]:
    LOGGER.info("Loading PDF: %s", source)
    yield from render_pdf(source, cfg)


def _load_directory(
    source: str,
    cfg: PdfRenderingConfig,
    completed_pages: Optional[Dict[str, Set[int]]] = None,
    render_workers: int = 4,
) -> Iterator[PageImage]:
    base = Path(source)
    pdf_files = sorted(base.rglob("*.pdf"))
    LOGGER.info("Found %d PDFs in %s", len(pdf_files), source)

    items = [(p, p.relative_to(base).with_suffix("").as_posix()) for p in pdf_files]
    yield from parallel_render(
        items, cfg,
        completed_pages=completed_pages or {},
        num_workers=render_workers,
    )


def _load_hf_dataset(source: str, cfg: PdfRenderingConfig, pdf_column: Optional[str] = None,
                     split: str = "train", token: Optional[str] = None,
                     completed_pages: Optional[Dict[str, Set[int]]] = None,
                     render_workers: int = 4) -> Iterator[PageImage]:
    repo_id = source.removeprefix("hf://")

    try:
        yield from _load_hf_repo_files(
            repo_id, cfg, token=token,
            completed_pages=completed_pages, render_workers=render_workers,
        )
        return
    except Exception:
        pass

    from datasets import load_dataset
    try:
        ds = load_dataset(repo_id, split=split, token=token, streaming=True)
    except Exception as exc:
        raise ValueError(f"Could not load HF source {repo_id!r}: {exc}") from exc

    col = _detect_pdf_column(ds, pdf_column)
    if col is None:
        raise ValueError(
            f"No PDF column found in dataset {repo_id!r}. "
            f"Columns: {ds.column_names}. Use --pdf-column to specify."
        )

    LOGGER.info("Streaming column %r from dataset %s", col, repo_id)

    for idx, row in enumerate(ds):
        value = row[col]
        doc_id = row.get("id", row.get("doc_id", f"doc_{idx:05d}"))
        if isinstance(doc_id, int):
            doc_id = f"doc_{doc_id:05d}"

        try:
            if isinstance(value, bytes):
                yield from render_pdf_bytes(value, cfg, doc_id=str(doc_id), source=repo_id)
            elif isinstance(value, str):
                if value.startswith("http://") or value.startswith("https://"):
                    import requests as req
                    resp = req.get(value, timeout=60)
                    resp.raise_for_status()
                    yield from render_pdf_bytes(resp.content, cfg, doc_id=str(doc_id), source=value)
                else:
                    yield from render_pdf(value, cfg, doc_id=str(doc_id))
            elif isinstance(value, dict) and "bytes" in value:
                yield from render_pdf_bytes(value["bytes"], cfg, doc_id=str(doc_id), source=repo_id)
            else:
                LOGGER.warning("Skipping row %d: unsupported value type %s", idx, type(value))
        except Exception:
            LOGGER.warning("Failed to render document %s, skipping", doc_id, exc_info=True)


def _load_hf_repo_files(
    repo_id: str,
    cfg: PdfRenderingConfig,
    token: Optional[str] = None,
    completed_pages: Optional[Dict[str, Set[int]]] = None,
    render_workers: int = 4,
) -> Iterator[PageImage]:
    api = HfApi()
    all_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    pdf_files = sorted(f for f in all_files if f.lower().endswith(".pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in repo {repo_id}")

    LOGGER.info("Found %d PDFs in HF repo %s (streaming downloads)", len(pdf_files), repo_id)

    def _download_one(filename: str) -> tuple[Path, str]:
        local_path = hf_hub_download(
            repo_id, filename, repo_type="dataset", token=token,
        )
        doc_id = Path(filename).with_suffix("").as_posix()
        return (Path(local_path), doc_id)

    def _iter_downloads() -> Iterator[tuple[Path, str]]:
        with ThreadPoolExecutor(max_workers=4) as dl_pool:
            yield from dl_pool.map(_download_one, pdf_files)

    yield from parallel_render(
        _iter_downloads(), cfg,
        completed_pages=completed_pages or {},
        num_workers=render_workers,
    )


def _detect_pdf_column(ds, pdf_column: Optional[str] = None) -> Optional[str]:
    if pdf_column and pdf_column in ds.column_names:
        return pdf_column

    candidates = ["pdf", "file", "content", "data", "document", "pdf_data", "pdf_bytes"]
    for name in candidates:
        if name in ds.column_names:
            return name

    return None


def load_pdfs(source: str, config: ModelConfig, pdf_column: Optional[str] = None,
              split: str = "train", token: Optional[str] = None,
              completed_pages: Optional[Set] = None) -> Iterator[PageImage]:
    input_type = detect_input_type(source)
    LOGGER.info("Detected input type: %s for source: %s", input_type.value, source)

    render_workers = config.inference.render_workers

    completed_by_doc: Dict[str, Set[int]] = {}
    if completed_pages:
        for doc_id, page_index in completed_pages:
            completed_by_doc.setdefault(doc_id, set()).add(page_index)

    if input_type == InputType.PDF_FILE:
        yield from _load_single_pdf(source, config.pdf_rendering)
    elif input_type == InputType.DIRECTORY:
        yield from _load_directory(
            source, config.pdf_rendering,
            completed_pages=completed_by_doc, render_workers=render_workers,
        )
    elif input_type == InputType.HF_DATASET:
        yield from _load_hf_dataset(
            source, config.pdf_rendering, pdf_column=pdf_column,
            split=split, token=token,
            completed_pages=completed_by_doc, render_workers=render_workers,
        )
