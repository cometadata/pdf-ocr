from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

import pypdfium2 as pdfium
from PIL import Image

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None  # type: ignore[assignment]

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


def parallel_render(
    pdf_items: List[tuple[Path, str]],
    cfg: PdfRenderingConfig,
    completed_pages: Dict[str, Set[int]],
    num_workers: int = 4,
    queue_size: int = 256,
) -> Iterator[PageImage]:
    if not pdf_items:
        return

    out_queue: queue.Queue[PageImage | None] = queue.Queue(maxsize=queue_size)
    error: list[BaseException] = []
    error_lock = threading.Lock()

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
                list(pool.map(_render_one, pdf_items))
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
        ds = load_dataset(repo_id, split=split, token=token)
    except Exception as exc:
        raise ValueError(f"Could not load HF source {repo_id!r}: {exc}") from exc

    col = _detect_pdf_column(ds, pdf_column)
    if col is None:
        raise ValueError(
            f"No PDF column found in dataset {repo_id!r}. "
            f"Columns: {ds.column_names}. Use --pdf-column to specify."
        )

    LOGGER.info("Using column %r from dataset %s (%d rows)", col, repo_id, len(ds))

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
    local_dir = snapshot_download(
        repo_id, allow_patterns="*.pdf", repo_type="dataset", token=token,
    )

    pdf_paths = sorted(Path(local_dir).rglob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in repo {repo_id}")

    LOGGER.info("Found %d PDFs in HF repo %s", len(pdf_paths), repo_id)

    items = [(p, p.relative_to(local_dir).with_suffix("").as_posix()) for p in pdf_paths]
    yield from parallel_render(
        items, cfg,
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
