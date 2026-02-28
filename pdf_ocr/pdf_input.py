"""PDF loading and page rendering for files, directories, and HF datasets."""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import pypdfium2 as pdfium
from PIL import Image

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
    """Render a pypdfium2 page to a PIL Image."""
    scale = cfg.dpi / 72
    image = page.render(scale=scale).to_pil()
    w, h = image.size
    if max(w, h) > cfg.max_dimension:
        ratio = cfg.max_dimension / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def render_pdf(source: str | Path, cfg: PdfRenderingConfig, doc_id: Optional[str] = None) -> List[PageImage]:
    """Render all pages from a PDF file."""
    t0 = time.monotonic()
    path = Path(source)
    if doc_id is None:
        doc_id = path.stem

    pdf = pdfium.PdfDocument(str(path))
    pages = []
    for i in range(len(pdf)):
        image = render_page(pdf[i], cfg)
        pages.append(PageImage(doc_id=doc_id, source=str(path), page_index=i, image=image))
    LOGGER.info("Rendered %d pages from %s in %.2fs", len(pages), doc_id, time.monotonic() - t0)
    return pages


def render_pdf_bytes(data: bytes, cfg: PdfRenderingConfig, doc_id: str, source: str = "") -> List[PageImage]:
    """Render all pages from PDF bytes."""
    t0 = time.monotonic()
    pdf = pdfium.PdfDocument(data)
    pages = []
    for i in range(len(pdf)):
        image = render_page(pdf[i], cfg)
        pages.append(PageImage(doc_id=doc_id, source=source or doc_id, page_index=i, image=image))
    LOGGER.info("Rendered %d pages from %s in %.2fs", len(pages), doc_id, time.monotonic() - t0)
    return pages


def detect_input_type(source: str) -> InputType:
    """Auto-detect the input type from the source string."""
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
    """Load pages from a single PDF file."""
    LOGGER.info("Loading PDF: %s", source)
    yield from render_pdf(source, cfg)


def _load_directory(source: str, cfg: PdfRenderingConfig) -> Iterator[PageImage]:
    """Load pages from all PDFs in a directory."""
    base = Path(source)
    pdf_files = sorted(base.rglob("*.pdf"))
    LOGGER.info("Found %d PDFs in %s", len(pdf_files), source)
    for pdf_path in pdf_files:
        doc_id = pdf_path.relative_to(base).with_suffix("").as_posix()
        yield from render_pdf(pdf_path, cfg, doc_id=doc_id)


def _load_hf_dataset(source: str, cfg: PdfRenderingConfig, pdf_column: Optional[str] = None,
                     split: str = "train", token: Optional[str] = None) -> Iterator[PageImage]:
    """Load pages from a HuggingFace dataset or repo containing PDFs."""
    repo_id = source.removeprefix("hf://")

    try:
        yield from _load_hf_repo_files(repo_id, cfg, token=token)
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


def _load_hf_repo_files(repo_id: str, cfg: PdfRenderingConfig, token: Optional[str] = None) -> Iterator[PageImage]:
    """Load PDFs from a HuggingFace repo (raw files, not a dataset)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset")
    pdf_files = [f for f in files if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in repo {repo_id}")

    LOGGER.info("Found %d PDFs in HF repo %s", len(pdf_files), repo_id)

    for pdf_path in sorted(pdf_files):
        local_path = hf_hub_download(repo_id, pdf_path, repo_type="dataset", token=token)
        doc_id = Path(pdf_path).with_suffix("").as_posix()
        yield from render_pdf(local_path, cfg, doc_id=doc_id)


def _detect_pdf_column(ds, pdf_column: Optional[str] = None) -> Optional[str]:
    """Detect which column contains PDF data."""
    if pdf_column and pdf_column in ds.column_names:
        return pdf_column

    # Try common names
    candidates = ["pdf", "file", "content", "data", "document", "pdf_data", "pdf_bytes"]
    for name in candidates:
        if name in ds.column_names:
            return name

    return None


def load_pdfs(source: str, config: ModelConfig, pdf_column: Optional[str] = None,
              split: str = "train", token: Optional[str] = None) -> Iterator[PageImage]:
    """Auto-detect input type and yield page images."""
    input_type = detect_input_type(source)
    LOGGER.info("Detected input type: %s for source: %s", input_type.value, source)

    if input_type == InputType.PDF_FILE:
        yield from _load_single_pdf(source, config.pdf_rendering)
    elif input_type == InputType.DIRECTORY:
        yield from _load_directory(source, config.pdf_rendering)
    elif input_type == InputType.HF_DATASET:
        yield from _load_hf_dataset(source, config.pdf_rendering, pdf_column=pdf_column,
                                     split=split, token=token)
