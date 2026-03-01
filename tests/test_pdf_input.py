import pytest
import threading
import types
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from PIL import Image

from pdf_ocr.pdf_input import (
    render_page, render_pdf, render_pdf_bytes, detect_input_type, InputType,
    _load_directory, _load_hf_repo_files, load_pdfs, PageImage, parallel_render,
)
from pdf_ocr.config import ModelConfig, PdfRenderingConfig


def test_render_page_basic():
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000))
    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)
    result = render_page(mock_page, cfg)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    mock_page.render.assert_called_once()


def test_render_page_resizes_when_exceeding_max_dimension():
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (3000, 4000))
    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)
    result = render_page(mock_page, cfg)

    assert max(result.size) <= 1540


def test_render_page_no_resize_when_within_limit():
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000))
    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)
    result = render_page(mock_page, cfg)

    assert result.size == (800, 1000)


def test_detect_input_type_single_pdf(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    assert detect_input_type(str(pdf_file)) == InputType.PDF_FILE


def test_detect_input_type_directory(tmp_path):
    assert detect_input_type(str(tmp_path)) == InputType.DIRECTORY


def test_detect_input_type_hf_dataset():
    assert detect_input_type("user/my-dataset") == InputType.HF_DATASET
    assert detect_input_type("hf://user/my-dataset") == InputType.HF_DATASET


def test_render_pdf_returns_iterator():
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000))

    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap

    mock_pdf = MagicMock()
    mock_pdf.__len__ = MagicMock(return_value=3)
    mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)

    with patch("pdf_ocr.pdf_input.pdfium.PdfDocument", return_value=mock_pdf):
        result = render_pdf("/fake/path.pdf", cfg)
        assert isinstance(result, types.GeneratorType)
        pages = list(result)
        assert len(pages) == 3


def test_render_pdf_bytes_returns_iterator():
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000))

    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap

    mock_pdf = MagicMock()
    mock_pdf.__len__ = MagicMock(return_value=2)
    mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)

    with patch("pdf_ocr.pdf_input.pdfium.PdfDocument", return_value=mock_pdf):
        result = render_pdf_bytes(b"fake-pdf-data", cfg, doc_id="test")
        assert isinstance(result, types.GeneratorType)
        pages = list(result)
        assert len(pages) == 2


def test_load_directory_skips_bad_pdf(tmp_path):
    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)

    (tmp_path / "a.pdf").write_bytes(b"dummy")
    (tmp_path / "corrupt.pdf").write_bytes(b"dummy")
    (tmp_path / "c.pdf").write_bytes(b"dummy")

    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (100, 100))
    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap
    good_pdf = MagicMock()
    good_pdf.__len__ = MagicMock(return_value=1)
    good_pdf.__getitem__ = MagicMock(return_value=mock_page)

    def pdf_factory(path):
        if "corrupt" in str(path):
            raise RuntimeError("corrupt PDF")
        return good_pdf

    with patch("pdf_ocr.pdf_input.pdfium.PdfDocument", side_effect=pdf_factory):
        pages = list(_load_directory(str(tmp_path), cfg))

    assert len(pages) == 2
    doc_ids = {p.doc_id for p in pages}
    assert "corrupt" not in doc_ids


def test_parallel_render_yields_all_pages():
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000))

    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap

    mock_pdf = MagicMock()
    mock_pdf.__len__ = MagicMock(return_value=3)
    mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)

    with patch("pdf_ocr.pdf_input.pdfium.PdfDocument", return_value=mock_pdf):
        items = [(Path("/fake/a.pdf"), "a"), (Path("/fake/b.pdf"), "b")]
        pages = list(parallel_render(items, cfg, completed_pages={}, num_workers=2))

    assert len(pages) == 6
    doc_ids = {p.doc_id for p in pages}
    assert doc_ids == {"a", "b"}


def test_parallel_render_skips_fully_completed_docs():
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000))

    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap

    mock_pdf = MagicMock()
    mock_pdf.__len__ = MagicMock(return_value=2)
    mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)

    completed = {"a": {0, 1}}  # doc "a" fully completed (2 pages)
    with patch("pdf_ocr.pdf_input.pdfium.PdfDocument", return_value=mock_pdf):
        items = [(Path("/fake/a.pdf"), "a"), (Path("/fake/b.pdf"), "b")]
        pages = list(parallel_render(items, cfg, completed_pages=completed, num_workers=2))

    assert len(pages) == 2
    assert all(p.doc_id == "b" for p in pages)


def test_parallel_render_skips_individual_completed_pages():
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000))

    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap

    mock_pdf = MagicMock()
    mock_pdf.__len__ = MagicMock(return_value=3)
    mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)

    completed = {"a": {0, 2}}  # pages 0 and 2 of doc "a" completed
    with patch("pdf_ocr.pdf_input.pdfium.PdfDocument", return_value=mock_pdf):
        items = [(Path("/fake/a.pdf"), "a")]
        pages = list(parallel_render(items, cfg, completed_pages=completed, num_workers=1))

    assert len(pages) == 1
    assert pages[0].page_index == 1


def test_parallel_render_survives_bad_pdf():
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000))

    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap

    good_pdf = MagicMock()
    good_pdf.__len__ = MagicMock(return_value=2)
    good_pdf.__getitem__ = MagicMock(return_value=mock_page)

    call_count = [0]
    def pdf_factory(path):
        call_count[0] += 1
        if "bad" in str(path):
            raise RuntimeError("corrupt PDF")
        return good_pdf

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)

    with patch("pdf_ocr.pdf_input.pdfium.PdfDocument", side_effect=pdf_factory):
        items = [(Path("/fake/good.pdf"), "good"), (Path("/fake/bad.pdf"), "bad")]
        pages = list(parallel_render(items, cfg, completed_pages={}, num_workers=2))

    assert len(pages) == 2
    assert all(p.doc_id == "good" for p in pages)


def test_parallel_render_empty_input():
    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)
    pages = list(parallel_render([], cfg, completed_pages={}, num_workers=2))
    assert pages == []


@patch("pdf_ocr.pdf_input.snapshot_download")
@patch("pdf_ocr.pdf_input.parallel_render")
def test_load_hf_repo_uses_snapshot_download(mock_parallel, mock_snapshot, tmp_path):
    mock_snapshot.return_value = str(tmp_path)
    (tmp_path / "doc1.pdf").write_bytes(b"fake")
    (tmp_path / "doc2.pdf").write_bytes(b"fake")

    mock_parallel.return_value = iter([
        PageImage(doc_id="doc1", source="doc1.pdf", page_index=0,
                  image=Image.new("RGB", (100, 100))),
    ])

    pages = list(_load_hf_repo_files("user/repo", PdfRenderingConfig(), token="tok"))

    mock_snapshot.assert_called_once_with(
        "user/repo", allow_patterns="*.pdf", repo_type="dataset", token="tok",
    )
    mock_parallel.assert_called_once()
    assert len(pages) == 1


@patch("pdf_ocr.pdf_input.snapshot_download")
@patch("pdf_ocr.pdf_input.parallel_render")
def test_load_pdfs_passes_completed_pages_to_hf(mock_parallel, mock_snapshot, tmp_path):
    mock_snapshot.return_value = str(tmp_path)
    (tmp_path / "doc.pdf").write_bytes(b"fake")
    mock_parallel.return_value = iter([])

    config = ModelConfig(model_id="test/m", served_model_name="m")
    completed = {("doc", 0)}
    list(load_pdfs("user/repo", config, token="tok", completed_pages=completed))

    call_kwargs = mock_parallel.call_args
    assert call_kwargs[1].get("completed_pages") is not None or len(call_kwargs[0]) >= 3


@patch("pdf_ocr.pdf_input.parallel_render")
def test_load_pdfs_passes_completed_pages_to_directory(mock_parallel, tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"fake")
    mock_parallel.return_value = iter([])

    config = ModelConfig(model_id="test/m", served_model_name="m")
    completed = {("doc", 0)}
    list(load_pdfs(str(tmp_path), config, completed_pages=completed))

    mock_parallel.assert_called_once()
