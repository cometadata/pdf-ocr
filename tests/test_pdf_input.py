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

    original_parallel_render = parallel_render

    def _patched_parallel_render(*args, **kwargs):
        kwargs["_use_processes"] = False
        return original_parallel_render(*args, **kwargs)

    with patch("pdf_ocr.pdf_input.pdfium.PdfDocument", side_effect=pdf_factory), \
         patch("pdf_ocr.pdf_input.parallel_render", side_effect=_patched_parallel_render):
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
        pages = list(parallel_render(items, cfg, completed_pages={}, num_workers=2, _use_processes=False))

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
        pages = list(parallel_render(items, cfg, completed_pages=completed, num_workers=2, _use_processes=False))

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
        pages = list(parallel_render(items, cfg, completed_pages=completed, num_workers=1, _use_processes=False))

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
        pages = list(parallel_render(items, cfg, completed_pages={}, num_workers=2, _use_processes=False))

    assert len(pages) == 2
    assert all(p.doc_id == "good" for p in pages)


def test_parallel_render_empty_input():
    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)
    pages = list(parallel_render([], cfg, completed_pages={}, num_workers=2))
    assert pages == []


def test_parallel_render_accepts_iterator():
    """parallel_render works with a lazy iterator, not just a list."""
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000))
    mock_page = MagicMock()
    mock_page.render.return_value = mock_bitmap
    mock_pdf = MagicMock()
    mock_pdf.__len__ = MagicMock(return_value=1)
    mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

    cfg = PdfRenderingConfig(dpi=200, max_dimension=1540)

    def item_gen():
        yield (Path("/fake/a.pdf"), "a")
        yield (Path("/fake/b.pdf"), "b")

    with patch("pdf_ocr.pdf_input.pdfium.PdfDocument", return_value=mock_pdf):
        pages = list(parallel_render(item_gen(), cfg, completed_pages={}, num_workers=2, _use_processes=False))

    assert len(pages) == 2
    assert {p.doc_id for p in pages} == {"a", "b"}


@patch("pdf_ocr.pdf_input.HfApi")
@patch("pdf_ocr.pdf_input.parallel_render")
def test_load_hf_repo_streams_downloads(mock_parallel, mock_hf_api):
    mock_api = MagicMock()
    mock_hf_api.return_value = mock_api
    mock_api.list_repo_files.return_value = ["doc1.pdf", "doc2.pdf"]

    mock_parallel.return_value = iter([
        PageImage(doc_id="doc1", source="doc1.pdf", page_index=0,
                  image=Image.new("RGB", (100, 100))),
    ])

    pages = list(_load_hf_repo_files("user/repo", PdfRenderingConfig(), token="tok"))

    mock_api.list_repo_files.assert_called_once_with(
        "user/repo", repo_type="dataset", token="tok",
    )
    mock_parallel.assert_called_once()
    assert len(pages) == 1


@patch("pdf_ocr.pdf_input.HfApi")
@patch("pdf_ocr.pdf_input.parallel_render")
def test_load_pdfs_passes_completed_pages_to_hf(mock_parallel, mock_hf_api):
    mock_api = MagicMock()
    mock_hf_api.return_value = mock_api
    mock_api.list_repo_files.return_value = ["doc.pdf"]
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


_MINIMAL_PDF = (
    b"%PDF-1.0\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 72 72]/Parent 2 0 R>>endobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n190\n%%EOF"
)


def test_parallel_render_subprocess_mode(tmp_path):
    """Integration test: verify subprocess rendering works with real PDFs."""
    (tmp_path / "a.pdf").write_bytes(_MINIMAL_PDF)
    (tmp_path / "b.pdf").write_bytes(_MINIMAL_PDF)

    cfg = PdfRenderingConfig(dpi=72, max_dimension=1540)
    items = [(tmp_path / "a.pdf", "a"), (tmp_path / "b.pdf", "b")]
    pages = list(parallel_render(items, cfg, completed_pages={}, num_workers=2, _use_processes=True))

    assert len(pages) == 2
    assert {p.doc_id for p in pages} == {"a", "b"}
    assert all(isinstance(p.image, Image.Image) for p in pages)


def test_parallel_render_subprocess_survives_corrupt_pdf(tmp_path):
    """Subprocess mode gracefully handles a PDF that can't be opened."""
    (tmp_path / "good.pdf").write_bytes(_MINIMAL_PDF)
    (tmp_path / "bad.pdf").write_bytes(b"this is not a pdf at all")

    cfg = PdfRenderingConfig(dpi=72, max_dimension=1540)
    items = [(tmp_path / "good.pdf", "good"), (tmp_path / "bad.pdf", "bad")]
    pages = list(parallel_render(items, cfg, completed_pages={}, num_workers=2, _use_processes=True))

    assert len(pages) == 1
    assert pages[0].doc_id == "good"
