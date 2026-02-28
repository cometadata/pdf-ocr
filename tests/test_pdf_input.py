import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from pdf_ocr.pdf_input import render_page, render_pdf, detect_input_type, InputType
from pdf_ocr.config import PdfRenderingConfig


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
