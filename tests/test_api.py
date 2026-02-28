import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from pdf_ocr import convert
from pdf_ocr.pdf_input import PageImage


def test_convert_single_pdf(tmp_path):
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    fake_pages = [
        PageImage(doc_id="test", source=str(fake_pdf), page_index=0,
                  image=Image.new("RGB", (100, 100))),
    ]

    from pdf_ocr.convert import PageResult

    def fake_streaming(*args, **kwargs):
        yield [("test", str(fake_pdf), PageResult(page_index=0, markdown="# Hello"))]

    with patch("pdf_ocr.convert.convert_pages_streaming", side_effect=fake_streaming), \
         patch("pdf_ocr.pdf_input.load_pdfs", return_value=iter(fake_pages)), \
         patch("pdf_ocr.server.launch_vllm") as mock_launch, \
         patch("pdf_ocr.server.wait_for_server", return_value=True), \
         patch("pdf_ocr.server.shutdown_server"), \
         patch("pdf_ocr.server.VLLMClient") as MockClient:

        results = convert(str(fake_pdf), model="lighton_ocr_2_1b")

        assert len(results) == 1
        assert results[0].doc_id == "test"
        assert results[0].pages[0].markdown == "# Hello"


def test_convert_with_base_url_skips_server_launch(tmp_path):
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    fake_pages = [
        PageImage(doc_id="test", source=str(fake_pdf), page_index=0,
                  image=Image.new("RGB", (100, 100))),
    ]

    from pdf_ocr.convert import PageResult

    def fake_streaming(*args, **kwargs):
        yield [("test", str(fake_pdf), PageResult(page_index=0, markdown="# Hello"))]

    with patch("pdf_ocr.convert.convert_pages_streaming", side_effect=fake_streaming), \
         patch("pdf_ocr.pdf_input.load_pdfs", return_value=iter(fake_pages)), \
         patch("pdf_ocr.server.launch_vllm") as mock_launch, \
         patch("pdf_ocr.server.VLLMClient") as MockClient:

        results = convert(str(fake_pdf), model="lighton_ocr_2_1b",
                          base_url="http://localhost:8000")

        mock_launch.assert_not_called()


def test_convert_offline_backend_creates_offline_engine(tmp_path):
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    fake_pages = [
        PageImage(doc_id="test", source=str(fake_pdf), page_index=0,
                  image=Image.new("RGB", (100, 100))),
    ]

    from pdf_ocr.convert import PageResult

    def fake_streaming(*args, **kwargs):
        yield [("test", str(fake_pdf), PageResult(page_index=0, markdown="# Hello"))]

    with patch("pdf_ocr.convert.convert_pages_streaming", side_effect=fake_streaming), \
         patch("pdf_ocr.pdf_input.load_pdfs", return_value=iter(fake_pages)), \
         patch("pdf_ocr.offline.VLLMOfflineEngine") as MockOffline:

        results = convert(str(fake_pdf), model="lighton_ocr_2_1b",
                          backend="offline")

        MockOffline.assert_called_once()
        assert len(results) == 1
