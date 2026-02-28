import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image

from pdf_ocr import convert
from pdf_ocr.config import load_config


def test_full_pipeline_with_local_output(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "output"

    from pdf_ocr.pdf_input import PageImage
    fake_pages = [
        PageImage(doc_id="test", source=str(pdf_file), page_index=0,
                  image=Image.new("RGB", (100, 100))),
        PageImage(doc_id="test", source=str(pdf_file), page_index=1,
                  image=Image.new("RGB", (100, 100))),
    ]

    with patch("pdf_ocr.pdf_input.load_pdfs", return_value=iter(fake_pages)), \
         patch("pdf_ocr.server.launch_vllm") as mock_launch, \
         patch("pdf_ocr.server.wait_for_server", return_value=True), \
         patch("pdf_ocr.server.shutdown_server"), \
         patch("pdf_ocr.server.VLLMClient") as MockClient:

        instance = MockClient.return_value
        instance.infer_batch.return_value = ["# Page 0\n\nHello world.", "## Page 1\n\nGoodbye."]

        results = convert(
            str(pdf_file),
            model="lighton_ocr_2_1b",
            output=str(output_dir),
        )

    assert len(results) == 1
    assert len(results[0].pages) == 2

    md_file = output_dir / "test.md"
    assert md_file.exists()
    content = md_file.read_text()
    assert "# Page 0" in content
    assert "## Page 1" in content
    assert "<!-- page 1 -->" in content


def test_config_loads_all_bundled():
    config = load_config("lighton_ocr_2_1b")
    assert config.model_id == "lightonai/LightOnOCR-2-1B"
    assert config.vllm_args  # Non-empty
    assert config.inference.max_tokens > 0
    assert config.pdf_rendering.dpi > 0
