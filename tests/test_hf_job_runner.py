import pytest
from unittest.mock import patch, MagicMock
import os
from PIL import Image

from pdf_ocr.pdf_input import PageImage
from pdf_ocr.convert import PageResult


def test_hf_job_runner_reads_env_vars():
    env = {
        "INPUT_SOURCE": "user/pdf-dataset",
        "MODEL_CONFIG": "lighton_ocr_2_1b",
        "HF_REPO_ID": "user/results",
        "HF_TOKEN": "hf_fake_token",
        "PRIVATE": "true",
        "BATCH_SIZE": "16",
        "MAX_PAGES": "100",
        "BACKEND": "offline",
    }

    fake_pages = [
        PageImage(doc_id="test", source="test.pdf", page_index=0,
                  image=Image.new("RGB", (100, 100))),
    ]

    def fake_streaming(*args, **kwargs):
        yield [("test", "test.pdf", PageResult(page_index=0, markdown="# Test"))]

    with patch.dict(os.environ, env, clear=False), \
         patch("pdf_ocr.convert.convert_pages_streaming", side_effect=fake_streaming) as mock_streaming, \
         patch("pdf_ocr.pdf_input.load_pdfs", return_value=iter(fake_pages)), \
         patch("pdf_ocr.storage.push_batch_to_hub") as mock_push, \
         patch("pdf_ocr.offline.VLLMOfflineEngine") as MockOffline:

        from pdf_ocr.hf_jobs.hf_job_runner import main
        main()

        mock_streaming.assert_called_once()
        call_kwargs = mock_streaming.call_args
        assert call_kwargs[1]["batch_size"] == 16
        assert call_kwargs[1]["max_pages"] == 100
        mock_push.assert_called_once()
