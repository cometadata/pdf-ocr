import pytest
from unittest.mock import patch, MagicMock
import os


def test_hf_job_runner_reads_env_vars():
    env = {
        "INPUT_SOURCE": "user/pdf-dataset",
        "MODEL_CONFIG": "lighton_ocr_2_1b",
        "HF_REPO_ID": "user/results",
        "HF_TOKEN": "hf_fake_token",
        "PRIVATE": "true",
        "BATCH_SIZE": "16",
        "MAX_PAGES": "100",
    }

    with patch.dict(os.environ, env, clear=False), \
         patch("pdf_ocr.convert") as mock_convert, \
         patch("pdf_ocr.storage.push_to_hub") as mock_push:

        from pdf_ocr.convert import ConversionResult, PageResult
        mock_convert.return_value = [
            ConversionResult(doc_id="test", source="test.pdf",
                             pages=[PageResult(page_index=0, markdown="# Test")])
        ]

        from pdf_ocr.hf_jobs.hf_job_runner import main
        main()

        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args
        assert call_kwargs[1]["source"] == "user/pdf-dataset"
        assert call_kwargs[1]["model"] == "lighton_ocr_2_1b"
        assert call_kwargs[1]["batch_size"] == 16
        assert call_kwargs[1]["max_pages"] == 100
