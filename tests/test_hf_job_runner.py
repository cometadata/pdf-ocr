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


def test_hf_job_runner_flush_every_env_var():
    env = {
        "INPUT_SOURCE": "user/pdf-dataset",
        "MODEL_CONFIG": "lighton_ocr_2_1b",
        "HF_REPO_ID": "user/results",
        "HF_TOKEN": "hf_fake_token",
        "BACKEND": "offline",
        "FLUSH_EVERY": "2",
    }

    fake_pages = [
        PageImage(doc_id="test", source="test.pdf", page_index=i,
                  image=Image.new("RGB", (100, 100)))
        for i in range(5)
    ]

    batch_num = 0
    def fake_streaming(*args, **kwargs):
        nonlocal batch_num
        for i in range(5):
            batch_num += 1
            yield [("test", "test.pdf", PageResult(page_index=i, markdown=f"# Page {i}"))]

    with patch.dict(os.environ, env, clear=False), \
         patch("pdf_ocr.convert.convert_pages_streaming", side_effect=fake_streaming), \
         patch("pdf_ocr.pdf_input.load_pdfs", return_value=iter(fake_pages)), \
         patch("pdf_ocr.storage.push_batch_to_hub") as mock_push, \
         patch("pdf_ocr.offline.VLLMOfflineEngine"):

        from pdf_ocr.hf_jobs.hf_job_runner import main
        main()

        # flush_every=2: flush at batch 2, 4, then final flush with batch 5
        assert mock_push.call_count == 3


def test_log_memory_does_not_crash():
    from pdf_ocr.hf_jobs.hf_job_runner import _log_memory
    # Should not raise on any platform (silently no-ops on macOS)
    _log_memory()


def test_hf_job_runner_flushes_on_exception():
    env = {
        "INPUT_SOURCE": "user/pdf-dataset",
        "MODEL_CONFIG": "lighton_ocr_2_1b",
        "HF_REPO_ID": "user/results",
        "HF_TOKEN": "hf_fake_token",
        "BACKEND": "offline",
        "FLUSH_EVERY": "100",  # high so no mid-loop flush
    }

    fake_pages = [
        PageImage(doc_id="test", source="test.pdf", page_index=0,
                  image=Image.new("RGB", (100, 100))),
    ]

    def fake_streaming(*args, **kwargs):
        yield [("test", "test.pdf", PageResult(page_index=0, markdown="# Test"))]
        raise RuntimeError("simulated crash")

    with patch.dict(os.environ, env, clear=False), \
         patch("pdf_ocr.convert.convert_pages_streaming", side_effect=fake_streaming), \
         patch("pdf_ocr.pdf_input.load_pdfs", return_value=iter(fake_pages)), \
         patch("pdf_ocr.storage.push_batch_to_hub") as mock_push, \
         patch("pdf_ocr.offline.VLLMOfflineEngine"), \
         pytest.raises(RuntimeError, match="simulated crash"):

        from pdf_ocr.hf_jobs.hf_job_runner import main
        main()

    # The finally block should have flushed the pending row
    mock_push.assert_called_once()
    call_args = mock_push.call_args
    assert len(call_args[0][0]) == 1  # 1 pending row


def test_hf_job_runner_flush_failure_still_shuts_down():
    env = {
        "INPUT_SOURCE": "user/pdf-dataset",
        "MODEL_CONFIG": "lighton_ocr_2_1b",
        "HF_REPO_ID": "user/results",
        "HF_TOKEN": "hf_fake_token",
        "BACKEND": "server",
        "FLUSH_EVERY": "100",
    }

    fake_pages = [
        PageImage(doc_id="test", source="test.pdf", page_index=0,
                  image=Image.new("RGB", (100, 100))),
    ]

    def fake_streaming(*args, **kwargs):
        yield [("test", "test.pdf", PageResult(page_index=0, markdown="# Test"))]
        raise RuntimeError("simulated crash")

    mock_server_process = MagicMock()

    with patch.dict(os.environ, env, clear=False), \
         patch("pdf_ocr.convert.convert_pages_streaming", side_effect=fake_streaming), \
         patch("pdf_ocr.pdf_input.load_pdfs", return_value=iter(fake_pages)), \
         patch("pdf_ocr.storage.push_batch_to_hub", side_effect=RuntimeError("flush failed")), \
         patch("pdf_ocr.server.launch_vllm", return_value=mock_server_process), \
         patch("pdf_ocr.server.wait_for_server", return_value=True), \
         patch("pdf_ocr.server.shutdown_server") as mock_shutdown, \
         patch("pdf_ocr.server.VLLMClient"), \
         pytest.raises(RuntimeError, match="simulated crash"):

        from pdf_ocr.hf_jobs.hf_job_runner import main
        main()

    # Server should still be shut down even though flush failed
    mock_shutdown.assert_called_once_with(mock_server_process)


def test_hf_job_runner_passes_completed_pages_to_load_pdfs():
    env = {
        "INPUT_SOURCE": "user/pdf-dataset",
        "MODEL_CONFIG": "lighton_ocr_2_1b",
        "HF_REPO_ID": "user/results",
        "HF_TOKEN": "hf_fake_token",
        "BACKEND": "offline",
        "FLUSH_EVERY": "100",
    }

    def fake_streaming(pages_iter, *args, **kwargs):
        pages_list = list(pages_iter)
        for page in pages_list:
            yield [("test", "test.pdf", PageResult(page_index=page.page_index, markdown=f"# Page {page.page_index}"))]

    with patch.dict(os.environ, env, clear=False), \
         patch("pdf_ocr.convert.convert_pages_streaming", side_effect=fake_streaming), \
         patch("pdf_ocr.pdf_input.load_pdfs") as mock_load, \
         patch("pdf_ocr.storage.push_batch_to_hub"), \
         patch("pdf_ocr.storage.load_hub_progress", return_value=(2, {("test", 0)})), \
         patch("pdf_ocr.offline.VLLMOfflineEngine"):

        mock_load.return_value = iter([
            PageImage(doc_id="test", source="test.pdf", page_index=1,
                      image=Image.new("RGB", (100, 100))),
        ])

        from pdf_ocr.hf_jobs.hf_job_runner import main
        main()

        # Verify completed_pages was passed to load_pdfs
        call_kwargs = mock_load.call_args
        assert "completed_pages" in call_kwargs[1] or len(call_kwargs[0]) > 2
        completed = call_kwargs[1].get("completed_pages") or call_kwargs[0][2]
        assert ("test", 0) in completed
