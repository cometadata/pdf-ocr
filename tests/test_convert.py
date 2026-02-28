import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from pdf_ocr.convert import _batch_pages, _group_by_document, convert_pages
from pdf_ocr.pdf_input import PageImage
from pdf_ocr.config import ModelConfig, InferenceConfig


def _make_page(doc_id: str, page_index: int) -> PageImage:
    return PageImage(
        doc_id=doc_id,
        source=f"{doc_id}.pdf",
        page_index=page_index,
        image=Image.new("RGB", (100, 100)),
    )


def test_batch_pages():
    pages = [_make_page("doc1", i) for i in range(7)]
    batches = list(_batch_pages(iter(pages), batch_size=3))
    assert len(batches) == 3
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 1


def test_group_by_document():
    from pdf_ocr.convert import PageResult
    results = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="page 0")),
        ("doc1", "doc1.pdf", PageResult(page_index=1, markdown="page 1")),
        ("doc2", "doc2.pdf", PageResult(page_index=0, markdown="hello")),
    ]
    grouped = _group_by_document(results)
    assert len(grouped) == 2
    assert grouped[0].doc_id == "doc1"
    assert len(grouped[0].pages) == 2
    assert grouped[1].doc_id == "doc2"
    assert len(grouped[1].pages) == 1


def test_convert_pages_calls_client():
    pages = [_make_page("doc1", 0), _make_page("doc1", 1)]

    mock_client = MagicMock()
    mock_client.infer_batch.return_value = ["# Page 0 markdown", "# Page 1 markdown"]

    results = convert_pages(iter(pages), mock_client, batch_size=4)

    assert len(results) == 1
    assert results[0].doc_id == "doc1"
    assert len(results[0].pages) == 2
    assert results[0].pages[0].markdown == "# Page 0 markdown"
    mock_client.infer_batch.assert_called_once()


def test_convert_pages_writes_checkpoints(tmp_path):
    pages = [_make_page("doc1", 0), _make_page("doc1", 1), _make_page("doc1", 2)]

    mock_client = MagicMock()
    mock_client.infer_batch.side_effect = [
        ["# Page 0", "# Page 1"],
        ["# Page 2"],
    ]

    results = convert_pages(
        iter(pages), mock_client, batch_size=2, checkpoint_dir=tmp_path
    )

    assert len(results) == 1
    assert len(results[0].pages) == 3

    checkpoint_dir = tmp_path / ".checkpoints"
    assert checkpoint_dir.is_dir()
    checkpoint_files = sorted(checkpoint_dir.glob("batch_*.json"))
    assert len(checkpoint_files) == 2


def test_convert_pages_resumes_from_checkpoint(tmp_path):
    from pdf_ocr.storage import save_batch_checkpoint
    from pdf_ocr.convert import PageResult

    prior_results = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="# Page 0")),
        ("doc1", "doc1.pdf", PageResult(page_index=1, markdown="# Page 1")),
    ]
    save_batch_checkpoint(prior_results, tmp_path, batch_index=0)

    pages = [
        _make_page("doc1", 0),
        _make_page("doc1", 1),
        _make_page("doc1", 2),
        _make_page("doc1", 3),
    ]

    mock_client = MagicMock()
    mock_client.infer_batch.return_value = ["# Page 2", "# Page 3"]

    results = convert_pages(
        iter(pages),
        mock_client,
        batch_size=4,
        checkpoint_dir=tmp_path,
        resume_from_checkpoint=True,
    )

    assert len(results) == 1
    assert len(results[0].pages) == 4
    assert results[0].pages[0].markdown == "# Page 0"
    assert results[0].pages[2].markdown == "# Page 2"
    mock_client.infer_batch.assert_called_once()
