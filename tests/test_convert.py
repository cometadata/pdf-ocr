import threading

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from pdf_ocr.convert import (
    PrefetchIterator,
    _batch_pages, _group_by_document, _infer_with_retry,
    convert_pages, convert_pages_streaming,
)
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


def test_infer_with_retry_success_first_try():
    batch = [_make_page("doc1", 0), _make_page("doc1", 1)]
    mock_client = MagicMock()
    mock_client.infer_batch.return_value = ["# Page 0", "# Page 1"]

    results = _infer_with_retry(mock_client, batch, max_depth=3)

    assert results == ["# Page 0", "# Page 1"]
    mock_client.infer_batch.assert_called_once()


def test_infer_with_retry_subdivides_on_failure():
    batch = [_make_page("doc1", i) for i in range(4)]
    mock_client = MagicMock()
    mock_client.infer_batch.side_effect = [
        RuntimeError("OOM"),
        ["md0", "md1"],
        ["md2", "md3"],
    ]

    results = _infer_with_retry(mock_client, batch, max_depth=3)

    assert results == ["md0", "md1", "md2", "md3"]
    assert mock_client.infer_batch.call_count == 3


def test_infer_with_retry_empty_at_max_depth():
    batch = [_make_page("doc1", 0)]
    mock_client = MagicMock()
    mock_client.infer_batch.side_effect = RuntimeError("always fails")

    results = _infer_with_retry(mock_client, batch, max_depth=0)

    assert results == [""]


def test_infer_with_retry_isolates_single_bad_page():
    batch = [_make_page("doc1", i) for i in range(4)]
    mock_client = MagicMock()

    call_count = [0]
    def side_effect(images):
        call_count[0] += 1
        if len(images) == 4:
            raise RuntimeError("OOM")
        if len(images) == 2 and images[1] is batch[2].image:
            raise RuntimeError("Bad page in sub-batch")
        if len(images) == 2 and images[0] is batch[2].image:
            raise RuntimeError("Bad page in sub-batch")
        if len(images) == 1 and images[0] is batch[2].image:
            raise RuntimeError("Single bad page")
        return [f"md{i}" for i in range(len(images))]

    mock_client.infer_batch.side_effect = side_effect

    results = _infer_with_retry(mock_client, batch, max_depth=3)

    assert len(results) == 4
    assert results[2] == ""
    assert all(r != "" for i, r in enumerate(results) if i != 2)


def test_convert_pages_streaming_yields_batches():
    pages = [_make_page("doc1", i) for i in range(5)]

    mock_client = MagicMock()
    mock_client.infer_batch.side_effect = [
        ["md0", "md1"],
        ["md2", "md3"],
        ["md4"],
    ]

    batches = list(convert_pages_streaming(iter(pages), mock_client, batch_size=2))

    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1
    assert batches[0][0][2].markdown == "md0"
    assert batches[2][0][2].markdown == "md4"


def test_convert_pages_streaming_checkpoint_compat(tmp_path):
    pages = [_make_page("doc1", i) for i in range(3)]

    mock_client = MagicMock()
    mock_client.infer_batch.side_effect = [
        ["md0", "md1"],
        ["md2"],
    ]

    batches = list(convert_pages_streaming(
        iter(pages), mock_client, batch_size=2, checkpoint_dir=tmp_path
    ))

    checkpoint_dir = tmp_path / ".checkpoints"
    assert checkpoint_dir.is_dir()
    checkpoint_files = sorted(checkpoint_dir.glob("batch_*.json"))
    assert len(checkpoint_files) == 2


def test_prefetch_iterator_yields_all_pages():
    pages = [_make_page("doc1", i) for i in range(6)]
    stop = threading.Event()
    prefetch = PrefetchIterator(iter(pages), queue_size=4, stop_event=stop)

    result = list(prefetch)
    stop.set()

    assert len(result) == 6
    assert [p.page_index for p in result] == [0, 1, 2, 3, 4, 5]
    assert all(p.doc_id == "doc1" for p in result)


def test_prefetch_iterator_empty_source():
    stop = threading.Event()
    prefetch = PrefetchIterator(iter([]), queue_size=4, stop_event=stop)

    result = list(prefetch)
    stop.set()

    assert result == []


def test_prefetch_iterator_stop_event():
    def infinite_source():
        i = 0
        while True:
            yield _make_page("doc1", i)
            i += 1

    stop = threading.Event()
    prefetch = PrefetchIterator(infinite_source(), queue_size=2, stop_event=stop)

    first = next(prefetch)
    second = next(prefetch)
    assert first.page_index == 0
    assert second.page_index == 1

    stop.set()

    remaining = list(prefetch)
    assert isinstance(remaining, list)


def test_prefetch_iterator_producer_runs_ahead():
    produced = threading.Event()

    def source():
        for i in range(4):
            yield _make_page("doc1", i)
        produced.set()

    stop = threading.Event()
    prefetch = PrefetchIterator(source(), queue_size=8, stop_event=stop)

    produced.wait(timeout=2.0)
    assert produced.is_set(), "Producer should have finished before consumer started"

    result = list(prefetch)
    stop.set()
    assert len(result) == 4


def test_prefetch_iterator_backpressure():
    pages = [_make_page("doc1", i) for i in range(10)]
    stop = threading.Event()
    prefetch = PrefetchIterator(iter(pages), queue_size=2, stop_event=stop)

    result = list(prefetch)
    stop.set()

    assert len(result) == 10
    assert [p.page_index for p in result] == list(range(10))
