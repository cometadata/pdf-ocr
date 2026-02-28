import pytest
from pathlib import Path

from pdf_ocr.storage import (
    save_local,
    results_to_dataset,
    save_batch_checkpoint,
    load_checkpoints,
    clear_checkpoints,
    CHECKPOINT_DIR_NAME,
)
from pdf_ocr.convert import ConversionResult, PageResult


def _make_results():
    return [
        ConversionResult(
            doc_id="paper1",
            source="paper1.pdf",
            pages=[
                PageResult(page_index=0, markdown="# Title\n\nFirst page."),
                PageResult(page_index=1, markdown="Second page content."),
            ],
        ),
        ConversionResult(
            doc_id="subdir/paper2",
            source="subdir/paper2.pdf",
            pages=[
                PageResult(page_index=0, markdown="Another paper."),
            ],
        ),
    ]


def test_save_local_creates_markdown_files(tmp_path):
    results = _make_results()
    save_local(results, tmp_path)

    paper1 = tmp_path / "paper1.md"
    assert paper1.exists()
    content = paper1.read_text()
    assert "# Title" in content
    assert "<!-- page 1 -->" in content
    assert "Second page content." in content

    paper2 = tmp_path / "subdir" / "paper2.md"
    assert paper2.exists()
    assert "Another paper." in paper2.read_text()


def test_save_local_single_page_no_separator(tmp_path):
    results = [
        ConversionResult(
            doc_id="single",
            source="single.pdf",
            pages=[PageResult(page_index=0, markdown="Only page.")],
        )
    ]
    save_local(results, tmp_path)
    content = (tmp_path / "single.md").read_text()
    assert content.strip() == "Only page."
    assert "<!-- page" not in content


def test_results_to_dataset():
    results = _make_results()
    ds = results_to_dataset(results)

    assert len(ds) == 3  # 2 pages + 1 page
    assert set(ds.column_names) == {"doc_id", "source", "page_index", "markdown"}
    assert ds[0]["doc_id"] == "paper1"
    assert ds[0]["page_index"] == 0
    assert ds[1]["page_index"] == 1
    assert ds[2]["doc_id"] == "subdir/paper2"


def test_save_batch_checkpoint(tmp_path):
    batch_results = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="# Page 0")),
        ("doc1", "doc1.pdf", PageResult(page_index=1, markdown="# Page 1")),
    ]
    save_batch_checkpoint(batch_results, tmp_path, batch_index=0)

    checkpoint_dir = tmp_path / CHECKPOINT_DIR_NAME
    assert checkpoint_dir.is_dir()
    files = list(checkpoint_dir.glob("batch_*.json"))
    assert len(files) == 1
    assert files[0].name == "batch_00000.json"


def test_load_checkpoints(tmp_path):
    batch_results = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="# Page 0")),
        ("doc1", "doc1.pdf", PageResult(page_index=1, markdown="# Page 1")),
    ]
    save_batch_checkpoint(batch_results, tmp_path, batch_index=0)

    batch_results_2 = [
        ("doc2", "doc2.pdf", PageResult(page_index=0, markdown="# Doc2 Page 0")),
    ]
    save_batch_checkpoint(batch_results_2, tmp_path, batch_index=1)

    loaded = load_checkpoints(tmp_path)
    assert len(loaded) == 3
    assert loaded[0] == ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="# Page 0"))
    assert loaded[1] == ("doc1", "doc1.pdf", PageResult(page_index=1, markdown="# Page 1"))
    assert loaded[2] == ("doc2", "doc2.pdf", PageResult(page_index=0, markdown="# Doc2 Page 0"))


def test_load_checkpoints_empty(tmp_path):
    loaded = load_checkpoints(tmp_path)
    assert loaded == []


def test_clear_checkpoints(tmp_path):
    batch_results = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="data")),
    ]
    save_batch_checkpoint(batch_results, tmp_path, batch_index=0)
    assert (tmp_path / CHECKPOINT_DIR_NAME).is_dir()

    clear_checkpoints(tmp_path)
    assert not (tmp_path / CHECKPOINT_DIR_NAME).exists()


def test_clear_checkpoints_noop_when_absent(tmp_path):
    # Should not raise
    clear_checkpoints(tmp_path)
