import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pdf_ocr.storage import (
    save_local,
    results_to_dataset,
    save_batch_checkpoint,
    save_batch_incremental,
    push_batch_to_hub,
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


# --- save_batch_incremental tests ---


def test_save_batch_incremental_creates_files(tmp_path):
    batch = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="# Page 0")),
        ("doc1", "doc1.pdf", PageResult(page_index=1, markdown="Page 1 content")),
    ]
    save_batch_incremental(batch, tmp_path)

    md_file = tmp_path / "doc1.md"
    assert md_file.exists()
    content = md_file.read_text()
    assert "# Page 0" in content
    assert "Page 1 content" in content
    assert "<!-- page 1 -->" in content


def test_save_batch_incremental_appends_across_batches(tmp_path):
    batch1 = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="First page")),
    ]
    batch2 = [
        ("doc1", "doc1.pdf", PageResult(page_index=1, markdown="Second page")),
    ]

    save_batch_incremental(batch1, tmp_path)
    save_batch_incremental(batch2, tmp_path)

    content = (tmp_path / "doc1.md").read_text()
    assert "First page" in content
    assert "Second page" in content
    assert "<!-- page 1 -->" in content


def test_save_batch_incremental_multiple_docs(tmp_path):
    batch = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="Doc 1")),
        ("doc2", "doc2.pdf", PageResult(page_index=0, markdown="Doc 2")),
    ]
    save_batch_incremental(batch, tmp_path)

    assert (tmp_path / "doc1.md").exists()
    assert (tmp_path / "doc2.md").exists()
    assert "Doc 1" in (tmp_path / "doc1.md").read_text()
    assert "Doc 2" in (tmp_path / "doc2.md").read_text()


def test_save_batch_incremental_subdirectory_doc(tmp_path):
    batch = [
        ("subdir/doc1", "subdir/doc1.pdf", PageResult(page_index=0, markdown="Nested")),
    ]
    save_batch_incremental(batch, tmp_path)

    md_file = tmp_path / "subdir" / "doc1.md"
    assert md_file.exists()
    assert "Nested" in md_file.read_text()


# --- push_batch_to_hub tests ---


def test_push_batch_to_hub_uploads_shard():
    batch = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="# Hello")),
        ("doc1", "doc1.pdf", PageResult(page_index=1, markdown="World")),
    ]

    mock_api_instance = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=mock_api_instance):
        push_batch_to_hub(batch, repo_id="user/results", shard_index=0, token="fake")

    mock_api_instance.create_repo.assert_called_once_with(
        "user/results", repo_type="dataset", private=False, exist_ok=True
    )
    mock_api_instance.upload_file.assert_called_once()
    call_kwargs = mock_api_instance.upload_file.call_args[1]
    assert call_kwargs["path_in_repo"] == "data/shard_00000.parquet"
    assert call_kwargs["repo_id"] == "user/results"
    assert call_kwargs["repo_type"] == "dataset"


def test_push_batch_to_hub_shard_numbering():
    batch = [
        ("doc1", "doc1.pdf", PageResult(page_index=0, markdown="data")),
    ]

    mock_api_instance = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=mock_api_instance):
        push_batch_to_hub(batch, repo_id="user/results", shard_index=42)

    call_kwargs = mock_api_instance.upload_file.call_args[1]
    assert call_kwargs["path_in_repo"] == "data/shard_00042.parquet"
