"""Output handling: local files and HuggingFace Hub."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from .convert import ConversionResult, PageResult

if TYPE_CHECKING:
    from datasets import Dataset

LOGGER = logging.getLogger(__name__)

CHECKPOINT_DIR_NAME = ".checkpoints"

PAGE_SEPARATOR = "\n\n<!-- page {n} -->\n\n"


def save_local(results: List[ConversionResult], output_dir: str | Path) -> None:
    """Save conversion results as local markdown files."""
    output_dir = Path(output_dir)
    for result in results:
        md_path = output_dir / f"{result.doc_id}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)

        parts = []
        for page in result.pages:
            if parts:
                parts.append(PAGE_SEPARATOR.format(n=page.page_index))
            parts.append(page.markdown)

        md_path.write_text("".join(parts), encoding="utf-8")
        LOGGER.info("Saved %s (%d pages)", md_path, len(result.pages))


def results_to_dataset(results: List[ConversionResult]) -> "Dataset":
    """Convert results to a HuggingFace Dataset."""
    from datasets import Dataset

    rows = []
    for result in results:
        for page in result.pages:
            rows.append({
                "doc_id": result.doc_id,
                "source": result.source,
                "page_index": page.page_index,
                "markdown": page.markdown,
            })
    return Dataset.from_list(rows)


def push_to_hub(
    results: List[ConversionResult],
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> None:
    """Push conversion results to HuggingFace Hub as a dataset."""
    from huggingface_hub import HfApi

    ds = results_to_dataset(results)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    ds.push_to_hub(
        repo_id,
        token=token,
        private=private,
        commit_message=commit_message or "Add OCR markdown results",
    )
    LOGGER.info("Pushed %d rows to %s (private=%s)", len(ds), repo_id, private)


def save_batch_checkpoint(
    results: List[Tuple[str, str, PageResult]],
    output_dir: Path,
    batch_index: int,
) -> None:
    """Write a single batch of results as a checkpoint file."""
    checkpoint_dir = output_dir / CHECKPOINT_DIR_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "doc_id": doc_id,
            "source": source,
            "page_index": page_result.page_index,
            "markdown": page_result.markdown,
        }
        for doc_id, source, page_result in results
    ]

    checkpoint_file = checkpoint_dir / f"batch_{batch_index:05d}.json"
    checkpoint_file.write_text(json.dumps(rows), encoding="utf-8")
    LOGGER.debug("Saved checkpoint %s (%d pages)", checkpoint_file.name, len(rows))


def load_checkpoints(output_dir: Path) -> List[Tuple[str, str, PageResult]]:
    """Load all checkpoint files and return flat list of (doc_id, source, PageResult)."""
    checkpoint_dir = output_dir / CHECKPOINT_DIR_NAME
    if not checkpoint_dir.is_dir():
        return []

    results: List[Tuple[str, str, PageResult]] = []
    for path in sorted(checkpoint_dir.glob("batch_*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            results.append((
                row["doc_id"],
                row["source"],
                PageResult(page_index=row["page_index"], markdown=row["markdown"]),
            ))

    LOGGER.info("Loaded %d pages from %s checkpoints", len(results), len(list(checkpoint_dir.glob("batch_*.json"))))
    return results


def clear_checkpoints(output_dir: Path) -> None:
    """Remove the checkpoints directory after successful completion."""
    checkpoint_dir = output_dir / CHECKPOINT_DIR_NAME
    if checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)
        LOGGER.info("Cleared checkpoints directory")
