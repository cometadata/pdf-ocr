"""Output handling: local files and HuggingFace Hub."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from .convert import ConversionResult

if TYPE_CHECKING:
    from datasets import Dataset

LOGGER = logging.getLogger(__name__)

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
