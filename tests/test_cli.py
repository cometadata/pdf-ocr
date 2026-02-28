import pytest
from unittest.mock import patch, MagicMock
from pdf_ocr.cli import build_parser


def test_parser_required_source():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["test.pdf"])
    assert args.source == "test.pdf"
    assert args.model == "lighton_ocr_2_1b"
    assert args.output is None
    assert args.hf_repo is None
    assert args.private is False
    assert args.base_url is None
    assert args.backend is None
    assert args.max_pages is None
    assert args.batch_size is None


def test_parser_all_options():
    parser = build_parser()
    args = parser.parse_args([
        "user/dataset",
        "--model", "custom.yaml",
        "--output", "./out",
        "--hf-repo", "user/results",
        "--private",
        "--base-url", "http://localhost:8000",
        "--max-pages", "10",
        "--batch-size", "8",
        "--pdf-column", "content",
    ])
    assert args.source == "user/dataset"
    assert args.model == "custom.yaml"
    assert args.output == "./out"
    assert args.hf_repo == "user/results"
    assert args.private is True
    assert args.base_url == "http://localhost:8000"
    assert args.max_pages == 10
    assert args.batch_size == 8
    assert args.pdf_column == "content"


def test_parser_backend_offline():
    parser = build_parser()
    args = parser.parse_args(["test.pdf", "--backend", "offline"])
    assert args.backend == "offline"


def test_parser_backend_server():
    parser = build_parser()
    args = parser.parse_args(["test.pdf", "--backend", "server"])
    assert args.backend == "server"


def test_parser_backend_invalid():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["test.pdf", "--backend", "invalid"])
