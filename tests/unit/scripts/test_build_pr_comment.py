"""Unit tests for scripts/build_pr_comment.py."""

import sys
from pathlib import Path


# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
from build_pr_comment import FOOTER, HEADER, build_comment


def test_happy_path(tmp_path: Path) -> None:
    """Valid metrics file produces output with header, content, and footer."""
    metrics = tmp_path / "metrics.md"
    metrics.write_text("| Tests | 42 |\n", encoding="utf-8")

    output = tmp_path / "pr_comment.md"
    result = build_comment(metrics, output)

    assert result == 0
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert content.startswith(HEADER)
    assert "| Tests | 42 |" in content
    assert content.endswith(FOOTER)


def test_missing_metrics_file(tmp_path: Path) -> None:
    """Missing metrics file returns non-zero exit code."""
    metrics = tmp_path / "nonexistent.md"
    output = tmp_path / "pr_comment.md"

    result = build_comment(metrics, output)

    assert result == 1
    assert not output.exists()


def test_empty_metrics_file(tmp_path: Path) -> None:
    """Empty metrics file still produces header and footer."""
    metrics = tmp_path / "metrics.md"
    metrics.write_text("", encoding="utf-8")

    output = tmp_path / "pr_comment.md"
    result = build_comment(metrics, output)

    assert result == 0
    content = output.read_text(encoding="utf-8")
    assert HEADER in content
    assert FOOTER in content


def test_output_contains_no_emoji_byte_escapes(tmp_path: Path) -> None:
    """Output header uses plain ASCII with no byte-escaped emoji."""
    metrics = tmp_path / "metrics.md"
    metrics.write_text("data\n", encoding="utf-8")

    output = tmp_path / "pr_comment.md"
    build_comment(metrics, output)

    raw = output.read_bytes()
    # Ensure no UTF-8 byte-escape sequences for emoji (\xf0\x9f...) appear
    assert b"\xf0\x9f" not in raw
