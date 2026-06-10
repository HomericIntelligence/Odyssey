"""Tests for scripts/check_adr_review_dates.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
import check_adr_review_dates as mod


def test_pattern_matches_canonical_format() -> None:
    """Test that the regex matches the canonical **Next Review** format."""
    text = "**Next Review**: 2026-09-03 (quarterly cadence)"
    assert mod.PATTERN.search(text).group(1) == "2026-09-03"


def test_pattern_does_not_match_last_reviewed() -> None:
    """Test that the regex does not match **Last Reviewed**."""
    text = "**Last Reviewed**: 2026-06-03"
    assert mod.PATTERN.search(text) is None


def test_main_passes_when_all_dates_future(tmp_path, monkeypatch, capsys) -> None:
    """Test that main() returns 0 when all ADR review dates are in the future."""
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)
    (adr_dir / "ADR-008-test.md").write_text("**Next Review**: 2099-01-01\n")
    monkeypatch.chdir(tmp_path)
    assert mod.main() == 0
    assert "✅" in capsys.readouterr().out


def test_main_fails_when_a_date_is_past(tmp_path, monkeypatch, capsys) -> None:
    """Test that main() returns 1 when any ADR review date is in the past."""
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)
    (adr_dir / "ADR-008-test.md").write_text("**Next Review**: 2020-01-01\n")
    monkeypatch.chdir(tmp_path)
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "overdue" in out.lower()
