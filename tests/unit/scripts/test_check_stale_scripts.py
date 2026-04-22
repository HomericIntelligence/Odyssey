"""Unit tests for scripts/check_stale_scripts.py."""

import sys
from pathlib import Path

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
from check_stale_scripts import (
    check_stale_scripts,
    find_stale_scripts,
    get_all_scripts,
    get_reference_targets,
)


def _make_scripts_dir(tmp_path: Path, names: list[str]) -> Path:
    """Create a scripts/ directory with empty .py files."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for name in names:
        (scripts_dir / name).write_text(f"# {name}\n", encoding="utf-8")
    return scripts_dir


def _make_repo(tmp_path: Path, scripts: list[str], justfile_content: str = "") -> Path:
    """Set up a minimal fake repo with scripts/ and a justfile."""
    _make_scripts_dir(tmp_path, scripts)
    if justfile_content:
        (tmp_path / "justfile").write_text(justfile_content, encoding="utf-8")
    return tmp_path


class TestGetAllScripts:
    def test_returns_basenames(self, tmp_path: Path) -> None:
        """get_all_scripts returns only *.py basenames."""
        scripts_dir = _make_scripts_dir(tmp_path, ["foo.py", "bar.py"])
        (scripts_dir / "not_python.sh").write_text("", encoding="utf-8")
        result = get_all_scripts(scripts_dir, extensions=(".py",))
        assert result == ["bar.py", "foo.py"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        """Empty scripts dir returns empty list."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        assert get_all_scripts(scripts_dir, extensions=(".py",)) == []

    def test_sorted(self, tmp_path: Path) -> None:
        """Results are alphabetically sorted."""
        scripts_dir = _make_scripts_dir(tmp_path, ["z.py", "a.py", "m.py"])
        assert get_all_scripts(scripts_dir, extensions=(".py",)) == ["a.py", "m.py", "z.py"]


class TestGetReferenceTargets:
    def test_includes_justfile(self, tmp_path: Path) -> None:
        """justfile is included when present."""
        (tmp_path / "justfile").write_text("", encoding="utf-8")
        targets = get_reference_targets(tmp_path)
        names = [t.name for t in targets]
        assert "justfile" in names

    def test_includes_precommit(self, tmp_path: Path) -> None:
        """.pre-commit-config.yaml is included when present."""
        (tmp_path / ".pre-commit-config.yaml").write_text("", encoding="utf-8")
        targets = get_reference_targets(tmp_path)
        names = [t.name for t in targets]
        assert ".pre-commit-config.yaml" in names

    def test_includes_github_workflows(self, tmp_path: Path) -> None:
        """GitHub workflow files are included."""
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text("", encoding="utf-8")
        targets = get_reference_targets(tmp_path)
        names = [t.name for t in targets]
        assert "ci.yml" in names

    def test_missing_files_excluded(self, tmp_path: Path) -> None:
        """Paths that don't exist are not included."""
        targets = get_reference_targets(tmp_path)
        assert targets == []


class TestFindStaleScripts:
    def test_all_referenced_returns_empty(self, tmp_path: Path) -> None:
        """No candidates when every script appears in justfile."""
        scripts = ["foo.py", "bar.py"]
        justfile_content = "python scripts/foo.py\npython scripts/bar.py\n"
        repo = _make_repo(tmp_path, scripts, justfile_content)
        assert find_stale_scripts(repo) == []

    def test_unreferenced_script_flagged(self, tmp_path: Path) -> None:
        """Script with no references is returned as a candidate."""
        scripts = ["active.py", "stale.py"]
        justfile_content = "python scripts/active.py\n"
        repo = _make_repo(tmp_path, scripts, justfile_content)
        result = find_stale_scripts(repo)
        assert "stale.py" in result
        assert "active.py" not in result

    def test_empty_scripts_dir_returns_empty(self, tmp_path: Path) -> None:
        """Empty scripts directory returns no candidates."""
        _make_scripts_dir(tmp_path, [])
        assert find_stale_scripts(tmp_path) == []

    def test_missing_scripts_dir_returns_empty(self, tmp_path: Path) -> None:
        """Missing scripts directory returns no candidates."""
        assert find_stale_scripts(tmp_path) == []

    def test_exclude_pattern_filters_scripts(self, tmp_path: Path) -> None:
        """Scripts matching exclude_pattern are not flagged."""
        _make_scripts_dir(tmp_path, ["download_mnist.py", "stale.py"])
        result = find_stale_scripts(tmp_path, exclude_pattern="download_")
        assert "download_mnist.py" not in result


class TestCheckStaleScripts:
    def test_returns_zero_no_stale(self, tmp_path: Path) -> None:
        """check_stale_scripts() returns 0 when all scripts are referenced."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "referenced.py").write_text("", encoding="utf-8")
        (tmp_path / "justfile").write_text("python scripts/referenced.py\n", encoding="utf-8")
        result = check_stale_scripts(tmp_path)
        assert result == 0

    def test_returns_zero_with_stale(self, tmp_path: Path) -> None:
        """check_stale_scripts() returns 0 even when stale candidates exist (warning only)."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "orphan.py").write_text("", encoding="utf-8")
        result = check_stale_scripts(tmp_path)
        assert result == 0

    def test_returns_one_strict_with_stale(self, tmp_path: Path) -> None:
        """check_stale_scripts() returns 1 in strict mode when stale scripts exist."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "orphan.py").write_text("", encoding="utf-8")
        result = check_stale_scripts(tmp_path, strict=True)
        assert result == 1

    def test_returns_zero_empty_repo(self, tmp_path: Path) -> None:
        """check_stale_scripts() returns 0 for a repo with no scripts."""
        result = check_stale_scripts(tmp_path)
        assert result == 0
