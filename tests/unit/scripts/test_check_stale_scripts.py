"""Unit tests for scripts/check_stale_scripts.py."""

import sys
from pathlib import Path

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
from check_stale_scripts import (
    ALWAYS_ACTIVE,
    find_references,
    find_stale_candidates,
    get_all_scripts,
    get_reference_targets,
    main,
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
        result = get_all_scripts(scripts_dir)
        assert result == ["bar.py", "foo.py"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        """Empty scripts dir returns empty list."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        assert get_all_scripts(scripts_dir) == []

    def test_sorted(self, tmp_path: Path) -> None:
        """Results are alphabetically sorted."""
        scripts_dir = _make_scripts_dir(tmp_path, ["z.py", "a.py", "m.py"])
        assert get_all_scripts(scripts_dir) == ["a.py", "m.py", "z.py"]


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


class TestFindReferences:
    def test_found_in_justfile(self, tmp_path: Path) -> None:
        """Returns True when script name appears in justfile."""
        scripts_dir = _make_scripts_dir(tmp_path, ["my_script.py"])
        justfile = tmp_path / "justfile"
        justfile.write_text("python scripts/my_script.py\n", encoding="utf-8")
        assert find_references("my_script.py", [justfile], scripts_dir) is True

    def test_not_found_anywhere(self, tmp_path: Path) -> None:
        """Returns False when script name is absent from all targets."""
        scripts_dir = _make_scripts_dir(tmp_path, ["orphan.py"])
        justfile = tmp_path / "justfile"
        justfile.write_text("python scripts/other_script.py\n", encoding="utf-8")
        assert find_references("orphan.py", [justfile], scripts_dir) is False

    def test_self_reference_not_counted(self, tmp_path: Path) -> None:
        """Appearance of script name inside its own file does not count."""
        scripts_dir = _make_scripts_dir(tmp_path, ["self_ref.py"])
        # Write the script name into its own body
        (scripts_dir / "self_ref.py").write_text(
            "# self_ref.py — this is the script itself\n", encoding="utf-8"
        )
        assert find_references("self_ref.py", [scripts_dir / "self_ref.py"], scripts_dir) is False

    def test_cross_script_reference(self, tmp_path: Path) -> None:
        """Returns True when script name is mentioned in another script."""
        scripts_dir = _make_scripts_dir(tmp_path, ["util.py", "caller.py"])
        (scripts_dir / "caller.py").write_text(
            "import subprocess\nsubprocess.run(['python', 'scripts/util.py'])\n", encoding="utf-8"
        )
        all_targets = list(scripts_dir.glob("*.py"))
        assert find_references("util.py", all_targets, scripts_dir) is True


class TestFindStaleCandidates:
    def test_all_referenced_returns_empty(self, tmp_path: Path) -> None:
        """No candidates when every script appears in justfile."""
        scripts = ["foo.py", "bar.py"]
        justfile_content = "python scripts/foo.py\npython scripts/bar.py\n"
        repo = _make_repo(tmp_path, scripts, justfile_content)
        assert find_stale_candidates(repo) == []

    def test_unreferenced_script_flagged(self, tmp_path: Path) -> None:
        """Script with no references is returned as a candidate."""
        scripts = ["active.py", "stale.py"]
        justfile_content = "python scripts/active.py\n"
        repo = _make_repo(tmp_path, scripts, justfile_content)
        result = find_stale_candidates(repo)
        assert "stale.py" in result
        assert "active.py" not in result

    def test_always_active_excluded(self, tmp_path: Path) -> None:
        """Scripts in ALWAYS_ACTIVE are never flagged."""
        # Include ALWAYS_ACTIVE scripts plus one stale script
        scripts = list(ALWAYS_ACTIVE) + ["stale.py"]
        repo = _make_repo(tmp_path, scripts, justfile_content="")
        result = find_stale_candidates(repo)
        for name in ALWAYS_ACTIVE:
            assert name not in result
        # stale.py should be flagged (unless it self-references — it won't here)
        assert "stale.py" in result

    def test_self_reference_still_stale(self, tmp_path: Path) -> None:
        """A script that only references itself is still considered stale."""
        scripts_dir = _make_scripts_dir(tmp_path, ["self_only.py"])
        # Write own name into its body — should not save it
        (scripts_dir / "self_only.py").write_text(
            "# self_only.py does something\n", encoding="utf-8"
        )
        result = find_stale_candidates(tmp_path)
        assert "self_only.py" in result

    def test_empty_scripts_dir_returns_empty(self, tmp_path: Path) -> None:
        """Empty scripts directory returns no candidates."""
        _make_scripts_dir(tmp_path, [])
        assert find_stale_candidates(tmp_path) == []

    def test_missing_scripts_dir_returns_empty(self, tmp_path: Path) -> None:
        """Missing scripts directory returns no candidates."""
        assert find_stale_candidates(tmp_path) == []


class TestMain:
    def test_returns_zero_no_stale(self, tmp_path: Path) -> None:
        """main() returns 0 when all scripts are referenced."""
        scripts = ["referenced.py"]
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "referenced.py").write_text("", encoding="utf-8")
        (tmp_path / "justfile").write_text("python scripts/referenced.py\n", encoding="utf-8")
        result = main(["--repo-root", str(tmp_path)])
        assert result == 0

    def test_returns_zero_with_stale(self, tmp_path: Path) -> None:
        """main() returns 0 even when stale candidates exist (warning only)."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "orphan.py").write_text("", encoding="utf-8")
        result = main(["--repo-root", str(tmp_path)])
        assert result == 0

    def test_returns_zero_empty_repo(self, tmp_path: Path) -> None:
        """main() returns 0 for a repo with no scripts."""
        result = main(["--repo-root", str(tmp_path)])
        assert result == 0
