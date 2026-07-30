#!/usr/bin/env python3
"""Tests for the canonical uv requirements exporter."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sync_requirements.py"
SPEC = importlib.util.spec_from_file_location("sync_requirements", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SYNC_REQUIREMENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC_REQUIREMENTS)


class RecordingRunner:
    """Return deterministic uv output while recording subprocess calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, kwargs))
        suffix = "dev" if "--all-groups" in command else "runtime"
        stdout = f"example-{suffix}==1.0\nmojo==1.0.0b2\n    # via mojo\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_generate_requirements_uses_two_frozen_filtered_uv_exports(tmp_path: Path) -> None:
    runner = RecordingRunner()

    generated = SYNC_REQUIREMENTS.generate_requirements(tmp_path, runner=runner)

    assert set(generated) == {"requirements.txt", "requirements-dev.txt"}
    assert len(runner.calls) == 2

    runtime_command, runtime_kwargs = runner.calls[0]
    dev_command, dev_kwargs = runner.calls[1]

    for command in (runtime_command, dev_command):
        assert command[:2] == ["uv", "export"]
        assert "--frozen" in command
        assert "--no-hashes" in command
        assert "--no-header" in command
        assert "--no-emit-project" in command
        assert "--upgrade" not in command
        assert "--no-emit-package" not in command

    assert "--no-dev" in runtime_command
    assert "--all-groups" not in runtime_command
    assert "--all-groups" in dev_command
    assert "--no-dev" not in dev_command

    for kwargs in (runtime_kwargs, dev_kwargs):
        assert kwargs == {
            "cwd": tmp_path,
            "check": True,
            "capture_output": True,
            "text": True,
        }


def test_generate_requirements_preserves_headers_and_normalizes_newline(tmp_path: Path) -> None:
    class NoNewlineRunner(RecordingRunner):
        def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            completed = super().__call__(command, **kwargs)
            completed.stdout = completed.stdout.rstrip("\n")
            return completed

    generated = SYNC_REQUIREMENTS.generate_requirements(tmp_path, runner=NoNewlineRunner())

    assert generated["requirements.txt"].startswith(SYNC_REQUIREMENTS.RUNTIME_HEADER)
    assert generated["requirements-dev.txt"].startswith(SYNC_REQUIREMENTS.DEV_HEADER)
    assert "example-runtime==1.0\n" in generated["requirements.txt"]
    assert "example-dev==1.0\n" in generated["requirements-dev.txt"]
    assert generated["requirements.txt"].endswith("\n")
    assert generated["requirements-dev.txt"].endswith("\n")
    assert "mojo==" not in generated["requirements.txt"].lower()
    assert "mojo==" not in generated["requirements-dev.txt"].lower()
    # Preserve uv's diagnostic comment structure. Filtering only package lines
    # intentionally leaves existing ``# via`` comments byte-for-byte stable.
    assert "    # via mojo\n" in generated["requirements.txt"]
    assert "    # via mojo\n" in generated["requirements-dev.txt"]
    assert "AUTO-GENERATED from uv.lock (ADR-018)" in generated["requirements.txt"]
    assert "AUTO-GENERATED from uv.lock (ADR-018)" in generated["requirements-dev.txt"]


def test_sync_requirements_writes_both_exports(tmp_path: Path) -> None:
    written = SYNC_REQUIREMENTS.sync_requirements(tmp_path, runner=RecordingRunner())

    assert written == [tmp_path / "requirements.txt", tmp_path / "requirements-dev.txt"]
    assert "example-runtime==1.0\n" in (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert "example-dev==1.0\n" in (tmp_path / "requirements-dev.txt").read_text(encoding="utf-8")


def test_check_requirements_accepts_exact_exports_without_writing(tmp_path: Path) -> None:
    runner = RecordingRunner()
    SYNC_REQUIREMENTS.sync_requirements(tmp_path, runner=runner)
    before = {name: (tmp_path / name).read_bytes() for name in ("requirements.txt", "requirements-dev.txt")}

    assert SYNC_REQUIREMENTS.check_requirements_up_to_date(tmp_path, runner=runner) is True
    assert {name: (tmp_path / name).read_bytes() for name in ("requirements.txt", "requirements-dev.txt")} == before


@pytest.mark.parametrize("filename", ["requirements.txt", "requirements-dev.txt"])
def test_check_requirements_rejects_missing_or_stale_export(tmp_path: Path, filename: str) -> None:
    runner = RecordingRunner()
    SYNC_REQUIREMENTS.sync_requirements(tmp_path, runner=runner)
    target = tmp_path / filename
    target.write_text("stale\n", encoding="utf-8")

    assert SYNC_REQUIREMENTS.check_requirements_up_to_date(tmp_path, runner=runner) is False

    target.unlink()
    assert SYNC_REQUIREMENTS.check_requirements_up_to_date(tmp_path, runner=runner) is False


def test_cli_check_returns_failure_for_stale_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(SYNC_REQUIREMENTS, "check_requirements_up_to_date", lambda repo_root: False)

    assert SYNC_REQUIREMENTS.main(["--check", "--repo-root", str(tmp_path)]) == 1


def test_cli_sync_writes_then_verifies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_sync(repo_root: Path) -> list[Path]:
        calls.append(("sync", repo_root))
        return [repo_root / "requirements.txt", repo_root / "requirements-dev.txt"]

    def fake_check(repo_root: Path) -> bool:
        calls.append(("check", repo_root))
        return True

    monkeypatch.setattr(SYNC_REQUIREMENTS, "sync_requirements", fake_sync)
    monkeypatch.setattr(SYNC_REQUIREMENTS, "check_requirements_up_to_date", fake_check)

    assert SYNC_REQUIREMENTS.main(["--repo-root", str(tmp_path)]) == 0
    assert calls == [("sync", tmp_path), ("check", tmp_path)]


def test_checked_in_exports_are_an_exact_no_op() -> None:
    """Current main's generated files must not churn on a sync/check cycle."""
    before = {name: (PROJECT_ROOT / name).read_bytes() for name in ("requirements.txt", "requirements-dev.txt")}

    assert SYNC_REQUIREMENTS.check_requirements_up_to_date(PROJECT_ROOT) is True
    assert {name: (PROJECT_ROOT / name).read_bytes() for name in ("requirements.txt", "requirements-dev.txt")} == before
