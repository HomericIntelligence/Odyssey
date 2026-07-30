#!/usr/bin/env python3
"""Tests for the read-only producer and trusted Dependabot publisher."""

from __future__ import annotations

import base64
import hashlib
import io
import importlib.util
import json
import stat
import subprocess
import warnings
import zipfile
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ci" / "commit_generated_dependencies.py"
SPEC = importlib.util.spec_from_file_location("commit_generated_dependencies", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)

REPOSITORY = "HomericIntelligence/Odyssey"
BRANCH = "dependabot/pip/example-2.0"
HEAD_SHA = "a" * 40
CHILD_SHA = "b" * 40
STALE_SHA = "c" * 40
PR_NUMBER = 5723
RUN_ID = 123456
RUN_ATTEMPT = 2
WORKFLOW_ID = 654321
REPOSITORY_ID = 1089067727


def _generated_contents() -> dict[str, bytes]:
    return {
        "uv.lock": b"version = 1\nrevision = 3\n",
        "requirements.txt": b"example==2.0\n",
        "requirements-dev.txt": b"example==2.0\npytest==9.0\n",
    }


def _real_generated_contents() -> dict[str, bytes]:
    return {path: (PROJECT_ROOT / path).read_bytes() for path in PUBLISHER.GENERATED_PATHS}


def _replace_dependency_declaration(
    content: bytes,
    name: str,
    replacement: str,
) -> bytes:
    marker = f'"{name}'.encode()
    assert content.count(marker) == 1
    start = content.index(marker)
    end = content.index(b'"', start + 1) + 1
    replacement_bytes = f'"{replacement}"'.encode()
    assert content[start:end] != replacement_bytes
    return content[:start] + replacement_bytes + content[end:]


def _replace_locked_version(
    content: bytes,
    name: str,
    replacement: str,
) -> bytes:
    marker = f'name = "{name}"\nversion = "'.encode()
    assert content.count(marker) == 1
    start = content.index(marker) + len(marker)
    end = content.index(b'"', start)
    replacement_bytes = replacement.encode()
    assert content[start:end] != replacement_bytes
    return content[:start] + replacement_bytes + content[end:]


def _replace_dependency_name(
    content: bytes,
    name: str,
    replacement: str,
) -> bytes:
    marker = f'"{name}'.encode()
    assert content.count(marker) == 1
    updated = content.replace(marker, f'"{replacement}'.encode(), 1)
    assert updated != content
    return updated


def _canonical_bundle(*, files: dict[str, bytes] | None = None) -> Any:
    return PUBLISHER.ArtifactBundle(
        repository=REPOSITORY,
        pr_number=PR_NUMBER,
        head_ref=BRANCH,
        head_sha=HEAD_SHA,
        changed=True,
        files=files or _real_generated_contents(),
    )


def _successful_validation_runner(
    commands: list[list[str]],
    *,
    regenerated_lock: bytes | None = None,
    rewrite_lock: bool = False,
) -> Any:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        cwd = Path(kwargs["cwd"])
        environment = kwargs["env"]
        expected_python = (PROJECT_ROOT / ".python-version").read_text().strip()
        assert "UV_NO_CONFIG" not in environment
        assert environment["UV_PYTHON"] == expected_python
        assert (cwd / ".python-version").read_text().strip() == expected_python
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, "uv 0.12.0\n", "")
        if command[1:] == ["lock", "--no-build", "--prerelease", "allow"]:
            if regenerated_lock is not None:
                (cwd / "uv.lock").write_bytes(regenerated_lock)
            if rewrite_lock:
                (cwd / "uv.lock").write_bytes((cwd / "uv.lock").read_bytes() + b"\n")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1:] == ["lock", "--check", "--offline", "--prerelease", "allow"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        assert command[1] == str(Path(PUBLISHER.__file__).resolve().parents[1] / "sync_requirements.py")
        assert command[2:] == ["--check", "--repo-root", str(cwd)]
        return subprocess.CompletedProcess(command, 0, "OK\n", "")

    return run


def _write_generated_files(root: Path) -> dict[str, bytes]:
    contents = _generated_contents()
    for relative_path, content in contents.items():
        (root / relative_path).write_bytes(content)
    return contents


def _context() -> Any:
    return PUBLISHER.RunContext(
        repository=REPOSITORY,
        default_branch="main",
        head_ref=BRANCH,
        head_sha=HEAD_SHA,
        run_id=RUN_ID,
        run_attempt=RUN_ATTEMPT,
        workflow_id=WORKFLOW_ID,
        repository_id=REPOSITORY_ID,
        head_repository_id=REPOSITORY_ID,
    )


def _package(tmp_path: Path, *, changed: bool = True, status: str = "success") -> tuple[Path, Any]:
    repo_root = tmp_path / "repo"
    artifact_root = tmp_path / "artifact"
    snapshot_path = tmp_path / "dependency-snapshot.json"
    repo_root.mkdir()
    _write_generated_files(repo_root)
    PUBLISHER.write_dependency_snapshot(repo_root, snapshot_path)
    PUBLISHER.package_artifact(
        repo_root=repo_root,
        artifact_root=artifact_root,
        snapshot_path=snapshot_path,
        repository=REPOSITORY,
        pr_number=PR_NUMBER,
        head_repository=REPOSITORY,
        head_ref=BRANCH,
        head_sha=HEAD_SHA,
        run_id=RUN_ID,
        run_attempt=RUN_ATTEMPT,
        producer_status=status,
        changed=changed,
    )
    return artifact_root, PUBLISHER.validate_artifact(artifact_root, context=_context())


def _event() -> dict[str, Any]:
    return {
        "repository": {
            "id": REPOSITORY_ID,
            "full_name": REPOSITORY,
            "default_branch": "main",
        },
        "workflow_run": {
            "id": RUN_ID,
            "run_attempt": RUN_ATTEMPT,
            "workflow_id": WORKFLOW_ID,
            "name": PUBLISHER.GENERATOR_WORKFLOW,
            "path": ".github/workflows/dependabot-uv-lock.yml",
            "event": "pull_request",
            "conclusion": "success",
            "actor": {"login": PUBLISHER.BOT_LOGIN},
            "repository": {"id": REPOSITORY_ID, "full_name": REPOSITORY},
            "head_repository": {"id": REPOSITORY_ID, "full_name": REPOSITORY},
            "head_branch": BRANCH,
            "head_sha": HEAD_SHA,
            "pull_requests": [{"number": PR_NUMBER}],
        },
    }


def _successful_create_response() -> dict[str, Any]:
    return {
        "data": {
            "createCommitOnBranch": {
                "commit": {
                    "oid": CHILD_SHA,
                    "url": f"https://github.com/{REPOSITORY}/commit/{CHILD_SHA}",
                    "signature": {
                        "isValid": True,
                        "state": "VALID",
                        "wasSignedByGitHub": True,
                    },
                },
                "ref": {
                    "name": f"refs/heads/{BRANCH}",
                    "target": {"oid": CHILD_SHA},
                },
            }
        }
    }


def _signature_response(*, valid: bool = True) -> dict[str, Any]:
    return {
        "data": {
            "repository": {
                "object": {
                    "oid": CHILD_SHA,
                    "url": f"https://github.com/{REPOSITORY}/commit/{CHILD_SHA}",
                    "signature": {
                        "isValid": valid,
                        "state": "VALID" if valid else "INVALID",
                        "wasSignedByGitHub": valid,
                    },
                }
            }
        }
    }


def _artifact_listing(
    *,
    artifact_id: int = 987654,
    size_in_bytes: int = 1024,
    digest: str = f"sha256:{'d' * 64}",
) -> dict[str, Any]:
    return {
        "total_count": 1,
        "artifacts": [
            {
                "id": artifact_id,
                "name": PUBLISHER.ARTIFACT_NAME,
                "expired": False,
                "size_in_bytes": size_in_bytes,
                "digest": digest,
                "workflow_run": {
                    "id": RUN_ID,
                    "repository_id": REPOSITORY_ID,
                    "head_repository_id": REPOSITORY_ID,
                    "head_branch": BRANCH,
                    "head_sha": HEAD_SHA,
                },
            }
        ],
    }


def _zip_entries(entries: list[tuple[str, bytes, int | None]]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, content, unix_mode in entries:
                if unix_mode is None:
                    bundle.writestr(name, content)
                    continue
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = unix_mode << 16
                bundle.writestr(info, content)
    return archive.getvalue()


def _zip_artifact(artifact_root: Path) -> bytes:
    return _zip_entries(
        [(path.name, path.read_bytes(), stat.S_IFREG | 0o600) for path in sorted(artifact_root.iterdir())]
    )


def test_package_and_validate_exact_four_file_artifact(tmp_path: Path) -> None:
    artifact_root, bundle = _package(tmp_path)

    assert {path.name for path in artifact_root.iterdir()} == {
        *PUBLISHER.GENERATED_PATHS,
        PUBLISHER.MANIFEST_NAME,
    }
    assert bundle.repository == REPOSITORY
    assert bundle.head_sha == HEAD_SHA
    assert bundle.changed is True
    assert bundle.files == _generated_contents()

    manifest = json.loads((artifact_root / PUBLISHER.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["producer_status"] == "success"
    assert [entry["path"] for entry in manifest["files"]] == list(PUBLISHER.GENERATED_PATHS)


def test_package_rejects_generated_file_mutated_after_verified_snapshot(tmp_path: Path) -> None:
    snapshot = getattr(PUBLISHER, "write_dependency_snapshot", None)
    assert callable(snapshot), "producer must record an immediate verified dependency snapshot"
    repo_root = tmp_path / "repo"
    artifact_root = tmp_path / "artifact"
    snapshot_path = tmp_path / "dependency-snapshot.json"
    repo_root.mkdir()
    _write_generated_files(repo_root)
    snapshot(repo_root, snapshot_path)
    (repo_root / "requirements.txt").write_text("tampered-after-check\n", encoding="utf-8")

    with pytest.raises(PUBLISHER.PublishError, match="verified snapshot"):
        PUBLISHER.package_artifact(
            repo_root=repo_root,
            artifact_root=artifact_root,
            snapshot_path=snapshot_path,
            repository=REPOSITORY,
            pr_number=PR_NUMBER,
            head_repository=REPOSITORY,
            head_ref=BRANCH,
            head_sha=HEAD_SHA,
            run_id=RUN_ID,
            run_attempt=RUN_ATTEMPT,
            producer_status="success",
            changed=True,
        )

    assert not artifact_root.exists()


def test_load_run_context_binds_exact_single_pr_and_event_metadata(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(_event()), encoding="utf-8")

    context = PUBLISHER.load_run_context(event_path, target_repository=REPOSITORY)

    assert context == _context()


def test_load_run_context_rejects_wrong_workflow_path(tmp_path: Path) -> None:
    event = _event()
    event["workflow_run"]["path"] = ".github/workflows/lookalike.yml"
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    with pytest.raises(PUBLISHER.PublishError, match="workflow path"):
        PUBLISHER.load_run_context(event_path, target_repository=REPOSITORY)


def test_live_source_run_is_refetched_and_bound_to_exact_workflow_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify = getattr(PUBLISHER, "verify_live_source_run", None)
    assert callable(verify), "trusted publisher must refetch and authenticate its source run"
    requested: list[str] = []

    def fake_rest_get(url: str, **_kwargs: Any) -> dict[str, Any]:
        requested.append(url)
        if url.endswith(f"/actions/runs/{RUN_ID}"):
            return {
                "id": RUN_ID,
                "run_attempt": RUN_ATTEMPT,
                "workflow_id": WORKFLOW_ID,
                "name": PUBLISHER.GENERATOR_WORKFLOW,
                "path": PUBLISHER.GENERATOR_WORKFLOW_PATH,
                "event": "pull_request",
                "conclusion": "success",
                "actor": {"login": PUBLISHER.BOT_LOGIN},
                "repository": {"id": REPOSITORY_ID, "full_name": REPOSITORY},
                "head_repository": {"id": REPOSITORY_ID, "full_name": REPOSITORY},
                "head_branch": BRANCH,
                "head_sha": HEAD_SHA,
            }
        assert url.endswith(f"/actions/workflows/{WORKFLOW_ID}")
        return {
            "id": WORKFLOW_ID,
            "name": PUBLISHER.GENERATOR_WORKFLOW,
            "path": PUBLISHER.GENERATOR_WORKFLOW_PATH,
            "state": "active",
        }

    monkeypatch.setattr(PUBLISHER, "_rest_get", fake_rest_get)

    verify(_context(), token="token")

    assert requested == [
        f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{RUN_ID}",
        f"https://api.github.com/repos/{REPOSITORY}/actions/workflows/{WORKFLOW_ID}",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workflow_id", WORKFLOW_ID + 1),
        ("path", ".github/workflows/lookalike.yml"),
        ("head_sha", STALE_SHA),
        ("run_attempt", RUN_ATTEMPT + 1),
    ],
)
def test_live_source_run_rejects_mismatched_authoritative_metadata(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    verify = getattr(PUBLISHER, "verify_live_source_run", None)
    assert callable(verify), "trusted publisher must refetch and authenticate its source run"
    response = {
        "id": RUN_ID,
        "run_attempt": RUN_ATTEMPT,
        "workflow_id": WORKFLOW_ID,
        "name": PUBLISHER.GENERATOR_WORKFLOW,
        "path": PUBLISHER.GENERATOR_WORKFLOW_PATH,
        "event": "pull_request",
        "conclusion": "success",
        "actor": {"login": PUBLISHER.BOT_LOGIN},
        "repository": {"id": REPOSITORY_ID, "full_name": REPOSITORY},
        "head_repository": {"id": REPOSITORY_ID, "full_name": REPOSITORY},
        "head_branch": BRANCH,
        "head_sha": HEAD_SHA,
    }
    response[field] = value
    monkeypatch.setattr(PUBLISHER, "_rest_get", lambda _url, **_kwargs: response)

    with pytest.raises(PUBLISHER.PublishError, match="source workflow run"):
        verify(_context(), token="token")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", WORKFLOW_ID + 1),
        ("name", "Lookalike dependency workflow"),
        ("path", ".github/workflows/lookalike.yml"),
        ("state", "disabled_manually"),
    ],
)
def test_live_source_run_rejects_mismatched_workflow_identity(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    run_response = {
        "id": RUN_ID,
        "run_attempt": RUN_ATTEMPT,
        "workflow_id": WORKFLOW_ID,
        "name": PUBLISHER.GENERATOR_WORKFLOW,
        "path": PUBLISHER.GENERATOR_WORKFLOW_PATH,
        "event": "pull_request",
        "conclusion": "success",
        "actor": {"login": PUBLISHER.BOT_LOGIN},
        "repository": {"id": REPOSITORY_ID, "full_name": REPOSITORY},
        "head_repository": {"id": REPOSITORY_ID, "full_name": REPOSITORY},
        "head_branch": BRANCH,
        "head_sha": HEAD_SHA,
    }
    workflow_response = {
        "id": WORKFLOW_ID,
        "name": PUBLISHER.GENERATOR_WORKFLOW,
        "path": PUBLISHER.GENERATOR_WORKFLOW_PATH,
        "state": "active",
    }
    workflow_response[field] = value
    responses = iter([run_response, workflow_response])
    monkeypatch.setattr(PUBLISHER, "_rest_get", lambda _url, **_kwargs: next(responses))

    with pytest.raises(PUBLISHER.PublishError, match="exact active generator workflow"):
        PUBLISHER.verify_live_source_run(_context(), token="token")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.update({"producer_status": "failure"}), "producer_status"),
        (lambda manifest: manifest.update({"workflow_run_id": RUN_ID + 1}), "workflow_run_id"),
        (lambda manifest: manifest["files"].append(dict(manifest["files"][0])), "exactly three"),
        (lambda manifest: manifest["files"][0].update({"sha256": "0" * 64}), "contradicts"),
        (lambda manifest: manifest.update({"unexpected": True}), "unexpected fields"),
    ],
)
def test_validate_artifact_rejects_malformed_or_contradictory_manifest(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    artifact_root, _bundle = _package(tmp_path)
    manifest_path = artifact_root / PUBLISHER.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PUBLISHER.PublishError, match=message):
        PUBLISHER.validate_artifact(artifact_root, context=_context())


def test_validate_artifact_rejects_extra_nested_or_symlink_entry(tmp_path: Path) -> None:
    artifact_root, _bundle = _package(tmp_path)
    (artifact_root / "extra").mkdir()

    with pytest.raises(PUBLISHER.PublishError, match="exactly"):
        PUBLISHER.validate_artifact(artifact_root, context=_context())


def test_validate_artifact_rejects_malformed_json(tmp_path: Path) -> None:
    artifact_root, _bundle = _package(tmp_path)
    (artifact_root / PUBLISHER.MANIFEST_NAME).write_text("{", encoding="utf-8")

    with pytest.raises(PUBLISHER.PublishError, match="valid JSON"):
        PUBLISHER.validate_artifact(artifact_root, context=_context())


def test_artifact_preflight_binds_exact_single_archive_to_source_run() -> None:
    preflight = getattr(PUBLISHER, "validate_artifact_listing", None)
    assert callable(preflight), "privileged publisher must preflight artifact metadata"

    metadata = preflight(_artifact_listing(), context=_context())

    assert metadata.artifact_id == 987654
    assert metadata.size_in_bytes == 1024
    assert metadata.digest == "d" * 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda listing: listing.update({"total_count": 0, "artifacts": []}), "exactly one"),
        (
            lambda listing: listing["artifacts"].append(dict(listing["artifacts"][0])),
            "exactly one",
        ),
        (lambda listing: listing["artifacts"][0].update({"expired": True}), "expired"),
        (
            lambda listing: listing["artifacts"][0].update({"size_in_bytes": 128 * 1024 * 1024}),
            "size",
        ),
        (lambda listing: listing["artifacts"][0].update({"digest": "sha256:bad"}), "digest"),
        (
            lambda listing: listing["artifacts"][0]["workflow_run"].update({"head_sha": STALE_SHA}),
            "source workflow run",
        ),
    ],
)
def test_artifact_preflight_rejects_ambiguous_or_unbound_archives(
    mutation: Any,
    message: str,
) -> None:
    preflight = getattr(PUBLISHER, "validate_artifact_listing", None)
    assert callable(preflight), "privileged publisher must preflight artifact metadata"
    listing = _artifact_listing()
    mutation(listing)

    with pytest.raises(PUBLISHER.PublishError, match=message):
        preflight(listing, context=_context())


def test_safe_archive_reader_accepts_only_exact_flat_bounded_files(tmp_path: Path) -> None:
    read_archive = getattr(PUBLISHER, "read_safe_artifact_archive", None)
    assert callable(read_archive), "privileged publisher must inspect archives before extracting"
    artifact_root, expected = _package(tmp_path)

    entries = read_archive(_zip_artifact(artifact_root))
    bundle = PUBLISHER.validate_artifact_entries(entries, context=_context())

    assert bundle == expected


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            [
                ("../uv.lock", b"escape", stat.S_IFREG | 0o600),
                ("requirements.txt", b"x", stat.S_IFREG | 0o600),
                ("requirements-dev.txt", b"x", stat.S_IFREG | 0o600),
                ("manifest.json", b"{}", stat.S_IFREG | 0o600),
            ],
            "exact flat",
        ),
        (
            [
                ("uv.lock", b"one", stat.S_IFREG | 0o600),
                ("uv.lock", b"two", stat.S_IFREG | 0o600),
                ("requirements.txt", b"x", stat.S_IFREG | 0o600),
                ("manifest.json", b"{}", stat.S_IFREG | 0o600),
            ],
            "duplicate",
        ),
        (
            [
                ("uv.lock", b"target", stat.S_IFLNK | 0o777),
                ("requirements.txt", b"x", stat.S_IFREG | 0o600),
                ("requirements-dev.txt", b"x", stat.S_IFREG | 0o600),
                ("manifest.json", b"{}", stat.S_IFREG | 0o600),
            ],
            "symlink",
        ),
    ],
)
def test_safe_archive_reader_rejects_traversal_duplicates_and_symlinks(
    entries: list[tuple[str, bytes, int | None]],
    message: str,
) -> None:
    read_archive = getattr(PUBLISHER, "read_safe_artifact_archive", None)
    assert callable(read_archive), "privileged publisher must inspect archives before extracting"

    with pytest.raises(PUBLISHER.PublishError, match=message):
        read_archive(_zip_entries(entries))


def test_privileged_loader_preflights_download_by_id_and_validates_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = getattr(PUBLISHER, "download_validated_artifact", None)
    assert callable(loader), "privileged publisher must own bounded artifact download and validation"
    artifact_root, expected = _package(tmp_path)
    archive = _zip_artifact(artifact_root)
    digest = hashlib.sha256(archive).hexdigest()
    listing = _artifact_listing(size_in_bytes=len(archive), digest=f"sha256:{digest}")
    requested: list[str] = []
    downloaded: list[int] = []

    def fake_rest_get(url: str, **_kwargs: Any) -> dict[str, Any]:
        requested.append(url)
        return listing

    def fake_download(metadata: Any, context: Any, **_kwargs: Any) -> bytes:
        assert context == _context()
        downloaded.append(metadata.artifact_id)
        return archive

    monkeypatch.setattr(PUBLISHER, "_rest_get", fake_rest_get)
    monkeypatch.setattr(PUBLISHER, "download_artifact_archive", fake_download)

    bundle = loader(_context(), token="token")

    assert bundle == expected
    assert requested == [
        f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{RUN_ID}/artifacts"
        "?name=dependabot-generated-dependencies&per_page=100"
    ]
    assert downloaded == [987654]


def test_privileged_loader_rejects_archive_digest_or_size_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = getattr(PUBLISHER, "download_validated_artifact", None)
    assert callable(loader), "privileged publisher must own bounded artifact download and validation"
    artifact_root, _expected = _package(tmp_path)
    archive = _zip_artifact(artifact_root)
    listing = _artifact_listing(
        size_in_bytes=len(archive) + 1,
        digest=f"sha256:{hashlib.sha256(archive).hexdigest()}",
    )
    monkeypatch.setattr(PUBLISHER, "_rest_get", lambda _url, **_kwargs: listing)
    monkeypatch.setattr(PUBLISHER, "download_artifact_archive", lambda *_args, **_kwargs: archive)

    with pytest.raises(PUBLISHER.PublishError, match="size"):
        loader(_context(), token="token")

    listing["artifacts"][0]["size_in_bytes"] = len(archive)
    listing["artifacts"][0]["digest"] = f"sha256:{'0' * 64}"
    with pytest.raises(PUBLISHER.PublishError, match="digest"):
        loader(_context(), token="token")


def test_trusted_validator_accepts_only_pinned_uv_canonical_lock_and_exports() -> None:
    validator = getattr(PUBLISHER, "validate_canonical_dependencies", None)
    assert callable(validator), "trusted publisher must independently validate generated dependency bytes"
    commands: list[list[str]] = []

    validator(
        _canonical_bundle(),
        candidate_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
        trusted_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
        trusted_lock=(PROJECT_ROOT / "uv.lock").read_bytes(),
        trusted_python_version=(PROJECT_ROOT / ".python-version").read_bytes(),
        runner=_successful_validation_runner(commands),
    )

    assert [command[1:] for command in commands[:3]] == [
        ["--version"],
        ["lock", "--no-build", "--prerelease", "allow"],
        ["lock", "--check", "--offline", "--prerelease", "allow"],
    ]
    assert commands[3][1:] == [
        str(Path(PUBLISHER.__file__).resolve().parents[1] / "sync_requirements.py"),
        "--check",
        "--repo-root",
        commands[3][-1],
    ]


@pytest.mark.parametrize(
    ("relative_path", "payload", "message"),
    [
        ("uv.lock", b"not a uv lock\n", "uv.lock"),
        ("requirements.txt", b"-r https://attacker.invalid/requirements.txt\n", "include|option|URL"),
        ("requirements-dev.txt", b"evil @ https://attacker.invalid/evil.whl\n", "URL"),
    ],
)
def test_trusted_validator_rejects_invalid_lock_urls_and_includes_before_running_uv(
    relative_path: str,
    payload: bytes,
    message: str,
) -> None:
    validator = getattr(PUBLISHER, "validate_canonical_dependencies", None)
    assert callable(validator), "trusted publisher must independently validate generated dependency bytes"
    files = _real_generated_contents()
    files[relative_path] = payload

    with pytest.raises(PUBLISHER.PublishError, match=message):
        validator(
            _canonical_bundle(files=files),
            candidate_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
            trusted_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
            trusted_lock=(PROJECT_ROOT / "uv.lock").read_bytes(),
            trusted_python_version=(PROJECT_ROOT / ".python-version").read_bytes(),
            runner=lambda *_args, **_kwargs: pytest.fail("unsafe bytes must fail before invoking uv"),
        )


def test_trusted_validator_rejects_noncanonical_lock_bytes() -> None:
    validator = getattr(PUBLISHER, "validate_canonical_dependencies", None)
    assert callable(validator), "trusted publisher must independently validate generated dependency bytes"

    with pytest.raises(PUBLISHER.PublishError, match="canonical"):
        validator(
            _canonical_bundle(),
            candidate_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
            trusted_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
            trusted_lock=(PROJECT_ROOT / "uv.lock").read_bytes(),
            trusted_python_version=(PROJECT_ROOT / ".python-version").read_bytes(),
            runner=_successful_validation_runner([], rewrite_lock=True),
        )


def test_trusted_validator_rejects_unused_canonical_looking_lock_package() -> None:
    validator = getattr(PUBLISHER, "validate_canonical_dependencies", None)
    assert callable(validator), "trusted publisher must independently validate generated dependency bytes"
    files = _real_generated_contents()
    files["uv.lock"] += (
        b'\n[[package]]\nname = "zzz-unused"\nversion = "1.0"\nsource = { registry = "https://pypi.org/simple" }\n'
    )

    with pytest.raises(PUBLISHER.PublishError, match="unreachable|trusted default lock"):
        validator(
            _canonical_bundle(files=files),
            candidate_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
            trusted_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
            trusted_lock=(PROJECT_ROOT / "uv.lock").read_bytes(),
            trusted_python_version=(PROJECT_ROOT / ".python-version").read_bytes(),
            runner=_successful_validation_runner([]),
        )


def test_trusted_validator_rejects_contaminated_lock_with_unrelated_upgrades() -> None:
    validator = getattr(PUBLISHER, "validate_canonical_dependencies", None)
    assert callable(validator), "trusted publisher must independently validate generated dependency bytes"
    trusted_pyproject = (PROJECT_ROOT / "pyproject.toml").read_bytes()
    candidate_pyproject = _replace_dependency_declaration(
        trusted_pyproject,
        "types-PyYAML",
        "types-PyYAML>=999.0.0",
    )
    trusted_lock = (PROJECT_ROOT / "uv.lock").read_bytes()
    canonical_lock = _replace_locked_version(
        trusted_lock,
        "types-pyyaml",
        "999.0.0",
    )
    files = _real_generated_contents()
    files["uv.lock"] = _replace_locked_version(
        canonical_lock,
        "gitpython",
        "999.0.0",
    )
    commands: list[list[str]] = []

    with pytest.raises(PUBLISHER.PublishError, match="trusted default lock"):
        validator(
            _canonical_bundle(files=files),
            candidate_pyproject=candidate_pyproject,
            trusted_pyproject=trusted_pyproject,
            trusted_lock=trusted_lock,
            trusted_python_version=(PROJECT_ROOT / ".python-version").read_bytes(),
            runner=_successful_validation_runner(
                commands,
                regenerated_lock=canonical_lock,
            ),
        )

    regeneration = commands[1]
    assert regeneration[1:] == ["lock", "--no-build", "--prerelease", "allow"]
    assert "--upgrade-package" not in regeneration


def test_trusted_validator_accepts_clean_recreated_dependency_head() -> None:
    validator = getattr(PUBLISHER, "validate_canonical_dependencies", None)
    assert callable(validator), "trusted publisher must independently validate generated dependency bytes"
    trusted_pyproject = (PROJECT_ROOT / "pyproject.toml").read_bytes()
    candidate_pyproject = _replace_dependency_declaration(
        trusted_pyproject,
        "types-PyYAML",
        "types-PyYAML>=999.0.0",
    )
    trusted_lock = (PROJECT_ROOT / "uv.lock").read_bytes()
    canonical_lock = _replace_locked_version(
        trusted_lock,
        "types-pyyaml",
        "999.0.0",
    )
    files = _real_generated_contents()
    files["uv.lock"] = canonical_lock
    commands: list[list[str]] = []

    validator(
        _canonical_bundle(files=files),
        candidate_pyproject=candidate_pyproject,
        trusted_pyproject=trusted_pyproject,
        trusted_lock=trusted_lock,
        trusted_python_version=(PROJECT_ROOT / ".python-version").read_bytes(),
        runner=_successful_validation_runner(
            commands,
            regenerated_lock=canonical_lock,
        ),
    )

    assert commands[1][1:] == ["lock", "--no-build", "--prerelease", "allow"]
    assert "--upgrade-package" not in commands[1]


@pytest.mark.parametrize(
    "rewritten_name",
    ["GitPython", "git-python", "git_python"],
)
def test_trusted_validator_rejects_identity_only_rewrite_without_upgrade_authority(
    rewritten_name: str,
) -> None:
    validator = getattr(PUBLISHER, "validate_canonical_dependencies", None)
    assert callable(validator), "trusted publisher must independently validate generated dependency bytes"
    trusted_pyproject = (PROJECT_ROOT / "pyproject.toml").read_bytes()
    candidate_pyproject = _replace_dependency_name(
        trusted_pyproject,
        "gitpython",
        rewritten_name,
    )

    with pytest.raises(PUBLISHER.PublishError, match="identit"):
        validator(
            _canonical_bundle(),
            candidate_pyproject=candidate_pyproject,
            trusted_pyproject=trusted_pyproject,
            trusted_lock=(PROJECT_ROOT / "uv.lock").read_bytes(),
            trusted_python_version=(PROJECT_ROOT / ".python-version").read_bytes(),
            runner=lambda *_args, **_kwargs: pytest.fail("identity-only rewrite must not invoke uv"),
        )


def test_trusted_validator_rejects_noncanonical_requirement_bytes() -> None:
    validator = getattr(PUBLISHER, "validate_canonical_dependencies", None)
    assert callable(validator), "trusted publisher must independently validate generated dependency bytes"
    commands: list[list[str]] = []
    successful_runner = _successful_validation_runner(commands)

    def reject_stale_export(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        completed = successful_runner(command, **kwargs)
        if len(command) > 1 and command[1].endswith("sync_requirements.py"):
            return subprocess.CompletedProcess(command, 1, "", "requirements.txt is stale")
        return completed

    files = _real_generated_contents()
    files["requirements.txt"] += b"\n"
    with pytest.raises(PUBLISHER.PublishError, match="canonical requirements export"):
        validator(
            _canonical_bundle(files=files),
            candidate_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
            trusted_pyproject=(PROJECT_ROOT / "pyproject.toml").read_bytes(),
            trusted_lock=(PROJECT_ROOT / "uv.lock").read_bytes(),
            trusted_python_version=(PROJECT_ROOT / ".python-version").read_bytes(),
            runner=reject_stale_export,
        )


def test_trusted_validator_rejects_candidate_project_urls_or_unrelated_changes() -> None:
    validator = getattr(PUBLISHER, "validate_canonical_dependencies", None)
    assert callable(validator), "trusted publisher must independently validate generated dependency bytes"
    trusted = (PROJECT_ROOT / "pyproject.toml").read_bytes()
    requirement_lines = [
        line for line in trusted.splitlines(keepends=True) if line.lstrip().startswith(b'"types-PyYAML')
    ]
    assert len(requirement_lines) == 1, "test fixture must contain exactly one types-PyYAML requirement"
    requirement_line = requirement_lines[0]
    indentation = requirement_line[: len(requirement_line) - len(requirement_line.lstrip())]
    line_ending = b"\n" if requirement_line.endswith(b"\n") else b""
    candidate = trusted.replace(
        requirement_line,
        indentation + b'"types-PyYAML @ https://attacker.invalid/types.whl",' + line_ending,
        1,
    )
    assert candidate != trusted, "adversarial dependency mutation must change the candidate"

    with pytest.raises(PUBLISHER.PublishError, match="dependency|URL"):
        validator(
            _canonical_bundle(),
            candidate_pyproject=candidate,
            trusted_pyproject=trusted,
            trusted_lock=(PROJECT_ROOT / "uv.lock").read_bytes(),
            trusted_python_version=(PROJECT_ROOT / ".python-version").read_bytes(),
            runner=lambda *_args, **_kwargs: pytest.fail("unsafe project metadata must fail before invoking uv"),
        )


def test_publish_rejects_uncanonical_artifact_before_diff_or_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifact_root, bundle = _package(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(PUBLISHER, "get_pull_request_head", lambda *_args, **_kwargs: HEAD_SHA)
    monkeypatch.setattr(PUBLISHER, "get_branch_head", lambda *_args, **_kwargs: HEAD_SHA)

    def reject_candidate(*_args: Any, **_kwargs: Any) -> None:
        calls.append("validate")
        raise PUBLISHER.PublishError("candidate is not canonical")

    monkeypatch.setattr(PUBLISHER, "validate_published_dependency_candidate", reject_candidate)
    monkeypatch.setattr(
        PUBLISHER,
        "compare_artifact_to_source",
        lambda *_args, **_kwargs: pytest.fail("diff must follow canonical validation"),
    )
    monkeypatch.setattr(
        PUBLISHER,
        "create_signed_commit",
        lambda *_args, **_kwargs: pytest.fail("commit payload must not be built"),
    )

    with pytest.raises(PUBLISHER.PublishError, match="not canonical"):
        PUBLISHER.publish_dependencies(
            context=_context(),
            bundle=bundle,
            trusted_default_oid=HEAD_SHA,
            token="token",
        )
    assert calls == ["validate"]


def test_artifact_download_uses_immutable_id_and_never_forwards_token() -> None:
    metadata = PUBLISHER.ArtifactMetadata(
        artifact_id=987654,
        size_in_bytes=7,
        digest="d" * 64,
    )
    archive = b"archive"
    signed_url = "https://results.example.invalid/signed-artifact"
    requests: list[tuple[str, str | None]] = []

    class FakeResponse:
        def __init__(self, *, status: int, headers: dict[str, str], body: bytes = b"") -> None:
            self.status = status
            self.headers = headers
            self.body = body

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, maximum_bytes: int = -1) -> bytes:
            return self.body if maximum_bytes < 0 else self.body[:maximum_bytes]

    def redirect_opener(request: Any, *, timeout: int) -> FakeResponse:
        assert timeout == 30
        requests.append((request.full_url, request.get_header("Authorization")))
        return FakeResponse(status=302, headers={"Location": signed_url})

    def download_opener(request: Any, *, timeout: int) -> FakeResponse:
        assert timeout == 30
        requests.append((request.full_url, request.get_header("Authorization")))
        return FakeResponse(
            status=200,
            headers={"Content-Length": str(len(archive))},
            body=archive,
        )

    result = PUBLISHER.download_artifact_archive(
        metadata,
        _context(),
        token="secret",
        redirect_opener=redirect_opener,
        download_opener=download_opener,
    )

    assert result == archive
    assert requests == [
        (
            f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/987654/zip",
            "Bearer secret",
        ),
        (signed_url, None),
    ]


@pytest.mark.parametrize("declared_length", ["invalid", "-1"])
def test_artifact_download_rejects_invalid_declared_length(declared_length: str) -> None:
    metadata = PUBLISHER.ArtifactMetadata(
        artifact_id=987654,
        size_in_bytes=7,
        digest="d" * 64,
    )

    class FakeResponse:
        status = 200
        headers = {"Content-Length": declared_length}

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _maximum_bytes: int = -1) -> bytes:
            return b"archive"

    class FakeRedirect(FakeResponse):
        status = 302
        headers = {"Location": "https://results.example.invalid/signed-artifact"}

    with pytest.raises(PUBLISHER.PublishError, match="invalid length"):
        PUBLISHER.download_artifact_archive(
            metadata,
            _context(),
            token="secret",
            redirect_opener=lambda *_args, **_kwargs: FakeRedirect(),
            download_opener=lambda *_args, **_kwargs: FakeResponse(),
        )


@pytest.mark.parametrize("pull_requests", [[], [{"number": 1}, {"number": 2}]])
def test_load_run_context_does_not_trust_unreliable_payload_pr_array(
    tmp_path: Path,
    pull_requests: list[dict[str, int]],
) -> None:
    event = _event()
    event["workflow_run"]["pull_requests"] = pull_requests
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    assert PUBLISHER.load_run_context(event_path, target_repository=REPOSITORY) == _context()


def test_commit_payload_contains_exact_files_parent_and_dco_body(tmp_path: Path) -> None:
    _artifact_root, bundle = _package(tmp_path)

    payload = PUBLISHER.build_commit_payload(bundle)

    mutation_input = payload["variables"]["input"]
    assert mutation_input["branch"] == {
        "repositoryNameWithOwner": REPOSITORY,
        "branchName": BRANCH,
    }
    assert mutation_input["expectedHeadOid"] == HEAD_SHA
    assert mutation_input["message"] == {
        "headline": f"fix(uv): regenerate dependency files for Dependabot PR #{PR_NUMBER}",
        "body": PUBLISHER.DCO_BODY,
    }
    additions = mutation_input["fileChanges"]["additions"]
    assert [addition["path"] for addition in additions] == list(PUBLISHER.GENERATED_PATHS)
    assert {
        addition["path"]: base64.b64decode(addition["contents"], validate=True) for addition in additions
    } == _generated_contents()


@pytest.mark.parametrize(
    "signature",
    [
        None,
        {"isValid": False, "state": "INVALID", "wasSignedByGitHub": True},
        {"isValid": True, "state": "VALID", "wasSignedByGitHub": False},
        {"isValid": True, "state": "UNKNOWN", "wasSignedByGitHub": True},
    ],
)
def test_create_response_rejects_missing_or_invalid_github_signature(signature: Any) -> None:
    response = _successful_create_response()
    response["data"]["createCommitOnBranch"]["commit"]["signature"] = signature

    with pytest.raises(PUBLISHER.PublishError, match="signature|signed by GitHub"):
        PUBLISHER.validate_create_response(response, branch=BRANCH)


def test_verify_exact_child_checks_parent_diff_message_signature_and_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifact_root, bundle = _package(tmp_path)

    def fake_rest_get(url: str, **_kwargs: Any) -> dict[str, Any]:
        assert url.endswith(f"/commits/{CHILD_SHA}")
        return {
            "sha": CHILD_SHA,
            "commit": {
                "message": f"fix(uv): regenerate dependency files for Dependabot PR #{PR_NUMBER}\n\n"
                f"{PUBLISHER.DCO_BODY}"
            },
            "parents": [{"sha": HEAD_SHA}],
            "files": [{"filename": "uv.lock"}],
        }

    monkeypatch.setattr(PUBLISHER, "_rest_get", fake_rest_get)
    monkeypatch.setattr(
        PUBLISHER,
        "get_file_at_oid",
        lambda _repository, path, _oid, **_kwargs: bundle.files[path],
    )

    result = PUBLISHER.verify_exact_child(
        bundle=bundle,
        commit_oid=CHILD_SHA,
        changed_paths=("uv.lock",),
        token="token",
        sender=lambda _payload, _token: _signature_response(),
    )

    assert result.oid == CHILD_SHA


def test_verify_exact_child_rejects_extra_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifact_root, bundle = _package(tmp_path)

    monkeypatch.setattr(
        PUBLISHER,
        "_rest_get",
        lambda _url, **_kwargs: {
            "sha": CHILD_SHA,
            "commit": {
                "message": f"fix(uv): regenerate dependency files for Dependabot PR #{PR_NUMBER}\n\n"
                f"{PUBLISHER.DCO_BODY}"
            },
            "parents": [{"sha": HEAD_SHA}],
            "files": [{"filename": "uv.lock"}, {"filename": "README.md"}],
        },
    )
    with pytest.raises(PUBLISHER.ExactChildMismatch, match="outside"):
        PUBLISHER.verify_exact_child(
            bundle=bundle,
            commit_oid=CHILD_SHA,
            changed_paths=("uv.lock",),
            token="token",
            sender=lambda _payload, _token: _signature_response(valid=False),
        )


def test_verify_exact_child_rejects_duplicate_parent_or_invalid_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifact_root, bundle = _package(tmp_path)
    response = {
        "sha": CHILD_SHA,
        "commit": {
            "message": f"fix(uv): regenerate dependency files for Dependabot PR #{PR_NUMBER}\n\n{PUBLISHER.DCO_BODY}"
        },
        "parents": [{"sha": HEAD_SHA}, {"sha": HEAD_SHA}],
        "files": [{"filename": "uv.lock"}],
    }
    monkeypatch.setattr(PUBLISHER, "_rest_get", lambda _url, **_kwargs: response)
    with pytest.raises(PUBLISHER.ExactChildMismatch, match="exact child"):
        PUBLISHER.verify_exact_child(
            bundle=bundle,
            commit_oid=CHILD_SHA,
            changed_paths=("uv.lock",),
            token="token",
            sender=lambda _payload, _token: _signature_response(),
        )

    response["parents"] = [{"sha": HEAD_SHA}]
    monkeypatch.setattr(
        PUBLISHER,
        "get_file_at_oid",
        lambda _repository, path, _oid, **_kwargs: bundle.files[path],
    )
    with pytest.raises(PUBLISHER.ExactChildMismatch, match="not valid"):
        PUBLISHER.verify_exact_child(
            bundle=bundle,
            commit_oid=CHILD_SHA,
            changed_paths=("uv.lock",),
            token="token",
            sender=lambda _payload, _token: _signature_response(valid=False),
        )


def test_changed_flag_must_match_immutable_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifact_root, bundle = _package(tmp_path, changed=True)
    monkeypatch.setattr(
        PUBLISHER,
        "get_file_at_oid",
        lambda _repository, path, _oid, **_kwargs: bundle.files[path],
    )

    with pytest.raises(PUBLISHER.PublishError, match="changed flag contradicts"):
        PUBLISHER.compare_artifact_to_source(bundle, token="token")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("user", {"login": "attacker"}, "not Dependabot"),
        (
            "head",
            {"ref": "dependabot/pip/other", "sha": HEAD_SHA, "repo": {"full_name": REPOSITORY}},
            "head branch",
        ),
        (
            "head",
            {"ref": BRANCH, "sha": HEAD_SHA, "repo": {"full_name": "attacker/fork"}},
            "head repository",
        ),
        ("base", {"ref": "release"}, "default branch"),
    ],
)
def test_live_pr_lookup_enforces_dependabot_same_repo_branch_and_base(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: dict[str, Any],
    message: str,
) -> None:
    response = {
        "state": "open",
        "user": {"login": PUBLISHER.BOT_LOGIN},
        "head": {"ref": BRANCH, "sha": HEAD_SHA, "repo": {"full_name": REPOSITORY}},
        "base": {"ref": "main"},
    }
    response[field] = value
    monkeypatch.setattr(PUBLISHER, "_rest_get", lambda _url, **_kwargs: response)

    with pytest.raises(PUBLISHER.PublishError, match=message):
        PUBLISHER.get_pull_request_head(
            _context(),
            pr_number=PR_NUMBER,
            token="token",
        )


def test_retry_after_partial_commit_verifies_existing_child_and_resumes_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifact_root, bundle = _package(tmp_path)
    verified_calls: list[str] = []
    dispatch_calls: list[str] = []

    monkeypatch.setattr(PUBLISHER, "get_pull_request_head", lambda *_args, **_kwargs: CHILD_SHA)
    monkeypatch.setattr(PUBLISHER, "get_branch_head", lambda *_args, **_kwargs: CHILD_SHA)
    monkeypatch.setattr(PUBLISHER, "validate_published_dependency_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(PUBLISHER, "compare_artifact_to_source", lambda *_args, **_kwargs: ("uv.lock",))
    monkeypatch.setattr(
        PUBLISHER,
        "create_signed_commit",
        lambda *_args, **_kwargs: pytest.fail("retry must not create a second commit"),
    )

    def verify_existing(**kwargs: Any) -> Any:
        verified_calls.append(kwargs["commit_oid"])
        return PUBLISHER.CommitResult(CHILD_SHA, f"https://github.com/{REPOSITORY}/commit/{CHILD_SHA}")

    def resume_dispatches(**kwargs: Any) -> int:
        dispatch_calls.append(kwargs["commit_oid"])
        return 2

    monkeypatch.setattr(PUBLISHER, "verify_exact_child", verify_existing)
    monkeypatch.setattr(PUBLISHER, "dispatch_required_workflows", resume_dispatches)
    monkeypatch.setattr(PUBLISHER, "dispatch_comment_workflow", lambda **_kwargs: None)

    result = PUBLISHER.publish_dependencies(
        context=_context(),
        bundle=bundle,
        trusted_default_oid=HEAD_SHA,
        token="token",
    )

    assert result == PUBLISHER.PublishResult("published", CHILD_SHA, 2)
    assert verified_calls == [CHILD_SHA]
    assert dispatch_calls == [CHILD_SHA]


@pytest.mark.parametrize(
    ("pr_head", "branch_head"),
    [
        (HEAD_SHA, CHILD_SHA),
        (CHILD_SHA, HEAD_SHA),
    ],
)
def test_retry_reconciles_split_parent_child_visibility_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pr_head: str,
    branch_head: str,
) -> None:
    """A partially visible exact child must resume instead of becoming stale."""
    _artifact_root, bundle = _package(tmp_path)
    verified_calls: list[str] = []
    waited_calls: list[str] = []
    dispatched_calls: list[str] = []

    monkeypatch.setattr(PUBLISHER, "get_pull_request_head", lambda *_args, **_kwargs: pr_head)
    monkeypatch.setattr(PUBLISHER, "get_branch_head", lambda *_args, **_kwargs: branch_head)
    monkeypatch.setattr(PUBLISHER, "validate_published_dependency_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(PUBLISHER, "compare_artifact_to_source", lambda *_args, **_kwargs: ("uv.lock",))
    monkeypatch.setattr(
        PUBLISHER,
        "create_signed_commit",
        lambda *_args, **_kwargs: pytest.fail("split visibility must not create a second commit"),
    )

    def verify_existing(**kwargs: Any) -> Any:
        verified_calls.append(kwargs["commit_oid"])
        return PUBLISHER.CommitResult(CHILD_SHA, f"https://github.com/{REPOSITORY}/commit/{CHILD_SHA}")

    def wait_for_child(**kwargs: Any) -> bool:
        waited_calls.append(kwargs["expected_oid"])
        return True

    def resume_dispatches(**kwargs: Any) -> int:
        dispatched_calls.append(kwargs["commit_oid"])
        return 4

    monkeypatch.setattr(PUBLISHER, "verify_exact_child", verify_existing)
    monkeypatch.setattr(PUBLISHER, "wait_for_published_head", wait_for_child)
    monkeypatch.setattr(PUBLISHER, "dispatch_required_workflows", resume_dispatches)
    monkeypatch.setattr(PUBLISHER, "dispatch_comment_workflow", lambda **_kwargs: None)

    result = PUBLISHER.publish_dependencies(
        context=_context(),
        bundle=bundle,
        trusted_default_oid=HEAD_SHA,
        token="token",
    )

    assert result == PUBLISHER.PublishResult("published", CHILD_SHA, 4)
    assert verified_calls == [CHILD_SHA]
    assert waited_calls == [CHILD_SHA]
    assert dispatched_calls == [CHILD_SHA]


def test_unrelated_moved_branch_is_safe_noop_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifact_root, bundle = _package(tmp_path)
    monkeypatch.setattr(PUBLISHER, "get_pull_request_head", lambda *_args, **_kwargs: STALE_SHA)
    monkeypatch.setattr(PUBLISHER, "get_branch_head", lambda *_args, **_kwargs: STALE_SHA)
    monkeypatch.setattr(PUBLISHER, "validate_published_dependency_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(PUBLISHER, "compare_artifact_to_source", lambda *_args, **_kwargs: ("uv.lock",))
    monkeypatch.setattr(
        PUBLISHER,
        "verify_exact_child",
        lambda **_kwargs: (_ for _ in ()).throw(PUBLISHER.ExactChildMismatch("not exact")),
    )
    monkeypatch.setattr(
        PUBLISHER,
        "dispatch_required_workflows",
        lambda **_kwargs: pytest.fail("stale branches must not dispatch"),
    )

    result = PUBLISHER.publish_dependencies(
        context=_context(),
        bundle=bundle,
        trusted_default_oid=HEAD_SHA,
        token="token",
    )

    assert result == PUBLISHER.PublishResult("stale", None, 0)


@pytest.mark.parametrize("lagging_endpoint", ["pull request", "branch"])
def test_published_head_poll_retries_split_parent_visibility_in_both_directions(
    monkeypatch: pytest.MonkeyPatch,
    lagging_endpoint: str,
) -> None:
    lagging_heads = iter([HEAD_SHA, CHILD_SHA])
    sleeps: list[float] = []
    if lagging_endpoint == "pull request":
        monkeypatch.setattr(
            PUBLISHER,
            "get_pull_request_head",
            lambda *_args, **_kwargs: next(lagging_heads),
        )
        monkeypatch.setattr(PUBLISHER, "get_branch_head", lambda *_args, **_kwargs: CHILD_SHA)
    else:
        monkeypatch.setattr(PUBLISHER, "get_pull_request_head", lambda *_args, **_kwargs: CHILD_SHA)
        monkeypatch.setattr(
            PUBLISHER,
            "get_branch_head",
            lambda *_args, **_kwargs: next(lagging_heads),
        )

    assert PUBLISHER.wait_for_published_head(
        context=_context(),
        pr_number=PR_NUMBER,
        expected_oid=CHILD_SHA,
        token="token",
        sleeper=sleeps.append,
    )
    assert sleeps == [1.0]


def test_published_head_poll_rejects_newer_oid_and_fails_closed_on_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PUBLISHER, "get_pull_request_head", lambda *_args, **_kwargs: STALE_SHA)
    monkeypatch.setattr(PUBLISHER, "get_branch_head", lambda *_args, **_kwargs: CHILD_SHA)
    assert not PUBLISHER.wait_for_published_head(
        context=_context(),
        pr_number=PR_NUMBER,
        expected_oid=CHILD_SHA,
        token="token",
        sleeper=lambda _seconds: None,
    )

    sleeps: list[float] = []
    monkeypatch.setattr(PUBLISHER, "get_pull_request_head", lambda *_args, **_kwargs: HEAD_SHA)
    with pytest.raises(PUBLISHER.PublishError, match="did not converge"):
        PUBLISHER.wait_for_published_head(
            context=_context(),
            pr_number=PR_NUMBER,
            expected_oid=CHILD_SHA,
            token="token",
            sleeper=sleeps.append,
            poll_attempts=2,
        )
    assert sleeps == [1.0]


def test_dispatch_allowlist_skips_existing_oid_runs_and_polls_new_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = set(PUBLISHER.REQUIRED_WORKFLOWS[:2])
    dispatched: list[str] = []
    run_queries: dict[str, int] = {}
    authenticated: list[tuple[str, str]] = []
    monkeypatch.setattr(PUBLISHER, "get_branch_head", lambda *_args, **_kwargs: CHILD_SHA)
    monkeypatch.setattr(
        PUBLISHER,
        "authenticate_required_workflows",
        lambda **kwargs: authenticated.append((kwargs["commit_oid"], kwargs["trusted_default_oid"])),
    )

    def workflow_runs(
        _repository: str,
        workflow: str,
        _branch: str,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        run_queries[workflow] = run_queries.get(workflow, 0) + 1
        if workflow in existing or run_queries[workflow] > 1:
            return [{"id": 1, "head_sha": CHILD_SHA, "html_url": "https://example.invalid/run"}]
        return []

    def dispatch(_repository: str, workflow: str, _branch: str, **_kwargs: Any) -> None:
        dispatched.append(workflow)

    monkeypatch.setattr(PUBLISHER, "_workflow_runs", workflow_runs)
    monkeypatch.setattr(PUBLISHER, "_dispatch_workflow", dispatch)

    count = PUBLISHER.dispatch_required_workflows(
        repository=REPOSITORY,
        branch=BRANCH,
        commit_oid=CHILD_SHA,
        trusted_default_oid=HEAD_SHA,
        token="token",
        sleeper=lambda _seconds: None,
    )

    assert PUBLISHER.REQUIRED_WORKFLOWS == (
        "_required.yml",
        "comprehensive-tests.yml",
        "pre-commit.yml",
        "workflow-smoke-test.yml",
    )
    assert authenticated == [(CHILD_SHA, HEAD_SHA)]
    assert dispatched == list(PUBLISHER.REQUIRED_WORKFLOWS[2:])
    assert count == 2


def test_commenter_dispatch_binds_default_branch_and_exact_dependabot_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch_commenter = getattr(PUBLISHER, "dispatch_comment_workflow", None)
    assert callable(dispatch_commenter), "trusted writer must dispatch the commenter as a sibling workflow"
    calls: list[dict[str, Any]] = []

    def dispatch(
        repository: str,
        workflow: str,
        branch: str,
        **kwargs: Any,
    ) -> None:
        calls.append(
            {
                "repository": repository,
                "workflow": workflow,
                "branch": branch,
                "inputs": kwargs.get("inputs"),
            }
        )

    monkeypatch.setattr(PUBLISHER, "_dispatch_workflow", dispatch)

    dispatch_commenter(
        repository=REPOSITORY,
        default_branch="main",
        head_ref=BRANCH,
        commit_oid=CHILD_SHA,
        token="token",
    )

    assert calls == [
        {
            "repository": REPOSITORY,
            "workflow": "comprehensive-test-pr-comments.yml",
            "branch": "main",
            "inputs": {
                "source_head_sha": CHILD_SHA,
                "source_head_branch": BRANCH,
            },
        }
    ]


def test_workflow_dispatch_serializes_exact_string_inputs() -> None:
    recorded: dict[str, Any] = {}

    class FakeResponse:
        status = 204

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

    def opener(request: Any, *, timeout: int) -> FakeResponse:
        recorded["url"] = request.full_url
        recorded["payload"] = json.loads(request.data)
        recorded["timeout"] = timeout
        return FakeResponse()

    PUBLISHER._dispatch_workflow(
        REPOSITORY,
        "comprehensive-test-pr-comments.yml",
        "main",
        token="token",
        opener=opener,
        inputs={
            "source_head_sha": CHILD_SHA,
            "source_head_branch": BRANCH,
        },
    )

    assert recorded == {
        "url": (
            "https://api.github.com/repos/HomericIntelligence/Odyssey/actions/"
            "workflows/comprehensive-test-pr-comments.yml/dispatches"
        ),
        "payload": {
            "ref": "main",
            "inputs": {
                "source_head_sha": CHILD_SHA,
                "source_head_branch": BRANCH,
            },
        },
        "timeout": 30,
    }


@pytest.mark.parametrize(
    "inputs",
    [
        {"source_head_sha": ""},
        {"source_head_sha": CHILD_SHA, 1: BRANCH},
        {"source_head_sha": CHILD_SHA, "source_head_branch": 1},
    ],
)
def test_workflow_dispatch_rejects_non_string_or_empty_inputs(inputs: Any) -> None:
    with pytest.raises(PUBLISHER.PublishError, match="inputs"):
        PUBLISHER._dispatch_workflow(
            REPOSITORY,
            "comprehensive-test-pr-comments.yml",
            "main",
            token="token",
            opener=lambda *_args, **_kwargs: pytest.fail("invalid input must not dispatch"),
            inputs=inputs,
        )


def test_publisher_dispatches_commenter_after_exact_required_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _artifact_root, bundle = _package(tmp_path)
    commenter_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(PUBLISHER, "get_pull_request_head", lambda *_args, **_kwargs: CHILD_SHA)
    monkeypatch.setattr(PUBLISHER, "get_branch_head", lambda *_args, **_kwargs: CHILD_SHA)
    monkeypatch.setattr(PUBLISHER, "validate_published_dependency_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(PUBLISHER, "compare_artifact_to_source", lambda *_args, **_kwargs: ("uv.lock",))
    monkeypatch.setattr(
        PUBLISHER,
        "verify_exact_child",
        lambda **_kwargs: PUBLISHER.CommitResult(
            CHILD_SHA,
            f"https://github.com/{REPOSITORY}/commit/{CHILD_SHA}",
        ),
    )
    monkeypatch.setattr(PUBLISHER, "dispatch_required_workflows", lambda **_kwargs: 4)
    monkeypatch.setattr(
        PUBLISHER,
        "dispatch_comment_workflow",
        lambda **kwargs: commenter_calls.append(kwargs),
        raising=False,
    )

    result = PUBLISHER.publish_dependencies(
        context=_context(),
        bundle=bundle,
        trusted_default_oid=HEAD_SHA,
        token="token",
    )

    assert result == PUBLISHER.PublishResult("published", CHILD_SHA, 4)
    assert commenter_calls == [
        {
            "repository": REPOSITORY,
            "default_branch": "main",
            "head_ref": BRANCH,
            "commit_oid": CHILD_SHA,
            "token": "token",
            "opener": PUBLISHER.urllib.request.urlopen,
        }
    ]


def test_required_workflow_definitions_are_authenticated_at_child_oid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticate = getattr(PUBLISHER, "authenticate_required_workflows", None)
    assert callable(authenticate), "trusted publisher must bind dispatched workflow bytes to the trusted default SHA"
    requested: list[tuple[str, str]] = []

    def content(_repository: str, path: str, oid: str, **_kwargs: Any) -> bytes:
        requested.append((path, oid))
        return f"trusted:{path}".encode()

    monkeypatch.setattr(PUBLISHER, "get_file_at_oid", content)

    authenticate(
        repository=REPOSITORY,
        commit_oid=CHILD_SHA,
        trusted_default_oid=HEAD_SHA,
        token="token",
    )

    expected_paths = [f".github/workflows/{workflow}" for workflow in PUBLISHER.REQUIRED_WORKFLOWS]
    assert requested == [request for path in expected_paths for request in ((path, HEAD_SHA), (path, CHILD_SHA))]


def test_required_workflow_authentication_rejects_child_definition_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticate = getattr(PUBLISHER, "authenticate_required_workflows", None)
    assert callable(authenticate), "trusted publisher must bind dispatched workflow bytes to the trusted default SHA"

    def content(_repository: str, path: str, oid: str, **_kwargs: Any) -> bytes:
        if oid == CHILD_SHA and path.endswith("pre-commit.yml"):
            return b"permissions: write-all\n"
        return f"trusted:{path}".encode()

    monkeypatch.setattr(PUBLISHER, "get_file_at_oid", content)

    with pytest.raises(PUBLISHER.PublishError, match="pre-commit.yml"):
        authenticate(
            repository=REPOSITORY,
            commit_oid=CHILD_SHA,
            trusted_default_oid=HEAD_SHA,
            token="token",
        )


def test_post_graphql_uses_fixed_endpoint_and_bearer_token() -> None:
    recorded: dict[str, Any] = {}
    body = json.dumps(_successful_create_response()).encode()

    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return body

    def opener(request: Any, *, timeout: int) -> FakeResponse:
        recorded["url"] = request.full_url
        recorded["authorization"] = request.get_header("Authorization")
        recorded["timeout"] = timeout
        return FakeResponse()

    result = PUBLISHER.post_graphql({"query": "query { viewer { login } }"}, "secret", opener=opener)

    assert result == _successful_create_response()
    assert recorded == {
        "url": PUBLISHER.GRAPHQL_URL,
        "authorization": "Bearer secret",
        "timeout": 30,
    }
