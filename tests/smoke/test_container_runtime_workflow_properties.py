#!/usr/bin/env python3
"""Regression tests for deterministic Podman runtime setup.

These tests keep the hosted-runner runtime repair fail-closed.  They validate
the shared runtime bootstrap as well as every workflow entry point that builds,
starts, tests, or publishes Odyssey container images.
"""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).parent.parent.parent
RUNTIME_BOOTSTRAP = REPO_ROOT / "scripts" / "ci" / "ensure-podman-runtime.sh"
SETUP_ACTION = REPO_ROOT / ".github" / "actions" / "setup-container" / "action.yml"
CONTAINER_PUBLISH = REPO_ROOT / ".github" / "workflows" / "container-publish.yml"
RELEASE = REPO_ROOT / ".github" / "workflows" / "release.yml"
WORKFLOW_SMOKE = REPO_ROOT / ".github" / "workflows" / "workflow-smoke-test.yml"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
JUSTFILE = REPO_ROOT / "justfile"


def _read(path: Path) -> str:
    assert path.is_file(), f"Required CI file is missing: {path.relative_to(REPO_ROOT)}"
    return path.read_text(encoding="utf-8")


def _job_block(workflow: str, job_id: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_id)}:.*?(?=^  [A-Za-z0-9_-]+:|\Z)",
        workflow,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"Could not find workflow job {job_id!r}"
    return match.group(0)


def _compose_service_block(compose: str, service_name: str) -> str:
    match = re.search(
        rf"^  {re.escape(service_name)}:.*?(?=^  [A-Za-z0-9_-]+:|^[A-Za-z0-9_-]+:|\Z)",
        compose,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"Could not find Compose service {service_name!r}"
    return match.group(0)


def _assert_order(content: str, *needles: str) -> None:
    positions = []
    for needle in needles:
        position = content.find(needle)
        assert position >= 0, f"Expected to find {needle!r}"
        positions.append(position)
    assert positions == sorted(positions), f"Expected this order: {' -> '.join(needles)}"


class TestRuntimeBootstrap:
    """The shared bootstrap must select and verify the compatible crun."""

    def test_uses_strict_rootless_runtime_configuration(self) -> None:
        bootstrap = _read(RUNTIME_BOOTSTRAP)

        assert "set -euo pipefail" in bootstrap
        assert "/usr/local/bin/crun" in bootstrap
        assert "/usr/bin/crun" in bootstrap
        assert "containers.conf.d" in bootstrap
        assert "[engine.runtimes]" in bootstrap
        assert re.search(r"crun\s*=\s*\[", bootstrap)
        assert "XDG_CONFIG_HOME" in bootstrap
        assert "/etc/containers" not in bootstrap
        assert "sudo" not in bootstrap
        assert not re.search(r"\brunc\b", bootstrap)
        assert not re.search(r"\b(?:cp|ln|mv)\b[^\n]*/usr/bin/crun", bootstrap)

    def test_logs_runner_and_runtime_versions(self) -> None:
        bootstrap = _read(RUNTIME_BOOTSTRAP)

        assert "ImageOS" in bootstrap
        assert "ImageVersion" in bootstrap
        assert "podman --version" in bootstrap
        assert re.search(r"/usr/local/bin/crun[\"']?\s+--version", bootstrap)
        assert re.search(r"/usr/bin/crun[\"']?\s+--version", bootstrap)

    def test_requeries_effective_runtime_and_fails_on_a_mismatch(self) -> None:
        bootstrap = _read(RUNTIME_BOOTSTRAP)

        assert bootstrap.count("podman info") >= 2
        assert "OCIRuntime" in bootstrap
        assert re.search(r"\bexit\s+1\b", bootstrap)
        assert "retry" not in bootstrap.lower()
        assert not re.search(r"\bsleep\b", bootstrap)


class TestSetupContainerAction:
    """The reusable setup action must prove Podman and the built image work."""

    def test_bootstraps_runtime_before_starting_podman(self) -> None:
        action = _read(SETUP_ACTION)

        _assert_order(
            action,
            "scripts/ci/ensure-podman-runtime.sh",
            "- name: Start Podman socket",
            "podman compose build",
        )

    def test_socket_startup_has_a_service_fallback_and_readiness_probe(self) -> None:
        action = _read(SETUP_ACTION)

        assert "systemctl --user start podman.socket" in action
        assert "podman system service --time=0" in action
        assert "curl" in action
        assert "--unix-socket" in action
        assert "_ping" in action
        assert re.search(r"\bexit\s+1\b", action)
        _assert_order(action, "_ping", 'echo "DOCKER_HOST=')

    def test_runs_a_real_image_smoke_before_compose_start(self) -> None:
        action = _read(SETUP_ACTION)

        _assert_order(
            action,
            "podman run --rm --entrypoint /bin/true odyssey:dev",
            "podman compose up -d odyssey-dev",
        )

    def test_does_not_make_the_checkout_world_writable(self) -> None:
        action_and_justfile = _read(SETUP_ACTION) + _read(JUSTFILE)

        assert not re.search(r"chmod\s+-R\s+a\+rwX\s+\.", action_and_justfile)

    def test_verifies_mojo_directly_in_the_running_container(self) -> None:
        action = _read(SETUP_ACTION)

        _assert_order(action, "podman compose exec -T odyssey-dev bash -c", "mojo --version")
        assert "podman compose exec -T odyssey-dev uv run mojo --version" not in action

    def test_probes_bind_mounted_workspace_write_access(self) -> None:
        action = _read(SETUP_ACTION)

        _assert_order(
            action,
            "podman compose up -d odyssey-dev",
            "mktemp -d /workspace/.odyssey-write-probe.",
            "mojo --version",
        )


class TestRootlessWorkspaceMapping:
    """Writable bind mounts must preserve the invoking user's UID and GID."""

    def test_writable_services_use_keep_id_user_namespace(self) -> None:
        compose = _read(COMPOSE_FILE)

        for service_name in ("odyssey-dev", "odyssey-ci"):
            service = _compose_service_block(compose, service_name)
            assert 'user: "${USER_ID}:${GROUP_ID}"' in service
            assert 'userns_mode: "keep-id"' in service, (
                f"{service_name} must map the rootless Podman caller to the same "
                "container UID/GID so /workspace can create build metadata"
            )
            assert re.search(r"^\s+- \.:/workspace(?::Z)?\s*$", service, re.MULTILINE)


class TestContainerPublishingWorkflow:
    """Container build and image-test jobs must use the same bootstrap."""

    def test_runtime_bootstrap_is_a_container_affecting_path(self) -> None:
        workflow = _read(CONTAINER_PUBLISH)

        assert workflow.count("'scripts/ci/ensure-podman-runtime.sh'") >= 2

    def test_build_job_bootstraps_immediately_after_checkout(self) -> None:
        job = _job_block(_read(CONTAINER_PUBLISH), "build-and-push")

        _assert_order(
            job,
            "- name: Checkout code",
            "scripts/ci/ensure-podman-runtime.sh",
            "- name: Cache Podman storage",
        )

    def test_build_job_smokes_every_matrix_image_before_push(self) -> None:
        job = _job_block(_read(CONTAINER_PUBLISH), "build-and-push")

        _assert_order(job, "- name: Build image", "podman run --rm", "- name: Push images")
        smoke_start = job.rfind("- name:", 0, job.index("podman run --rm"))
        smoke_end = job.find("- name:", job.index("podman run --rm"))
        smoke_step = job[smoke_start:smoke_end]
        assert "mojo --version" in smoke_step
        assert "id -un" in smoke_step
        assert "pull_request" not in smoke_step

    def test_image_test_job_bootstraps_before_registry_login(self) -> None:
        job = _job_block(_read(CONTAINER_PUBLISH), "test-images")

        _assert_order(
            job,
            "- name: Checkout code",
            "scripts/ci/ensure-podman-runtime.sh",
            "- name: Log in to Container Registry",
        )

    def test_published_images_use_installed_tools_directly(self) -> None:
        job = _job_block(_read(CONTAINER_PUBLISH), "test-images")

        assert "uv run mojo" not in job
        assert "uv run pre-commit" not in job
        assert "mojo --version" in job
        assert "pre-commit --version" in job


class TestReleaseWorkflow:
    """Release publishing must validate the production image before pushing."""

    def test_release_container_checks_use_mojo_directly(self) -> None:
        workflow = _read(RELEASE)

        assert "podman compose exec -T odyssey-dev uv run mojo --version" not in workflow
        assert "podman compose exec -T odyssey-dev mojo --version" in workflow

    def test_publish_job_bootstraps_before_registry_login(self) -> None:
        job = _job_block(_read(RELEASE), "publish-container")

        _assert_order(
            job,
            "- name: Checkout code",
            "scripts/ci/ensure-podman-runtime.sh",
            "- name: Log in to Container Registry",
        )

    def test_publish_job_smokes_release_image_before_push(self) -> None:
        job = _job_block(_read(RELEASE), "publish-container")

        _assert_order(job, "- name: Build release image", "podman run --rm", "- name: Push release images")
        smoke_start = job.rfind("- name:", 0, job.index("podman run --rm"))
        smoke_end = job.find("- name:", job.index("podman run --rm"))
        smoke_step = job[smoke_start:smoke_end]
        assert "mojo --version" in smoke_step
        assert "id -un" in smoke_step


class TestWorkflowSmokeWiring:
    """The property suite must rerun whenever its contract inputs change."""

    def test_paths_cover_runtime_contract_inputs(self) -> None:
        workflow = _read(WORKFLOW_SMOKE)

        expected_paths = {
            ".github/actions/setup-container/action.yml",
            ".github/workflows/container-publish.yml",
            ".github/workflows/release.yml",
            "docker-compose.yml",
            "justfile",
            "scripts/ci/ensure-podman-runtime.sh",
            "tests/smoke/test_container_runtime_workflow_properties.py",
        }
        missing = sorted(path for path in expected_paths if f"'{path}'" not in workflow)
        assert not missing, f"workflow-smoke-test.yml paths omit: {missing}"

    def test_pytest_invocation_includes_runtime_properties(self) -> None:
        workflow = _read(WORKFLOW_SMOKE)

        assert "tests/smoke/test_container_runtime_workflow_properties.py" in workflow
