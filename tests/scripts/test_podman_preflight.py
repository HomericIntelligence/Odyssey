"""Regression tests for the local Podman development contract.

The executable tests use a stub ``podman`` binary, so they never inspect or
modify a developer's real Podman machine.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[2]
PREFLIGHT = REPO_ROOT / "scripts" / "podman-preflight.sh"
JUSTFILE = REPO_ROOT / "justfile"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
COMPREHENSIVE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "comprehensive-tests.yml"
SETUP_JUST_ACTION = "extractions/setup-just@53165ef7e734c5c07cb06b3c8e7b647c5aa16db3"
SETUP_JUST_VERSION = "1.36.0"


def _write_podman_stub(tmp_path: Path) -> Path:
    """Create a configurable Podman CLI double and return its bin directory."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    podman = bin_dir / "podman"
    podman.write_text(
        f"""#!{sys.executable}
import json
import os
import sys

args = sys.argv[1:]
if call_log := os.environ.get("STUB_CALL_LOG"):
    with open(call_log, "a", encoding="utf-8") as stream:
        stream.write(" ".join(args) + "\\n")

if args == ["--version"]:
    print("podman version 5.6.0")
elif args == ["compose", "version"]:
    if os.environ.get("STUB_COMPOSE_OK", "1") == "1":
        print("podman-compose version 1.4.0")
    else:
        raise SystemExit(125)
elif args == ["compose", "config"]:
    if os.environ.get("STUB_COMPOSE_CONFIG_OK", "1") != "1":
        raise SystemExit(125)
    memory = os.environ.get(
        "STUB_RENDERED_MEM_LIMIT", os.environ.get("ODYSSEY_MEM_LIMIT", "14g")
    )
    cpus = os.environ.get(
        "STUB_RENDERED_CPU_LIMIT", os.environ.get("ODYSSEY_CPU_LIMIT", "6.0")
    )
    for service in ("odyssey-dev", "odyssey-ci", "odyssey-prod"):
        key = service.upper().replace("-", "_")
        service_memory = os.environ.get(
            f"STUB_RENDERED_{{key}}_MEM_LIMIT", memory
        )
        service_cpus = os.environ.get(
            f"STUB_RENDERED_{{key}}_CPU_LIMIT", cpus
        )
        if service == "odyssey-dev":
            print("services:")
        print(f"  {{service}}:")
        print(f"    cpus: '{{service_cpus}}'")
        print(f"    mem_limit: '{{service_memory}}'")
    if env_log := os.environ.get("STUB_ENV_LOG"):
        with open(env_log, "a", encoding="utf-8") as stream:
            stream.write(
                "|".join(
                    [
                        os.environ.get("USER_ID", ""),
                        os.environ.get("GROUP_ID", ""),
                        os.environ.get("BUILD_PARALLELISM", ""),
                        os.environ.get("ODYSSEY_MEM_LIMIT", ""),
                        os.environ.get("ODYSSEY_CPU_LIMIT", ""),
                    ]
                )
                + "\\n"
            )
elif args[:2] == ["machine", "inspect"]:
    if os.environ.get("STUB_MACHINE_PRESENT", "1") != "1":
        raise SystemExit(125)
    print(
        "|".join(
            [
                os.environ.get("STUB_MACHINE_NAME", "podman-machine-default"),
                os.environ.get("STUB_MACHINE_STATE", "running"),
                os.environ.get("STUB_MACHINE_CPUS", "6"),
                os.environ.get("STUB_MACHINE_MEMORY_MIB", "16384"),
            ]
        )
    )
elif args == ["info", "--format", "json"]:
    if os.environ.get("STUB_INFO_OK", "1") != "1":
        raise SystemExit(125)
    print(
        json.dumps(
            {{
                "host": {{
                    "cpus": int(os.environ.get("STUB_HOST_CPUS", "6")),
                    "memTotal": int(
                        os.environ.get("STUB_HOST_MEMORY_BYTES", str(16 * 1024**3))
                    ),
                }}
            }}
        )
    )
elif args == ["info"]:
    if os.environ.get("STUB_INFO_OK", "1") != "1":
        raise SystemExit(125)
    print("host: stub")
else:
    print(f"unexpected podman arguments: {{args!r}}", file=sys.stderr)
    raise SystemExit(64)
""",
        encoding="utf-8",
    )
    podman.chmod(podman.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def _run_preflight(
    tmp_path: Path,
    *,
    include_podman: bool = True,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the real preflight script against a configurable Podman stub."""
    bin_dir = _write_podman_stub(tmp_path) if include_podman else tmp_path / "empty-bin"
    bin_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}" if include_podman else str(bin_dir)
    env.pop("ODYSSEY_MEM_LIMIT", None)
    env.pop("BUILD_PARALLELISM", None)
    env.pop("ODYSSEY_PODMAN_MIN_CPUS", None)
    env.pop("ODYSSEY_PODMAN_MIN_MEMORY_GIB", None)
    env.pop("ODYSSEY_CPU_LIMIT", None)
    env.pop("USER_ID", None)
    env.pop("GROUP_ID", None)
    env.pop("USER_NAME", None)
    env["USER_ID"] = "1000"
    env["GROUP_ID"] = "1000"
    if overrides:
        env.update(overrides)

    return subprocess.run(
        ["/bin/bash", str(PREFLIGHT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_just_preflight(
    tmp_path: Path,
    *,
    dotenv: str,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``just podman-preflight`` from a copied project with a real .env."""
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(JUSTFILE, project / "justfile")
    shutil.copy2(COMPOSE_FILE, project / "docker-compose.yml")
    shutil.copy2(PREFLIGHT, scripts / PREFLIGHT.name)
    (project / ".env").write_text(dotenv, encoding="utf-8")

    bin_dir = _write_podman_stub(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    for variable in (
        "USER_ID",
        "GROUP_ID",
        "BUILD_PARALLELISM",
        "ODYSSEY_MEM_LIMIT",
        "ODYSSEY_CPU_LIMIT",
    ):
        env.pop(variable, None)
    if overrides:
        env.update(overrides)

    return subprocess.run(
        ["just", "podman-preflight"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _recipe_header(justfile: str, recipe: str) -> str:
    """Return the header line for one public Just recipe."""
    match = re.search(rf"^{re.escape(recipe)}(?: [^:]*)?:[^\n]*$", justfile, re.MULTILINE)
    assert match is not None, f"missing Just recipe: {recipe}"
    return match.group(0)


def test_preflight_is_an_executable_script() -> None:
    """The public Just recipe must delegate to a checked-in executable."""
    assert PREFLIGHT.is_file()
    assert PREFLIGHT.stat().st_mode & stat.S_IXUSR


def test_preflight_fails_when_podman_is_missing(tmp_path: Path) -> None:
    """A missing runtime must fail before a container command is attempted."""
    result = _run_preflight(tmp_path, include_podman=False)

    assert result.returncode != 0
    assert "Podman is not installed" in result.stderr


def test_preflight_fails_when_compose_provider_is_missing(tmp_path: Path) -> None:
    """Podman without a Compose provider cannot run the development recipes."""
    result = _run_preflight(
        tmp_path,
        overrides={"STUB_COMPOSE_OK": "0"},
    )

    assert result.returncode != 0
    assert "Podman Compose is unavailable" in result.stderr


def test_stopped_default_machine_does_not_mask_reachable_active_engine(
    tmp_path: Path,
) -> None:
    """Resources come from the reachable connection, not a stopped default VM."""
    call_log = tmp_path / "podman-calls.log"
    result = _run_preflight(
        tmp_path,
        overrides={
            "STUB_MACHINE_STATE": "stopped",
            "STUB_MACHINE_CPUS": "1",
            "STUB_MACHINE_MEMORY_MIB": "1024",
            "STUB_CALL_LOG": str(call_log),
        },
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert "info --format json" in calls
    assert not any(call.startswith("machine inspect") for call in calls)


@pytest.mark.parametrize(
    ("cpus", "memory_mib"),
    [
        ("5", "16384"),
        ("6", "8192"),
    ],
)
def test_preflight_fails_when_active_engine_is_below_default_resources(
    tmp_path: Path,
    cpus: str,
    memory_mib: str,
) -> None:
    """The default profile requires the Compose-aligned 6 CPU / 16 GiB VM."""
    result = _run_preflight(
        tmp_path,
        overrides={
            "STUB_HOST_CPUS": cpus,
            "STUB_HOST_MEMORY_BYTES": str(int(memory_mib) * 1024**2),
        },
    )

    assert result.returncode != 0
    assert "active Podman engine" in result.stderr
    assert "podman system connection list" in result.stderr


def test_preflight_rejects_implicit_low_memory_profile(tmp_path: Path) -> None:
    """An 8 GiB VM is unsafe unless both constrained-profile knobs are explicit."""
    result = _run_preflight(
        tmp_path,
        overrides={
            "STUB_HOST_MEMORY_BYTES": str(8 * 1024**3),
            "BUILD_PARALLELISM": "1",
        },
    )

    assert result.returncode != 0
    assert "ODYSSEY_MEM_LIMIT=7g" in result.stderr
    assert "BUILD_PARALLELISM=1" in result.stderr


def test_preflight_accepts_explicit_constrained_profile(tmp_path: Path) -> None:
    """The documented serial 7 GiB Compose limit may run on an 8 GiB VM."""
    result = _run_preflight(
        tmp_path,
        overrides={
            "STUB_HOST_MEMORY_BYTES": str(8 * 1024**3),
            "ODYSSEY_MEM_LIMIT": "7g",
            "BUILD_PARALLELISM": "1",
        },
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("usable_memory_mib", [7907, 7680])
def test_constrained_profile_accepts_bounded_guest_reservation(
    tmp_path: Path,
    usable_memory_mib: int,
) -> None:
    """An 8-GiB nominal VM may reserve at most 512 MiB from Host.MemTotal."""
    result = _run_preflight(
        tmp_path,
        overrides={
            "STUB_HOST_MEMORY_BYTES": str(usable_memory_mib * 1024**2),
            "ODYSSEY_MEM_LIMIT": "7g",
            "BUILD_PARALLELISM": "1",
        },
    )

    assert result.returncode == 0, result.stderr


def test_constrained_profile_rejects_excess_guest_reservation(
    tmp_path: Path,
) -> None:
    """A usable Host.MemTotal below 7680 MiB exceeds the reservation budget."""
    result = _run_preflight(
        tmp_path,
        overrides={
            "STUB_HOST_MEMORY_BYTES": str(7679 * 1024**2),
            "ODYSSEY_MEM_LIMIT": "7g",
            "BUILD_PARALLELISM": "1",
        },
    )

    assert result.returncode != 0
    assert "8 GiB nominal configured capacity" in result.stderr
    assert "Host.MemTotal of at least 7680 MiB" in result.stderr


def test_default_profile_accepts_usable_memory_boundary(tmp_path: Path) -> None:
    """A 16-GiB nominal VM may reserve exactly 512 MiB from Host.MemTotal."""
    result = _run_preflight(
        tmp_path,
        overrides={"STUB_HOST_MEMORY_BYTES": str(15872 * 1024**2)},
    )

    assert result.returncode == 0, result.stderr


def test_default_profile_rejects_below_usable_memory_boundary(
    tmp_path: Path,
) -> None:
    """A default profile below 15872 MiB usable memory fails closed."""
    result = _run_preflight(
        tmp_path,
        overrides={"STUB_HOST_MEMORY_BYTES": str(15871 * 1024**2)},
    )

    assert result.returncode != 0
    assert "16 GiB nominal configured capacity" in result.stderr
    assert "Host.MemTotal of at least 15872 MiB" in result.stderr


def test_preflight_rejects_undocumented_resource_profile(tmp_path: Path) -> None:
    """Only the reviewed default and constrained resource profiles are valid."""
    result = _run_preflight(
        tmp_path,
        overrides={
            "ODYSSEY_MEM_LIMIT": "10g",
            "BUILD_PARALLELISM": "2",
        },
    )

    assert result.returncode != 0
    assert "supported resource profile" in result.stderr


@pytest.mark.parametrize("memory", ["7g", "7G", "7gb", "7168m", "7516192768"])
def test_preflight_normalizes_equivalent_constrained_memory_units(
    tmp_path: Path,
    memory: str,
) -> None:
    """Compose-equivalent memory spellings select the same safe profile."""
    result = _run_preflight(
        tmp_path,
        overrides={
            "STUB_HOST_MEMORY_BYTES": str(8 * 1024**3),
            "ODYSSEY_MEM_LIMIT": memory,
            "BUILD_PARALLELISM": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "constrained profile" in result.stdout


@pytest.mark.parametrize(
    ("parallelism", "diagnostic"),
    [
        ("0", "BUILD_PARALLELISM must be a positive integer"),
        ("many", "BUILD_PARALLELISM must be a positive integer"),
        ("7", "cannot exceed the effective Compose CPU limit"),
    ],
)
def test_preflight_rejects_invalid_or_oversized_build_parallelism(
    tmp_path: Path,
    parallelism: str,
    diagnostic: str,
) -> None:
    """Build concurrency must be positive and fit the checked runtime."""
    result = _run_preflight(
        tmp_path,
        overrides={"BUILD_PARALLELISM": parallelism},
    )

    assert result.returncode != 0
    assert diagnostic in result.stderr


@pytest.mark.parametrize(
    ("overrides", "diagnostic"),
    [
        ({"USER_ID": "0"}, "USER_ID must be a positive integer"),
        ({"GROUP_ID": "staff"}, "GROUP_ID must be a positive integer"),
        (
            {"ODYSSEY_MEM_LIMIT": "a-lot"},
            "ODYSSEY_MEM_LIMIT must be a positive Compose byte value",
        ),
        (
            {"ODYSSEY_CPU_LIMIT": "0"},
            "ODYSSEY_CPU_LIMIT must be a positive number",
        ),
    ],
)
def test_preflight_rejects_invalid_identity_and_compose_resource_values(
    tmp_path: Path,
    overrides: dict[str, str],
    diagnostic: str,
) -> None:
    """Invalid interpolation values fail before Compose consumes them."""
    result = _run_preflight(tmp_path, overrides=overrides)

    assert result.returncode != 0
    assert diagnostic in result.stderr


def test_preflight_requires_reachable_engine(tmp_path: Path) -> None:
    """A correctly sized VM does not pass when its engine is unreachable."""
    result = _run_preflight(
        tmp_path,
        overrides={"STUB_INFO_OK": "0"},
    )

    assert result.returncode != 0
    assert "Podman engine is not reachable" in result.stderr
    assert "podman system connection list" in result.stderr


def test_preflight_rejects_compose_memory_quota_above_active_runtime(
    tmp_path: Path,
) -> None:
    """The effective container memory quota must fit the connected engine."""
    result = _run_preflight(
        tmp_path,
        overrides={"STUB_HOST_MEMORY_BYTES": str(13 * 1024**3)},
    )

    assert result.returncode != 0
    assert "Compose memory limit" in result.stderr
    assert "exceeds active Podman engine memory" in result.stderr


def test_preflight_rejects_compose_cpu_quota_above_active_runtime(
    tmp_path: Path,
) -> None:
    """The effective container CPU quota must fit the connected engine."""
    result = _run_preflight(
        tmp_path,
        overrides={"STUB_HOST_CPUS": "5"},
    )

    assert result.returncode != 0
    assert "Compose CPU limit" in result.stderr
    assert "exceeds active Podman engine CPUs" in result.stderr


def test_preflight_validates_compose_config_after_engine_readiness(
    tmp_path: Path,
) -> None:
    """The resolved Compose model is checked only after the engine responds."""
    call_log = tmp_path / "podman-calls.log"
    result = _run_preflight(
        tmp_path,
        overrides={
            "STUB_CALL_LOG": str(call_log),
            "STUB_COMPOSE_CONFIG_OK": "0",
        },
    )

    assert result.returncode != 0
    assert "Podman Compose configuration is invalid" in result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls.index("info --format json") < calls.index("compose config")


@pytest.mark.parametrize(
    ("overrides", "diagnostic"),
    [
        (
            {"STUB_RENDERED_MEM_LIMIT": "7g"},
            "rendered Compose memory limit",
        ),
        (
            {"STUB_RENDERED_ODYSSEY_CI_CPU_LIMIT": "5"},
            "rendered Compose CPU limit",
        ),
        (
            {"STUB_RENDERED_ODYSSEY_PROD_MEM_LIMIT": ""},
            "rendered Compose memory limit",
        ),
    ],
)
def test_preflight_fails_when_rendered_compose_resources_contradict_environment(
    tmp_path: Path,
    overrides: dict[str, str],
    diagnostic: str,
) -> None:
    """The rendered service model, not source-text assumptions, is enforced."""
    result = _run_preflight(tmp_path, overrides=overrides)

    assert result.returncode != 0
    assert diagnostic in result.stderr


def test_preflight_checks_native_linux_host_resources(tmp_path: Path) -> None:
    """Hosts without a Podman VM use engine-reported CPU and memory capacity."""
    result = _run_preflight(
        tmp_path,
        overrides={
            "STUB_MACHINE_PRESENT": "0",
            "STUB_HOST_CPUS": "4",
            "STUB_HOST_MEMORY_BYTES": str(32 * 1024**3),
        },
    )

    assert result.returncode != 0
    assert "exceeds active Podman engine CPUs" in result.stderr


def test_preflight_passes_for_default_resources(tmp_path: Path) -> None:
    """A reachable engine with the default resource profile passes."""
    result = _run_preflight(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Podman preflight passed" in result.stdout


def test_just_loads_dotenv_once_for_preflight_and_compose(tmp_path: Path) -> None:
    """Just exports one .env-derived contract to preflight and Compose."""
    env_log = tmp_path / "compose-environment.log"
    result = _run_just_preflight(
        tmp_path,
        dotenv=("USER_ID=1234\nGROUP_ID=2345\nBUILD_PARALLELISM=1\nODYSSEY_MEM_LIMIT=7168m\nODYSSEY_CPU_LIMIT=6\n"),
        overrides={
            "STUB_ENV_LOG": str(env_log),
            "STUB_HOST_MEMORY_BYTES": str(8 * 1024**3),
        },
    )

    assert result.returncode == 0, result.stderr
    assert env_log.read_text(encoding="utf-8") == "1234|2345|1|7168m|6\n"
    assert "constrained profile" in result.stdout


def test_python_ci_installs_pinned_just_before_running_integration_tests() -> None:
    """The Python producer must provide the CLI exercised by its test suite."""
    workflow = yaml.safe_load(COMPREHENSIVE_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test-python"]["steps"]
    setup_index = next(index for index, step in enumerate(steps) if step.get("uses") == SETUP_JUST_ACTION)
    test_index = next(index for index, step in enumerate(steps) if step.get("name") == "Run Python tests")

    assert setup_index < test_index
    assert steps[setup_index].get("with", {}).get("just-version") == SETUP_JUST_VERSION


def test_just_container_entry_recipes_depend_on_preflight() -> None:
    """Every local entry point must fail fast through the shared preflight."""
    justfile = JUSTFILE.read_text(encoding="utf-8")

    recipes = (
        "podman-up",
        "podman-build",
        "podman-rebuild",
        "podman-build-ci",
        "podman-test-image",
        "podman-run-tests",
        "podman-run-shell",
        "ci-podman-build",
        "ci-podman-validate",
        "shell",
    )
    for recipe in recipes:
        assert "podman-preflight" in _recipe_header(justfile, recipe)


def test_non_compute_podman_recipes_are_not_needlessly_gated() -> None:
    """Inspection, shutdown, and publishing controls stay independently usable."""
    justfile = JUSTFILE.read_text(encoding="utf-8")

    for recipe in (
        "podman-down",
        "podman-logs",
        "podman-status",
        "podman-push",
    ):
        assert "podman-preflight" not in _recipe_header(justfile, recipe)


def test_just_exports_one_shared_container_identity() -> None:
    """Compose calls consume globally exported identity values, not local copies."""
    justfile = JUSTFILE.read_text(encoding="utf-8")

    assert re.search(r"^set dotenv-load$", justfile, re.MULTILINE)
    for variable in (
        "USER_ID",
        "GROUP_ID",
        "USER_NAME",
        "BUILD_PARALLELISM",
        "ODYSSEY_MEM_LIMIT",
        "ODYSSEY_CPU_LIMIT",
    ):
        assert re.search(rf"^export {variable} :=", justfile, re.MULTILINE)

    assert 'export USER_ID="' not in justfile
    assert "USER_ID={{USER_ID}}" not in justfile
    assert "-e USER_ID=" not in justfile


def test_local_container_username_is_fixed_to_dev() -> None:
    """The hard-coded Dockerfile sudo policy cannot drift from Compose."""
    justfile = JUSTFILE.read_text(encoding="utf-8")
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    env_example = ENV_EXAMPLE.read_text(encoding="utf-8")

    assert re.search(r'^export USER_NAME := "dev"$', justfile, re.MULTILINE)
    assert 'env_var_or_default("USER_NAME"' not in justfile
    assert "${USER_NAME" not in compose
    assert "USER_NAME" not in env_example


def test_image_test_recipe_uses_sorted_per_file_mojo_execution() -> None:
    """Container image tests use Mojo 1.0's executable-file interface."""
    justfile = JUSTFILE.read_text(encoding="utf-8")
    start = justfile.index("podman-run-tests target=")
    end = justfile.index("\n# Interactive shell in container image", start)
    recipe = justfile[start:end]

    assert "mojo test" not in recipe
    assert "find tests" in recipe
    assert "| sort" in recipe
    assert "No Mojo test files found" in recipe
    assert "uv run mojo" not in recipe
    assert 'mojo --Werror -I src -I . "$test_file"' in recipe


def test_image_smoke_invokes_the_installed_mojo_directly() -> None:
    """Published images expose Mojo on PATH without depending on uv at runtime."""
    justfile = JUSTFILE.read_text(encoding="utf-8")
    start = justfile.index("podman-test-image target=")
    end = justfile.index("\n# Run tests in container image", start)
    recipe = justfile[start:end]

    assert "uv run mojo" not in recipe
    assert " mojo --version" in recipe


def test_compose_propagates_build_parallelism_to_all_build_services() -> None:
    """Every Compose service receives the bounded build concurrency setting."""
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    services = compose["services"]

    for service_name in ("odyssey-dev", "odyssey-ci", "odyssey-prod"):
        environment = services[service_name]["environment"]
        assert "BUILD_PARALLELISM=${BUILD_PARALLELISM:-4}" in environment


def test_compose_services_share_the_preflighted_resource_contract() -> None:
    """Every service uses the same Just-exported CPU and memory limits."""
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))

    for service_name in ("odyssey-dev", "odyssey-ci", "odyssey-prod"):
        service = compose["services"][service_name]
        assert service["mem_limit"] == "${ODYSSEY_MEM_LIMIT:-14g}"
        assert service["cpus"] == "${ODYSSEY_CPU_LIMIT:-6.0}"


def test_example_environment_exposes_only_supported_resource_profiles() -> None:
    """The template gives complete, internally consistent profile selections."""
    env_example = ENV_EXAMPLE.read_text(encoding="utf-8")
    assignments = set(re.findall(r"^# ([A-Z_]+=\S+)$", env_example, re.MULTILINE))

    assert {
        "BUILD_PARALLELISM=4",
        "ODYSSEY_MEM_LIMIT=14g",
        "ODYSSEY_CPU_LIMIT=6.0",
    } <= assignments
    assert {
        "BUILD_PARALLELISM=1",
        "ODYSSEY_MEM_LIMIT=7g",
        "ODYSSEY_CPU_LIMIT=6.0",
    } <= assignments
    assert "ODYSSEY_PODMAN_MIN_CPUS" not in env_example
    assert "ODYSSEY_PODMAN_MIN_MEMORY_GIB" not in env_example


def test_example_environment_does_not_override_host_identity_or_advertise_native() -> None:
    """Copying the example file preserves auto-detection and Podman-only routing."""
    env_example = ENV_EXAMPLE.read_text(encoding="utf-8")

    assert not re.search(r"^(?:USER_ID|GROUP_ID)=\d+$", env_example, re.MULTILINE)
    assert "NATIVE" not in env_example
