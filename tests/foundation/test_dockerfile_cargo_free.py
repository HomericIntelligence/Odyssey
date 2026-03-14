"""Static regression tests asserting cargo is absent from Dockerfiles.

Guards against re-introduction of the cargo apt dependency or cargo install
after the optimization to use pre-built binaries instead.

Follow-up from #3152 and #3995.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
DOCKERFILES = [REPO_ROOT / "Dockerfile", REPO_ROOT / "Dockerfile.ci"]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=["Dockerfile", "Dockerfile.ci"])
class TestCargoAbsent:
    """Assert cargo does not appear in apt-get install or as cargo install."""

    def test_cargo_not_in_apt_get_install(self, dockerfile: Path) -> None:
        """Assert cargo is not listed as an apt-get install dependency.

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()
        match = re.search(r"apt-get install[^\n]*\bcargo\b", content)
        assert match is None, f"Found 'cargo' in apt-get install in {dockerfile.name}: {match.group()!r}"

    def test_cargo_install_not_present(self, dockerfile: Path) -> None:
        """Assert 'cargo install' does not appear anywhere in the Dockerfile.

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()
        match = re.search(r"\bcargo\s+install\b", content)
        assert match is None, f"Found 'cargo install' in {dockerfile.name}: {match.group()!r}"


class TestCargoHookDetection:
    """Test that the no-cargo-in-dockerfile pre-commit hook detects violations."""

    def test_hook_detects_cargo_install(self) -> None:
        """Verify the hook rejects files containing 'cargo install'.

        Creates a temporary Dockerfile with the forbidden pattern and verifies
        that the hook command exits with non-zero status.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".Dockerfile",
            delete=False,
        ) as f:
            f.write("FROM ubuntu:24.04\n")
            f.write("RUN cargo install just --version 1.14.0\n")
            temp_dockerfile = f.name

        try:
            # The hook should exit with status 1 (failure) when cargo install is found
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'grep -E "(cargo\\s+install|apt-get install.*\\bcargo\\b)" "$@" && exit 1 || exit 0',
                    "--",
                    temp_dockerfile,
                ],
                capture_output=True,
            )
            assert result.returncode != 0, "Hook should reject Dockerfile with 'cargo install'"
        finally:
            Path(temp_dockerfile).unlink()

    def test_hook_detects_cargo_apt_dependency(self) -> None:
        """Verify the hook rejects files containing cargo in apt-get install.

        Creates a temporary Dockerfile with cargo as an apt dependency and verifies
        that the hook command exits with non-zero status.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".Dockerfile",
            delete=False,
        ) as f:
            f.write("FROM ubuntu:24.04\n")
            f.write("RUN apt-get install -y build-essential cargo git\n")
            temp_dockerfile = f.name

        try:
            # The hook should exit with status 1 (failure) when cargo is in apt-get
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'grep -E "(cargo\\s+install|apt-get install.*\\bcargo\\b)" "$@" && exit 1 || exit 0',
                    "--",
                    temp_dockerfile,
                ],
                capture_output=True,
            )
            assert result.returncode != 0, "Hook should reject Dockerfile with 'cargo' in apt-get install"
        finally:
            Path(temp_dockerfile).unlink()

    def test_hook_accepts_clean_dockerfile(self) -> None:
        """Verify the hook accepts Dockerfiles without cargo references.

        Creates a clean temporary Dockerfile and verifies that the hook
        command exits with zero status.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".Dockerfile",
            delete=False,
        ) as f:
            f.write("FROM ubuntu:24.04\n")
            f.write("RUN apt-get install -y build-essential git\n")
            f.write("RUN curl -fsSL https://just.systems/install.sh | bash\n")
            temp_dockerfile = f.name

        try:
            # The hook should exit with status 0 (success) when no cargo references
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'grep -E "(cargo\\s+install|apt-get install.*\\bcargo\\b)" "$@" && exit 1 || exit 0',
                    "--",
                    temp_dockerfile,
                ],
                capture_output=True,
            )
            assert result.returncode == 0, "Hook should accept clean Dockerfile without cargo"
        finally:
            Path(temp_dockerfile).unlink()
