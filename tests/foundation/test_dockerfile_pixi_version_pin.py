"""Static regression tests asserting Pixi version is pinned in Dockerfiles.

Guards against re-introduction of unpinned Pixi installations or version drift
after the version pin is documented and fixed.

Follow-up from #3350.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
DOCKERFILES = [REPO_ROOT / "Dockerfile", REPO_ROOT / "Dockerfile.ci"]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=["Dockerfile", "Dockerfile.ci"])
class TestPixiVersionPin:
    """Assert Pixi version is pinned with PIXI_VERSION env var."""

    def test_pixi_version_env_defined(self, dockerfile: Path) -> None:
        """Assert PIXI_VERSION environment variable is set.

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()
        # Look for PIXI_VERSION env var definition
        match = re.search(r"ENV\s+PIXI_VERSION=([0-9]+\.[0-9]+\.[0-9]+)", content)

        if match is None:
            # Skip if Pixi is not installed in this Dockerfile
            if "pixi.sh/install.sh" not in content:
                pytest.skip(f"Pixi not installed in {dockerfile.name}")
            # If Pixi is installed but PIXI_VERSION is not set, fail
            pytest.fail(
                f"PIXI_VERSION env var is not defined in {dockerfile.name}. "
                "Use: ENV PIXI_VERSION=X.Y.Z before pixi.sh/install.sh command"
            )

    def test_pixi_version_env_value_is_semver(self, dockerfile: Path) -> None:
        """Assert PIXI_VERSION has a valid semantic version (X.Y.Z format).

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()

        # Skip if Pixi is not installed in this Dockerfile
        if "pixi.sh/install.sh" not in content:
            pytest.skip(f"Pixi not installed in {dockerfile.name}")

        # Match PIXI_VERSION with version number (semver format)
        match = re.search(r"ENV\s+PIXI_VERSION=([0-9]+\.[0-9]+\.[0-9]+)", content)
        assert match is not None, (
            f"Could not find PIXI_VERSION with version number (X.Y.Z format) "
            f"in {dockerfile.name}. Ensure PIXI_VERSION is followed by a semantic "
            "version like 0.65.0"
        )
        version = match.group(1)
        assert len(version) > 0, f"PIXI_VERSION is empty in {dockerfile.name}"

    def test_pixi_version_used_in_install(self, dockerfile: Path) -> None:
        """Assert PIXI_VERSION env var is used in the install command.

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()

        # Skip if Pixi is not installed in this Dockerfile
        if "pixi.sh/install.sh" not in content:
            pytest.skip(f"Pixi not installed in {dockerfile.name}")

        # Look for PIXI_VERSION variable reference in install
        match = re.search(r"PIXI_VERSION=\$\{PIXI_VERSION\}\s+bash", content)
        assert match is not None, (
            f"PIXI_VERSION env var is not passed to pixi.sh/install.sh "
            f"in {dockerfile.name}. Use: PIXI_VERSION=${{PIXI_VERSION}} bash"
        )
