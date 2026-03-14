"""Static regression tests asserting just tool version is pinned in Dockerfiles.

Guards against re-introduction of unpinned just installations or version drift
after the version pin is documented and fixed.

Follow-up from #3349.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
DOCKERFILES = [REPO_ROOT / "Dockerfile", REPO_ROOT / "Dockerfile.ci"]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=["Dockerfile", "Dockerfile.ci"])
class TestJustVersionPin:
    """Assert just tool version is pinned with --tag flag."""

    def test_just_install_has_tag_flag(self, dockerfile: Path) -> None:
        """Assert just install command includes a --tag version pin.

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()
        # Look for just.systems/install.sh command
        match = re.search(r"just\.systems/install\.sh.*", content)

        if match is None:
            # Skip if just is not installed in this Dockerfile (e.g., Dockerfile.ci)
            pytest.skip(f"just.systems/install.sh not found in {dockerfile.name}")

        install_line = match.group(0)
        assert "--tag" in install_line, (
            f"just install command is missing --tag flag in {dockerfile.name}: {install_line!r}. "
            "Use: curl -fsSL https://just.systems/install.sh | bash -s -- --to /path --tag VERSION"
        )

    def test_just_tag_has_version(self, dockerfile: Path) -> None:
        """Assert --tag flag has an actual version number (not empty).

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()

        # Skip if just is not installed in this Dockerfile
        if "just.systems/install.sh" not in content:
            pytest.skip(f"just.systems/install.sh not found in {dockerfile.name}")

        # Match --tag followed by version number (semver format)
        match = re.search(r"--tag\s+([0-9]+\.[0-9]+\.[0-9]+)", content)
        assert match is not None, (
            f"Could not find --tag with version number (X.Y.Z format) in {dockerfile.name}. "
            "Ensure --tag is followed by a semantic version like 1.14.0"
        )
        version = match.group(1)
        assert len(version) > 0, f"--tag version is empty in {dockerfile.name}"
