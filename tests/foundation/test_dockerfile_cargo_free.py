"""Static regression tests asserting cargo is absent from Dockerfiles.

Guards against re-introduction of the cargo apt dependency or cargo install
after the optimization to use pre-built binaries instead.

Follow-up from #3152.
"""

import re
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
