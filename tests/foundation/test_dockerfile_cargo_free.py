"""Static regression tests asserting cargo is absent from Dockerfiles.

Guards against re-introduction of the cargo apt dependency or cargo install
after the optimization to use pre-built binaries instead.

Follow-up from #3152.
"""

from pathlib import Path

import pytest

from tests.foundation.helpers import assert_pkg_absent

REPO_ROOT = Path(__file__).parents[2]
DOCKERFILES = [REPO_ROOT / "Dockerfile", REPO_ROOT / "Dockerfile.ci"]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=["Dockerfile", "Dockerfile.ci"])
class TestCargoAbsent:
    """Assert cargo does not appear in apt-get install or as cargo install."""

    def test_cargo_absent(self, dockerfile: Path) -> None:
        """Assert cargo is absent from the Dockerfile using the shared helper.

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        assert_pkg_absent(dockerfile, "cargo")
