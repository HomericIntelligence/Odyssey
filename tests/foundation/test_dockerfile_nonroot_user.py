"""Static regression tests asserting Dockerfiles use non-root users.

Guards against re-introduction of root-only stages in Dockerfiles.
All runtime, CI, and production stages must drop root privileges
to reduce blast radius if the container is compromised.

Follow-up from #5037.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
DOCKERFILES = [REPO_ROOT / "Dockerfile", REPO_ROOT / "Dockerfile.ci"]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=["Dockerfile", "Dockerfile.ci"])
class TestNonRootUser:
    """Assert Dockerfiles create and switch to a non-root user."""

    def test_user_directive_present(self, dockerfile: Path) -> None:
        """Assert at least one USER directive exists.

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()
        match = re.search(r"^USER\s+", content, re.MULTILINE)
        assert match is not None, (
            f"No USER directive found in {dockerfile.name}. "
            "All Dockerfiles must drop root privileges with a USER directive."
        )

    def test_no_root_path_in_env(self, dockerfile: Path) -> None:
        """Assert PATH does not reference /root/.pixi.

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()
        root_path_refs = re.findall(r'ENV\s+PATH="/root/\.pixi', content)
        assert len(root_path_refs) == 0, (
            f"Found {len(root_path_refs)} reference(s) to /root/.pixi in "
            f"PATH in {dockerfile.name}. Use a non-root user home directory "
            "instead (e.g., /home/dev/.pixi)."
        )

    def test_user_creation_exists(self, dockerfile: Path) -> None:
        """Assert a non-root user is created (useradd or adduser).

        Args:
            dockerfile: Path to the Dockerfile under test.
        """
        content = dockerfile.read_text()
        has_useradd = "useradd" in content
        has_adduser = "adduser" in content
        assert has_useradd or has_adduser, (
            f"No user creation command (useradd/adduser) found in {dockerfile.name}. A non-root user must be created."
        )
