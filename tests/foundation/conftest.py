"""
Shared pytest fixtures for foundation test suite.

This module provides common fixtures used across all foundation tests
to ensure consistency and reduce duplication.

Fixtures:
- repo_root: Real repository root directory
- supporting_dirs: All supporting directory paths
- papers_dir: Papers directory path
- shared_dir: Shared library directory path
- template_dir: Papers template directory path
- expected_template_structure: Expected template directory layout
"""

import re

import pytest
from pathlib import Path
from typing import Dict, List


def assert_pkg_absent(dockerfile: Path, pkg: str) -> None:
    """Assert that a package does not appear as an apt-get dependency or standalone install.

    Checks two patterns:
    - ``apt-get install ... <pkg>`` (package listed as an apt dependency)
    - ``<pkg> install`` (package used as its own installer, e.g. ``cargo install``)

    Args:
        dockerfile: Path to the Dockerfile to inspect.
        pkg: Package name to assert is absent (e.g. ``"cargo"``, ``"rustc"``).

    Raises:
        AssertionError: If either forbidden pattern is found, with a message
            identifying the matched line and the Dockerfile name.
    """
    content = dockerfile.read_text()

    apt_match = re.search(rf"apt-get install[^\n]*\b{re.escape(pkg)}\b", content)
    assert apt_match is None, f"Found {pkg!r} in apt-get install in {dockerfile.name}: {apt_match.group()!r}"

    install_match = re.search(rf"\b{re.escape(pkg)}\s+install\b", content)
    assert install_match is None, f"Found '{pkg} install' in {dockerfile.name}: {install_match.group()!r}"


@pytest.fixture
def repo_root() -> Path:
    """
    Provide the real repository root directory for testing.

    Returns:
        Path to the actual repository root directory
    """
    # Navigate up from tests/foundation/ to repository root
    current_file = Path(__file__)
    return current_file.parent.parent.parent


@pytest.fixture
def papers_dir(repo_root: Path) -> Path:
    """
    Provide the papers/ directory path.

    Args:
        repo_root: Real repository root directory

    Returns:
        Path to papers directory
    """
    return repo_root / "papers"


@pytest.fixture
def shared_dir(repo_root: Path) -> Path:
    """
    Provide the src/odyssey/ directory path.

    Args:
        repo_root: Real repository root directory

    Returns:
        Path to shared directory
    """
    return repo_root / "src" / "odyssey"


@pytest.fixture
def template_dir(papers_dir: Path) -> Path:
    """
    Provide the papers template directory path.

    Args:
        papers_dir: Papers directory path

    Returns:
        Path to template directory
    """
    return papers_dir / "_template"


@pytest.fixture
def shared_core_dir(shared_dir: Path) -> Path:
    """
    Provide the src/odyssey/core/ directory path.

    Args:
        shared_dir: Shared directory path

    Returns:
        Path to src/odyssey/core directory
    """
    return shared_dir / "core"


@pytest.fixture
def shared_training_dir(shared_dir: Path) -> Path:
    """
    Provide the src/odyssey/training/ directory path.

    Args:
        shared_dir: Shared directory path

    Returns:
        Path to src/odyssey/training directory
    """
    return shared_dir / "training"


@pytest.fixture
def shared_data_dir(shared_dir: Path) -> Path:
    """
    Provide the src/odyssey/data/ directory path.

    Args:
        shared_dir: Shared directory path

    Returns:
        Path to src/odyssey/data directory
    """
    return shared_dir / "data"


@pytest.fixture
def shared_utils_dir(shared_dir: Path) -> Path:
    """
    Provide the src/odyssey/utils/ directory path.

    Args:
        shared_dir: Shared directory path

    Returns:
        Path to src/odyssey/utils directory
    """
    return shared_dir / "utils"


@pytest.fixture
def expected_template_structure() -> Dict[str, List[str]]:
    """
    Provide expected template directory structure.

    Returns:
        Dictionary mapping directory paths to expected contents
    """
    return {
        "root": [
            "README.md",
            "src",
            "scripts",
            "tests",
            "data",
            "configs",
            "notebooks",
            "examples",
        ],
        "src": ["__init__.mojo", ".gitkeep"],
        "scripts": [".gitkeep"],
        "tests": ["__init__.mojo", ".gitkeep"],
        "data": ["raw", "processed", "cache"],
        "configs": ["config.yaml", ".gitkeep"],
        "notebooks": [".gitkeep"],
        "examples": [".gitkeep"],
    }


@pytest.fixture
def expected_shared_structure() -> Dict[str, List[str]]:
    """
    Provide expected shared directory structure.

    Returns:
        Dictionary mapping directory paths to expected contents
    """
    return {
        "root": ["README.md", "__init__.mojo", "core", "training", "data", "utils"],
        "core": ["README.md", "__init__.mojo"],
        "training": ["README.md", "__init__.mojo"],
        "data": ["README.md", "__init__.mojo"],
        "utils": ["README.md", "__init__.mojo"],
    }


@pytest.fixture
def benchmarks_dir(repo_root: Path) -> Path:
    """
    Provide the benchmarks/ directory path.

    Args:
        repo_root: Real repository root directory

    Returns:
        Path to benchmarks directory
    """
    return repo_root / "benchmarks"


@pytest.fixture
def docs_dir(repo_root: Path) -> Path:
    """
    Provide the docs/ directory path.

    Args:
        repo_root: Real repository root directory

    Returns:
        Path to docs directory
    """
    return repo_root / "docs"


@pytest.fixture
def agents_dir(repo_root: Path) -> Path:
    """
    Provide the agents/ directory path.

    Args:
        repo_root: Real repository root directory

    Returns:
        Path to agents directory
    """
    return repo_root / "agents"


@pytest.fixture
def tools_dir(repo_root: Path) -> Path:
    """
    Provide the tools/ directory path.

    Args:
        repo_root: Real repository root directory

    Returns:
        Path to tools directory
    """
    return repo_root / "tools"


@pytest.fixture
def configs_dir(repo_root: Path) -> Path:
    """
    Provide the configs/ directory path.

    Args:
        repo_root: Real repository root directory

    Returns:
        Path to configs directory
    """
    return repo_root / "configs"


@pytest.fixture
def supporting_dirs(
    benchmarks_dir: Path,
    docs_dir: Path,
    agents_dir: Path,
    tools_dir: Path,
    configs_dir: Path,
) -> Dict[str, Path]:
    """
    Provide dictionary of all supporting directory paths.

    Args:
        benchmarks_dir: Path to benchmarks directory
        docs_dir: Path to docs directory
        agents_dir: Path to agents directory
        tools_dir: Path to tools directory
        configs_dir: Path to configs directory

    Returns:
        Dictionary mapping directory names to their paths
    """
    return {
        "benchmarks": benchmarks_dir,
        "docs": docs_dir,
        "agents": agents_dir,
        "tools": tools_dir,
        "configs": configs_dir,
    }
