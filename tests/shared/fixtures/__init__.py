#!/usr/bin/env python3
"""Python test fixtures for pytest integration.

This module provides pytest fixtures and utilities for Python-based tests,
including markdown validation, link checking, and file structure validation.

Key fixtures:
- temp_dir: Temporary directory fixture with automatic cleanup
- mock_config_file: Create temporary config files
- Markdown validation utilities
- Link checking helpers
- Repository structure validators

Use these fixtures with pytest's dependency injection:
    def test_something(temp_dir):
        # temp_dir is automatically created and cleaned up
        config_path = temp_dir / "config.yaml"
        ...
"""

import json
import tempfile
import yaml
from pathlib import Path
from typing import Callable, Dict, List, Optional, Generator
import pytest


# ============================================================================
# Temporary File Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory that's automatically cleaned up.

    Yields:
        Path to temporary directory.

    Example:
        ```mojo
        ef test_config_loading(temp_dir):
            config_path = temp_dir / "config.yaml"
            config_path.write_text("key: value")
            # Directory cleaned up automatically after test
        ```
    """
    with tempfile.TemporaryDirectory(prefix="ml_odyssey_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config_file(temp_dir: Path) -> Callable:
    """Factory fixture for creating config files.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Function to create config files.

    Example:
        ```mojo
        ef test_yaml_loading(mock_config_file):
            config_path = mock_config_file("config.yaml", {"key": "value"})
            # Load and test config
        ```
    """

    def _create_config(filename: str, content: Dict) -> Path:
        """Create config file with given content.

        Args:
            filename: Name of config file.
            content: Dictionary to write as YAML/JSON.

        Returns:
            Path to created file.
        """
        file_path = temp_dir / filename

        if filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(file_path, "w") as f:
                yaml.dump(content, f)
        elif filename.endswith(".json"):
            with open(file_path, "w") as f:
                json.dump(content, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {filename}")

        return file_path

    return _create_config


@pytest.fixture
def mock_text_file(temp_dir: Path) -> Callable:
    """Factory fixture for creating text files.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Function to create text files.

    Example:
        ```mojo
        ef test_file_reading(mock_text_file):
            file_path = mock_text_file("data.txt", ["line1", "line2"])
            # Read and test file
        ```
    """

    def _create_text_file(filename: str, lines: List[str]) -> Path:
        """Create text file with given lines.

        Args:
            filename: Name of file.
            lines: List of lines to write.

        Returns:
            Path to created file.
        """
        file_path = temp_dir / filename
        with open(file_path, "w") as f:
            f.write("\n".join(lines))
        return file_path

    return _create_text_file


# ============================================================================
# Markdown Validation Utilities
# ============================================================================


def validate_markdown_links(file_path: Path) -> List[str]:
    """Validate links in markdown file.

    Args:
        file_path: Path to markdown file.

    Returns:
        List of broken link descriptions (empty if all valid).

    Example:
        ```mojo
        roken_links = validate_markdown_links(Path("README.md"))
        assert len(broken_links) == 0, f"Broken links: {broken_links}"
        ```
    """
    import re

    broken_links = []
    content = file_path.read_text()

    # Find all markdown links: [text](url)
    link_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
    matches = re.finditer(link_pattern, content)

    for match in matches:
        link_text = match.group(1)
        link_url = match.group(2)

        # Skip external URLs (http/https)
        if link_url.startswith(("http://", "https://")):
            continue

        # Skip anchors
        if link_url.startswith("#"):
            continue

        # Check if local file exists
        if link_url.startswith("/"):
            # Absolute path from repo root
            # Need to find repo root
            repo_root = find_repo_root(file_path)
            if repo_root:
                target_path = repo_root / link_url.lstrip("/")
            else:
                broken_links.append(f"Cannot resolve absolute link: {link_url} (repo root not found)")
                continue
        else:
            # Relative path
            target_path = (file_path.parent / link_url).resolve()

        # Remove anchor if present
        target_path_str = str(target_path).split("#")[0]
        target_path = Path(target_path_str)

        if not target_path.exists():
            broken_links.append(f"Broken link '{link_text}' -> {link_url}")

    return broken_links


def validate_markdown_code_blocks(file_path: Path) -> List[str]:
    """Validate code blocks in markdown file.

    Checks that all code blocks have language specified and are properly formatted.

    Args:
        file_path: Path to markdown file.

    Returns:
        List of validation errors (empty if all valid).

    Example:
        ```mojo
        rrors = validate_markdown_code_blocks(Path("README.md"))
        assert len(errors) == 0, f"Code block errors: {errors}"
        ```
    """
    errors = []
    content = file_path.read_text()
    lines = content.split("\n")

    in_code_block = False
    code_block_line = 0

    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            if not in_code_block:
                # Starting code block
                in_code_block = True
                code_block_line = i

                # Check if language is specified
                if line.strip() == "```":
                    errors.append(f"Line {i}: Code block missing language specification")
            else:
                # Ending code block
                in_code_block = False

    if in_code_block:
        errors.append(f"Line {code_block_line}: Unclosed code block")

    return errors


def find_repo_root(start_path: Path) -> Optional[Path]:
    """Find repository root by looking for .git directory.

    Args:
        start_path: Path to start searching from.

    Returns:
        Path to repository root, or None if not found.
    """
    current = start_path if start_path.is_dir() else start_path.parent

    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    return None


# ============================================================================
# File Structure Validation
# ============================================================================


def validate_directory_structure(root_dir: Path, expected_structure: Dict) -> List[str]:
    """Validate directory structure matches expected layout.

    Args:
        root_dir: Root directory to validate.
        expected_structure: Dict describing expected structure.
            Keys are directory/file names, values are:
            - None: file must exist
            - Dict: subdirectory with nested structure

    Returns:
        List of validation errors (empty if valid).

    Example:
        ```mojo
        xpected = {
            "README.md": None,
            "src": {
                "__init__.py": None,
                "main.py": None,
            }
        }
        errors = validate_directory_structure(Path("project"), expected)
        ```
    """
    errors = []

    def _check_structure(current_dir: Path, structure: Dict, path_prefix: str = ""):
        for name, value in structure.items():
            full_path = current_dir / name
            display_path = f"{path_prefix}/{name}" if path_prefix else name

            if value is None:
                # File should exist
                if not full_path.exists():
                    errors.append(f"Missing file: {display_path}")
                elif not full_path.is_file():
                    errors.append(f"Expected file but found directory: {display_path}")
            else:
                # Directory with nested structure
                if not full_path.exists():
                    errors.append(f"Missing directory: {display_path}")
                elif not full_path.is_dir():
                    errors.append(f"Expected directory but found file: {display_path}")
                else:
                    _check_structure(full_path, value, display_path)

    _check_structure(root_dir, expected_structure)
    return errors


def find_files_by_pattern(root_dir: Path, pattern: str) -> List[Path]:
    """Find all files matching pattern under root directory.

    Args:
        root_dir: Root directory to search.
        pattern: Glob pattern (e.g., "*.py", "**/*.mojo").

    Returns:
        List of matching file paths.

    Example:
        ```mojo
        ojo_files = find_files_by_pattern(Path("src"), "**/*.mojo")
        ```
    """
    return list(root_dir.glob(pattern))


# ============================================================================
# Configuration Validation Helpers
# ============================================================================


def validate_yaml_file(file_path: Path) -> Optional[str]:
    """Validate YAML file syntax.

    Args:
        file_path: Path to YAML file.

    Returns:
        Error message if invalid, None if valid.

    Example:
        ```mojo
        rror = validate_yaml_file(Path("config.yaml"))
        assert error is None, f"YAML validation failed: {error}"
        ```
    """
    try:
        with open(file_path, "r") as f:
            yaml.safe_load(f)
        return None
    except yaml.YAMLError as e:
        return f"YAML syntax error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"


def validate_json_file(file_path: Path) -> Optional[str]:
    """Validate JSON file syntax.

    Args:
        file_path: Path to JSON file.

    Returns:
        Error message if invalid, None if valid.

    Example:
        ```mojo
        rror = validate_json_file(Path("config.json"))
        assert error is None, f"JSON validation failed: {error}"
        ```
    """
    try:
        with open(file_path, "r") as f:
            json.load(f)
        return None
    except json.JSONDecodeError as e:
        return f"JSON syntax error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"


# ============================================================================
# Test Data Helpers
# ============================================================================


def get_fixtures_dir() -> Path:
    """Get path to fixtures directory.

    Returns:
        Path to tests/shared/fixtures/ directory.

    Example:
        ```mojo
        ixtures_dir = get_fixtures_dir()
        sample_data = fixtures_dir / "images" / "sample.png"
        ```
    """
    # Find fixtures directory relative to this file
    return Path(__file__).parent


def get_test_data_path(filename: str) -> Path:
    """Get path to test data file in fixtures directory.

    Args:
        filename: Name of test data file (can include subdirectory).

    Returns:
        Full path to test data file.

    Example:
        ```mojo
        mage_path = get_test_data_path("images/sample.png")
        ```
    """
    return get_fixtures_dir() / filename


def create_sample_yaml_config() -> Dict:
    """Create sample YAML configuration dictionary.

    Returns:
        Dictionary with sample configuration.

    Example:
        ```mojo
        onfig = create_sample_yaml_config()
        yaml_str = yaml.dump(config)
        ```
    """
    return {
        "model": {
            "name": "TestModel",
            "input_dim": 10,
            "output_dim": 5,
        },
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
        },
    }


def create_sample_json_config() -> Dict:
    """Create sample JSON configuration dictionary.

    Returns:
        Dictionary with sample configuration.

    Example:
        ```mojo
        onfig = create_sample_json_config()
        json_str = json.dumps(config, indent=2)
        ```
    """
    return create_sample_yaml_config()  # Same structure


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_files_equal(file1: Path, file2: Path):
    """Assert two files have identical content.

    Args:
        file1: First file path.
        file2: Second file path.

    Raises:
        AssertionError if files differ.

    Example:
        ```mojo
        ssert_files_equal(expected_output, actual_output)
        ```
    """
    content1 = file1.read_text()
    content2 = file2.read_text()

    assert content1 == content2, (
        f"Files differ:\nFile 1: {file1}\nFile 2: {file2}\nExpected:\n{content1}\nActual:\n{content2}"
    )


def assert_file_contains(file_path: Path, expected_content: str):
    """Assert file contains expected string.

    Args:
        file_path: File to check.
        expected_content: String that should be in file.

    Raises:
        AssertionError if content not found.

    Example:
        ```mojo
        ssert_file_contains(log_file, "Training completed")
        ```
    """
    content = file_path.read_text()
    assert expected_content in content, (
        f"Expected content not found in {file_path}:\nLooking for: {expected_content}\nFile content:\n{content}"
    )


# ============================================================================
# Pytest Markers
# ============================================================================

# Define custom markers for test categorization
# Add these to pytest.ini or pyproject.toml:
#
# [tool.pytest.ini_options]
# markers = [
#     "slow: marks tests as slow (deselect with '-m \"not slow\"')",
#     "integration: marks tests as integration tests",
#     "unit: marks tests as unit tests",
# ]
