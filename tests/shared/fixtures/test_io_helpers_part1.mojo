"""Test suite for io_helpers.mojo test utilities - Part 1.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_io_helpers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Test Coverage (Part 1):
- Tier 1: file_exists, dir_exists, create_temp_dir (6 tests)
- Tier 2: create_mock_config, create_mock_text_file - file creation (2 tests)
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
)
from tests.shared.fixtures.io_helpers import (
    file_exists,
    dir_exists,
    create_temp_dir,
    cleanup_temp_dir,
    create_mock_config,
    create_mock_text_file,
    temp_file_path,
)

# Use /tmp/ prefix for all temp files to avoid Docker permission issues
# when the source tree is mounted read-only.
comptime _TMP_PREFIX = "/tmp/ml_odyssey_test_io_part1_"


fn _make_tmp_dir(suffix: String) raises -> String:
    """Create a unique temp directory under /tmp/ using Mojo builtins."""
    var path = _TMP_PREFIX + suffix
    from os import mkdir

    try:
        mkdir(path)
    except:
        # Directory may already exist from a previous failed run
        pass
    return path


fn _cleanup_tmp_dir(path: String) raises:
    """Remove a temp directory under /tmp/ using Python shutil (safe)."""
    # Only clean paths under /tmp/ for safety
    if not path.startswith("/tmp/"):
        raise Error("Refusing to clean non-/tmp/ path: " + path)
    from python import Python

    var shutil = Python.import_module("shutil")
    try:
        shutil.rmtree(path)
    except:
        pass  # Ignore cleanup errors


# ============================================================================
# Tier 1 Tests - Foundation Functions
# ============================================================================


fn test_file_exists_positive() raises:
    """Test file_exists returns True for existing file."""
    print("TEST: test_file_exists_positive")

    # Create a temporary file in /tmp/ to test with
    var temp_dir = _make_tmp_dir("file_exists_pos")
    var test_file = temp_dir + "/test_file.txt"

    # Create the file directly in /tmp/
    with open(test_file, "w") as f:
        f.write("test content")

    # Test: file should exist
    var exists = file_exists(test_file)
    assert_true(exists, "File should exist after creation")

    # Cleanup
    _cleanup_tmp_dir(temp_dir)
    print("PASS: file_exists returns True for existing file")


fn test_file_exists_negative() raises:
    """Test file_exists returns False for non-existent file."""
    print("TEST: test_file_exists_negative")

    # Test with a path that definitely doesn't exist
    var nonexistent = "/tmp/ml_odyssey_nonexistent_file_12345.txt"

    # Test: file should NOT exist
    var exists = file_exists(nonexistent)
    assert_false(exists, "File should not exist")

    print("PASS: file_exists returns False for non-existent file")


fn test_dir_exists_positive() raises:
    """Test dir_exists returns True for existing directory."""
    print("TEST: test_dir_exists_positive")

    # Create a temporary directory under /tmp/
    var temp_dir = _make_tmp_dir("dir_exists_pos")

    # Test: directory should exist
    var exists = dir_exists(temp_dir)
    assert_true(exists, "Directory should exist after creation")

    # Cleanup
    _cleanup_tmp_dir(temp_dir)
    print("PASS: dir_exists returns True for existing directory")


fn test_dir_exists_negative() raises:
    """Test dir_exists returns False for non-existent directory."""
    print("TEST: test_dir_exists_negative")

    # Test with a path that definitely doesn't exist
    var nonexistent = "/tmp/ml_odyssey_nonexistent_dir_12345"

    # Test: directory should NOT exist
    var exists = dir_exists(nonexistent)
    assert_false(exists, "Directory should not exist")

    print("PASS: dir_exists returns False for non-existent directory")


fn test_create_temp_dir() raises:
    """Test create_temp_dir creates a directory."""
    print("TEST: test_create_temp_dir")

    # Create temporary directory
    var temp_dir = create_temp_dir()

    # Test: directory should exist
    var exists = dir_exists(temp_dir)
    assert_true(exists, "Created directory should exist")

    # Test: path should start with /tmp/
    var starts_with_tmp = temp_dir.startswith("/tmp/")
    assert_true(starts_with_tmp, "Temp directory should be in /tmp/")

    # Cleanup
    cleanup_temp_dir(temp_dir)
    print("PASS: create_temp_dir creates directory in /tmp/")


fn test_create_temp_dir_unique() raises:
    """Test create_temp_dir creates unique directories."""
    print("TEST: test_create_temp_dir_unique")

    # Create two temporary directories
    var temp_dir1 = create_temp_dir()
    var temp_dir2 = create_temp_dir()

    # Test: directories should have different paths
    assert_not_equal(
        temp_dir1, temp_dir2, "Each call should create unique directory"
    )

    # Cleanup
    cleanup_temp_dir(temp_dir1)
    cleanup_temp_dir(temp_dir2)
    print("PASS: create_temp_dir creates unique directories")


# ============================================================================
# Tier 2 Tests - File Creation Functions (file existence checks)
# ============================================================================


fn test_create_mock_config_creates_file() raises:
    """Test create_mock_config creates a file."""
    print("TEST: test_create_mock_config_creates_file")

    # Setup - use /tmp/ directly to avoid Docker permission issues
    var temp_dir = _make_tmp_dir("mock_config")
    var config_path = temp_dir + "/config.yaml"
    var content = "key: value\ntest: 123"

    # Create config file
    create_mock_config(config_path, content)

    # Test: file should exist
    var exists = file_exists(config_path)
    assert_true(exists, "Config file should exist after creation")

    # Cleanup
    _cleanup_tmp_dir(temp_dir)
    print("PASS: create_mock_config creates file")


fn test_create_mock_text_file_creates_file() raises:
    """Test create_mock_text_file creates a file."""
    print("TEST: test_create_mock_text_file_creates_file")

    # Setup - use /tmp/ directly to avoid Docker permission issues
    var temp_dir = _make_tmp_dir("mock_text")
    var text_path = temp_dir + "/data.txt"

    # Create text file
    create_mock_text_file(text_path, num_lines=5)

    # Test: file should exist
    var exists = file_exists(text_path)
    assert_true(exists, "Text file should exist after creation")

    # Cleanup
    _cleanup_tmp_dir(temp_dir)
    print("PASS: create_mock_text_file creates file")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("=" * 70)
    print("Running io_helpers.mojo test suite - Part 1")
    print("=" * 70)
    print()

    # Tier 1 Tests - Foundation
    print("--- Tier 1: Foundation Functions ---")
    test_file_exists_positive()
    test_file_exists_negative()
    test_dir_exists_positive()
    test_dir_exists_negative()
    test_create_temp_dir()
    test_create_temp_dir_unique()
    print()

    # Tier 2 Tests - File Creation (existence checks)
    print("--- Tier 2: File Creation Functions (existence checks) ---")
    test_create_mock_config_creates_file()
    test_create_mock_text_file_creates_file()
    print()

    print("=" * 70)
    print("All 8 tests passed!")
    print("=" * 70)
