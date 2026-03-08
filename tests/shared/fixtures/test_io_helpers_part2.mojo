"""Test suite for io_helpers.mojo test utilities - Part 2.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_io_helpers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Test Coverage (Part 2):
- Tier 2: create_mock_config, create_mock_text_file - content verification (2 tests)
- Tier 3: create_mock_checkpoint, get_test_data_path, cleanup_temp_dir (5 tests)
- Integration: Full lifecycle (create → write → cleanup) (1 test)
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
)
from tests.shared.fixtures.io_helpers import (
    file_exists,
    dir_exists,
    create_temp_dir,
    cleanup_temp_dir,
    create_mock_config,
    create_mock_text_file,
    create_mock_checkpoint,
    get_test_data_path,
    temp_file_path,
)


# ============================================================================
# Tier 2 Tests - File Creation Functions (content verification)
# ============================================================================


fn test_create_mock_config_writes_content() raises:
    """Test create_mock_config writes correct content."""
    print("TEST: test_create_mock_config_writes_content")

    # Setup
    var temp_dir = create_temp_dir()
    var config_path = temp_file_path(temp_dir, "config.yaml")
    var content = "model:\n  name: TestModel\n  layers: 3"

    # Create config file
    create_mock_config(config_path, content)

    # Read back the file to verify content
    var read_content: String
    with open(config_path, "r") as f:
        read_content = f.read()

    # Test: content should match
    assert_equal(read_content, content, "Written content should match input")

    # Cleanup
    cleanup_temp_dir(temp_dir)
    print("PASS: create_mock_config writes correct content")


fn test_create_mock_text_file_correct_format() raises:
    """Test create_mock_text_file writes correct line format."""
    print("TEST: test_create_mock_text_file_correct_format")

    # Setup
    var temp_dir = create_temp_dir()
    var text_path = temp_file_path(temp_dir, "data.txt")

    # Create text file with 3 lines
    create_mock_text_file(text_path, num_lines=3)

    # Read back the file
    var read_content: String
    with open(text_path, "r") as f:
        read_content = f.read()

    # Test: content should have correct format
    var expected = "Line 1\nLine 2\nLine 3"
    assert_equal(
        read_content, expected, "Text file should have 'Line N' format"
    )

    # Cleanup
    cleanup_temp_dir(temp_dir)
    print("PASS: create_mock_text_file writes correct format")


# ============================================================================
# Tier 3 Tests - Advanced Functions
# ============================================================================


fn test_create_mock_checkpoint_creates_file() raises:
    """Test create_mock_checkpoint creates a file."""
    print("TEST: test_create_mock_checkpoint_creates_file")

    # Setup
    var temp_dir = create_temp_dir()
    var ckpt_path = temp_file_path(temp_dir, "model.ckpt")

    # Create checkpoint file
    create_mock_checkpoint(ckpt_path, num_params=100, random_seed=42)

    # Test: file should exist
    var exists = file_exists(ckpt_path)
    assert_true(exists, "Checkpoint file should exist after creation")

    # Cleanup
    cleanup_temp_dir(temp_dir)
    print("PASS: create_mock_checkpoint creates file")


fn test_create_mock_checkpoint_correct_format() raises:
    """Test create_mock_checkpoint writes correct format."""
    print("TEST: test_create_mock_checkpoint_correct_format")

    # Setup
    var temp_dir = create_temp_dir()
    var ckpt_path = temp_file_path(temp_dir, "model.ckpt")

    # Create checkpoint file
    create_mock_checkpoint(ckpt_path, num_params=150, random_seed=99)

    # Read back the file
    var read_content: String
    with open(ckpt_path, "r") as f:
        read_content = f.read()

    # Test: content should have checkpoint format
    var has_epoch = read_content.__contains__("EPOCH:")
    var has_loss = read_content.__contains__("LOSS:")
    var has_accuracy = read_content.__contains__("ACCURACY:")
    var has_meta = read_content.__contains__("META:")

    assert_true(has_epoch, "Checkpoint should contain EPOCH field")
    assert_true(has_loss, "Checkpoint should contain LOSS field")
    assert_true(has_accuracy, "Checkpoint should contain ACCURACY field")
    assert_true(has_meta, "Checkpoint should contain META fields")

    # Cleanup
    cleanup_temp_dir(temp_dir)
    print("PASS: create_mock_checkpoint writes correct format")


fn test_get_test_data_path() raises:
    """Test get_test_data_path returns correct path."""
    print("TEST: test_get_test_data_path")

    # Test with a filename
    var path = get_test_data_path("sample.txt")

    # Test: path should start with fixtures directory
    var expected_prefix = "tests/shared/fixtures/"
    var starts_correctly = path.startswith(expected_prefix)
    assert_true(starts_correctly, "Path should start with fixtures directory")

    # Test: path should end with filename
    var ends_correctly = path.endswith("sample.txt")
    assert_true(ends_correctly, "Path should end with filename")

    print("PASS: get_test_data_path returns correct path")


fn test_cleanup_temp_dir() raises:
    """Test cleanup_temp_dir removes directory."""
    print("TEST: test_cleanup_temp_dir")

    # Create a temporary directory with files
    var temp_dir = create_temp_dir()
    var test_file = temp_file_path(temp_dir, "test.txt")
    create_mock_text_file(test_file, num_lines=1)

    # Verify directory exists before cleanup
    var exists_before = dir_exists(temp_dir)
    assert_true(exists_before, "Directory should exist before cleanup")

    # Cleanup
    cleanup_temp_dir(temp_dir)

    # Test: directory should NOT exist after cleanup
    var exists_after = dir_exists(temp_dir)
    assert_false(exists_after, "Directory should not exist after cleanup")

    print("PASS: cleanup_temp_dir removes directory")


fn test_cleanup_temp_dir_safety_check() raises:
    """Test cleanup_temp_dir rejects non-tmp paths."""
    print("TEST: test_cleanup_temp_dir_safety_check")

    # Try to cleanup a path outside /tmp
    var unsafe_path = "/home/user/data"

    # Test: should raise an error
    var raised_error = False
    try:
        cleanup_temp_dir(unsafe_path)
    except:
        raised_error = True

    assert_true(raised_error, "cleanup_temp_dir should reject non-/tmp paths")

    print("PASS: cleanup_temp_dir enforces /tmp safety check")


# ============================================================================
# Integration Tests - Full Lifecycle
# ============================================================================


fn test_full_lifecycle() raises:
    """Test complete create → write → cleanup workflow."""
    print("TEST: test_full_lifecycle")

    # Step 1: Create temporary directory
    var temp_dir = create_temp_dir()
    assert_true(dir_exists(temp_dir), "Directory should exist after creation")

    # Step 2: Create multiple files
    var config_path = temp_file_path(temp_dir, "config.yaml")
    var text_path = temp_file_path(temp_dir, "data.txt")
    var ckpt_path = temp_file_path(temp_dir, "model.ckpt")

    create_mock_config(config_path, "key: value")
    create_mock_text_file(text_path, num_lines=5)
    create_mock_checkpoint(ckpt_path, num_params=50, random_seed=42)

    # Step 3: Verify all files exist
    assert_true(file_exists(config_path), "Config file should exist")
    assert_true(file_exists(text_path), "Text file should exist")
    assert_true(file_exists(ckpt_path), "Checkpoint file should exist")

    # Step 4: Cleanup everything
    cleanup_temp_dir(temp_dir)

    # Step 5: Verify directory is gone
    assert_false(
        dir_exists(temp_dir), "Directory should be removed after cleanup"
    )

    print("PASS: Full lifecycle works correctly")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("=" * 70)
    print("Running io_helpers.mojo test suite - Part 2")
    print("=" * 70)
    print()

    # Tier 2 Tests - Content Verification
    print("--- Tier 2: File Creation Functions (content verification) ---")
    test_create_mock_config_writes_content()
    test_create_mock_text_file_correct_format()
    print()

    # Tier 3 Tests - Advanced
    print("--- Tier 3: Advanced Functions ---")
    test_create_mock_checkpoint_creates_file()
    test_create_mock_checkpoint_correct_format()
    test_get_test_data_path()
    test_cleanup_temp_dir()
    test_cleanup_temp_dir_safety_check()
    print()

    # Integration Tests
    print("--- Integration Tests ---")
    test_full_lifecycle()
    print()

    print("=" * 70)
    print("All 8 tests passed!")
    print("=" * 70)
