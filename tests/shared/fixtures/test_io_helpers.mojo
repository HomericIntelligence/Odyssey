"""Test suite for io_helpers.mojo test utilities.

Test Coverage:
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


comptime _TMP_PREFIX = "/tmp/ml_odyssey_test_io_"


def _make_tmp_dir(suffix: String) raises -> String:
    """Create a unique temp directory under /tmp/ using Mojo builtins."""
    var path = _TMP_PREFIX + suffix
    from std.os import mkdir

    try:
        mkdir(path)
    except:
        # Directory may already exist from a previous failed run
        pass
    return path


def _cleanup_tmp_dir(path: String) raises:
    """Remove a temp directory under /tmp/ using Python shutil (safe)."""
    # Only clean paths under /tmp/ for safety
    if not path.startswith("/tmp/"):
        raise Error("Refusing to clean non-/tmp/ path: " + path)
    from std.python import Python

    var shutil = Python.import_module("shutil")
    try:
        shutil.rmtree(path)
    except:
        pass  # Ignore cleanup errors


def test_file_exists_positive() raises:
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


def test_file_exists_negative() raises:
    """Test file_exists returns False for non-existent file."""
    print("TEST: test_file_exists_negative")

    # Test with a path that definitely doesn't exist
    var nonexistent = "/tmp/ml_odyssey_nonexistent_file_12345.txt"

    # Test: file should NOT exist
    var exists = file_exists(nonexistent)
    assert_false(exists, "File should not exist")

    print("PASS: file_exists returns False for non-existent file")


def test_dir_exists_positive() raises:
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


def test_dir_exists_negative() raises:
    """Test dir_exists returns False for non-existent directory."""
    print("TEST: test_dir_exists_negative")

    # Test with a path that definitely doesn't exist
    var nonexistent = "/tmp/ml_odyssey_nonexistent_dir_12345"

    # Test: directory should NOT exist
    var exists = dir_exists(nonexistent)
    assert_false(exists, "Directory should not exist")

    print("PASS: dir_exists returns False for non-existent directory")


def test_create_temp_dir() raises:
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


def test_create_temp_dir_unique() raises:
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


def test_create_mock_config_creates_file() raises:
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


def test_create_mock_text_file_creates_file() raises:
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


def test_create_mock_config_writes_content() raises:
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


def test_create_mock_text_file_correct_format() raises:
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


def test_create_mock_checkpoint_creates_file() raises:
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


def test_create_mock_checkpoint_correct_format() raises:
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


def test_get_test_data_path() raises:
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


def test_cleanup_temp_dir() raises:
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


def test_cleanup_temp_dir_safety_check() raises:
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


def test_full_lifecycle() raises:
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


def main() raises:
    """Run all test_io_helpers tests."""
    print("Running test_io_helpers tests...")

    test_file_exists_positive()
    print("✓ test_file_exists_positive")

    test_file_exists_negative()
    print("✓ test_file_exists_negative")

    test_dir_exists_positive()
    print("✓ test_dir_exists_positive")

    test_dir_exists_negative()
    print("✓ test_dir_exists_negative")

    test_create_temp_dir()
    print("✓ test_create_temp_dir")

    test_create_temp_dir_unique()
    print("✓ test_create_temp_dir_unique")

    test_create_mock_config_creates_file()
    print("✓ test_create_mock_config_creates_file")

    test_create_mock_text_file_creates_file()
    print("✓ test_create_mock_text_file_creates_file")

    test_create_mock_config_writes_content()
    print("✓ test_create_mock_config_writes_content")

    test_create_mock_text_file_correct_format()
    print("✓ test_create_mock_text_file_correct_format")

    test_create_mock_checkpoint_creates_file()
    print("✓ test_create_mock_checkpoint_creates_file")

    test_create_mock_checkpoint_correct_format()
    print("✓ test_create_mock_checkpoint_correct_format")

    test_get_test_data_path()
    print("✓ test_get_test_data_path")

    test_cleanup_temp_dir()
    print("✓ test_cleanup_temp_dir")

    test_cleanup_temp_dir_safety_check()
    print("✓ test_cleanup_temp_dir_safety_check")

    test_full_lifecycle()
    print("✓ test_full_lifecycle")

    print("\nAll test_io_helpers tests passed!")
