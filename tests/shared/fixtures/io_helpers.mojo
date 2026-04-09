"""File I/O test helpers for temporary files and test data.

This module provides utilities for managing temporary files and directories
during testing, including creating/cleaning up test files, paths, and configs.

Key functions:
- create_temp_dir(): Create temporary test directory
- cleanup_temp_dir(): Remove temporary directory
- create_mock_config(): Write configuration file
- get_test_data_path(): Resolve test data file paths
- create_mock_checkpoint(): Create mock checkpoint file

All functions are designed for use in test setup/teardown.
"""

from pathlib import Path
from sys import argv
from python import Python, PythonObject


# ============================================================================
# Temporary Directory Management
# ============================================================================


fn create_temp_dir(prefix: String = "ml_odyssey_test_") raises -> String:
    """Create temporary directory for testing.

    Creates a unique temporary directory in the system temp location
    with a timestamp-based name for uniqueness.

    Args:
        prefix: Prefix for directory name (default: "ml_odyssey_test_").

    Returns:
        Absolute path to created directory.

    Example:
        ```mojo
        var temp_dir = create_temp_dir()
        # temp_dir = "/tmp/ml_odyssey_test_1699999999/"
        ```

    Note:
        Directory is NOT automatically cleaned up - use cleanup_temp_dir()
        in test teardown or try/finally block.
    """
    # Use Python's tempfile.mkdtemp for atomic directory creation
    # Explicitly use /tmp as base to avoid permission issues in Docker
    # containers where TMPDIR may point to a non-writable directory
    var tempfile = Python.import_module("tempfile")
    var temp_path = tempfile.mkdtemp(
        prefix=PythonObject(prefix), dir=PythonObject("/tmp")
    )
    return String(temp_path)


fn cleanup_temp_dir(path: String) raises:
    """Remove temporary directory and all contents.

    Recursively removes directory and all files/subdirectories.

    Args:
        path: Path to directory to remove.

    Raises:
        Error if path doesn't exist or removal fails.

    Example:
        ```mojo
        var temp_dir = create_temp_dir()
        try:
            # Use temp_dir for testing
            var config_path = temp_dir + "/config.yaml"
            create_mock_config(config_path, "key: value")
        finally:
            cleanup_temp_dir(temp_dir)
        ```

    Warning:
        This permanently deletes the directory and all contents.
        Be careful not to call on non-temporary directories.
    """
    # Validation
    if len(path) == 0:
        raise Error("Cannot cleanup empty path")

    # Safety check - verify path is in system temp directory
    # Get system temp directory using Python's tempfile module
    var tempfile = Python.import_module("tempfile")
    var temp_base = String(tempfile.gettempdir())

    # Verify the path is within the system temp directory or /tmp
    # (create_temp_dir explicitly uses /tmp which may differ from gettempdir)
    if not path.startswith(temp_base) and not path.startswith("/tmp"):
        raise Error(
            "cleanup_temp_dir only works with system temp paths for safety. "
            "Expected path within: "
            + temp_base
        )

    # Use Python's shutil.rmtree for recursive directory removal
    var shutil = Python.import_module("shutil")
    shutil.rmtree(path)


fn temp_file_path(directory: String, filename: String) -> String:
    """Construct path to temporary file.

    Args:
        directory: Temporary directory path.
        filename: Name of file.

    Returns:
        Full path to file in directory.

    Example:
        ```mojo
        var temp_dir = create_temp_dir()
        var config_path = temp_file_path(temp_dir, "config.yaml")
        # config_path = "/tmp/ml_odyssey_test_123/config.yaml"
        ```
    """
    if directory.endswith("/"):
        return directory + filename
    else:
        return directory + "/" + filename


# ============================================================================
# Mock File Creation
# ============================================================================


fn create_mock_config(path: String, content: String) raises:
    """Create mock configuration file.

    Writes a configuration file (YAML or JSON) to the specified path.

    Args:
        path: Full path where file should be created.
        content: File content to write.

    Raises:
        Error if file cannot be created.

    Example:
        ```mojo
        var temp_dir = create_temp_dir()
        var config_path = temp_file_path(temp_dir, "config.yaml")

        var yaml_content = \"\"\"
        model:
          name: TestModel
          layers: 3
        training:
          batch_size: 32
          epochs: 10
        \"\"\"

        create_mock_config(config_path, yaml_content)
        ```

    Note:
        This doesn't validate YAML/JSON syntax - it just writes the string.
        Use config_fixtures.mojo for pre-validated config templates.
    """
    # Validation
    if len(path) == 0:
        raise Error("Path cannot be empty")
    if len(content) == 0:
        raise Error("Content cannot be empty")

    # Write content to file using builtin open()
    with open(path, "w") as f:
        f.write(content)


fn create_mock_checkpoint(
    path: String, num_params: Int = 100, random_seed: Int = 42
) raises:
    """Create mock model checkpoint file.

    Creates a simple checkpoint file with mock parameter data.

    Args:
        path: Full path where checkpoint should be created.
        num_params: Number of parameters to include (default: 100).
        random_seed: Random seed for parameter values (default: 42).

    Raises:
        Error if file cannot be created.

    Example:
        ```mojo
        var temp_dir = create_temp_dir()
        var ckpt_path = temp_file_path(temp_dir, "model.ckpt")
        create_mock_checkpoint(ckpt_path, num_params=1000)
        ```

    Note:
        This creates a simplified checkpoint format for testing.
        Not compatible with production checkpoint formats.
    """
    # Validation
    if len(path) == 0:
        raise Error("Path cannot be empty")
    if num_params <= 0:
        raise Error("num_params must be positive")

    # Create simplified checkpoint format matching shared/utils/io.mojo:86-125
    var content = "EPOCH:0\n"
    content += "LOSS:0.0\n"
    content += "ACCURACY:0.0\n"
    content += "META:num_params=" + String(num_params) + "\n"
    content += "META:seed=" + String(random_seed)

    # Write checkpoint to file
    with open(path, "w") as f:
        f.write(content)


fn create_mock_text_file(path: String, num_lines: Int = 10) raises:
    """Create mock text file with sample lines.

    Args:
        path: Full path where file should be created.
        num_lines: Number of lines to write (default: 10).

    Raises:
        Error if file cannot be created.

    Example:
        ```mojo
        var temp_dir = create_temp_dir()
        var data_path = temp_file_path(temp_dir, "data.txt")
        create_mock_text_file(data_path, num_lines=100)
        ```
    """
    # Validation
    if len(path) == 0:
        raise Error("Path cannot be empty")
    if num_lines <= 0:
        raise Error("num_lines must be positive")

    # Generate content in "Line 1\nLine 2\n..." format
    var content = ""
    for i in range(1, num_lines + 1):
        if i > 1:
            content += "\n"
        content += "Line " + String(i)

    # Write content to file
    with open(path, "w") as f:
        f.write(content)


# ============================================================================
# Test Data Path Resolution
# ============================================================================


fn get_test_data_path(filename: String) -> String:
    """Resolve path to test data file.

    Returns absolute path to a file in the test data directory.
    Useful for loading reference data, images, etc.

    Args:
        filename: Name of test data file.

    Returns:
        Absolute path to file in tests/shared/fixtures/ directory.

    Example:
        ```mojo
        var image_path = get_test_data_path("sample_image.png")
        # image_path = "/path/to/tests/shared/fixtures/images/sample_image.png"
        ```

    Note:
        Assumes test data is stored in tests/shared/fixtures/ subdirectories.
        Create subdirectories as needed: `images/`, `tensors/`, `models/`, `reference/`.
    """
    # Get test fixtures directory
    # Assumes tests run from repository root
    var fixtures_dir = "tests/shared/fixtures/"
    return fixtures_dir + filename


fn get_fixtures_dir() -> String:
    """Get path to fixtures directory.

    Returns:
        Absolute path to tests/shared/fixtures/ directory.

    Example:
        ```mojo
        var fixtures = get_fixtures_dir()
        var images_dir = fixtures + "images/"
        ```
    """
    return "tests/shared/fixtures/"


# ============================================================================
# File Existence Checking
# ============================================================================


fn file_exists(path: String) -> Bool:
    """Check if file exists at path.

    Args:
        path: Path to check.

    Returns:
        True if file exists, False otherwise.

    Example:
        ```mojo
        var temp_dir = create_temp_dir()
        var config_path = temp_file_path(temp_dir, "config.yaml")

        if not file_exists(config_path):
            create_mock_config(config_path, "key: value")
        ```

    Note:
        Returns False for directories - only checks for files.
    """
    # Try to open file for reading - if successful, it exists
    try:
        with open(path, "r") as f:
            _ = f.read()  # Attempt to read (confirms it's a file)
        return True
    except:
        return False


fn dir_exists(path: String) -> Bool:
    """Check if directory exists at path.

    Args:
        path: Path to check.

    Returns:
        True if directory exists, False otherwise.

    Example:
        ```mojo
        var temp_dir = "/tmp/my_test_dir"
        if not dir_exists(temp_dir):
            temp_dir = create_temp_dir()
        ```
    """
    # Use Mojo's Path.is_dir() native method
    var dir_path = Path(path)
    return dir_path.is_dir()


# ============================================================================
# Path Utilities
# ============================================================================


fn join_paths(parts: List[String]) -> String:
    """Join path components with proper separators.

    Args:
        parts: List of path components.

    Returns:
        Joined path string.

    Example:
        ```mojo
        var parts = List[String]()
        parts.append("/tmp")
        parts.append("test_dir")
        parts.append("config.yaml")

        var path = join_paths(parts)
        # path = "/tmp/test_dir/config.yaml"
        ```
    """
    if len(parts) == 0:
        return ""

    var result = parts[0]
    for i in range(1, len(parts)):
        var part = parts[i]
        # Add separator if needed
        if not result.endswith("/"):
            result = result + "/"
        # Skip leading slash in part
        if part.startswith("/"):
            result = result + part[1:]
        else:
            result = result + part

    return result


fn get_filename(path: String) -> String:
    """Extract filename from path.

    Args:
        path: Full file path.

    Returns:
        Filename without directory.

    Example:
        ```mojo
        var path = "/tmp/test_dir/config.yaml"
        var filename = get_filename(path)
        # filename = "config.yaml"
        ```
    """
    # Find last slash
    var last_slash = -1
    for i in range(len(path) - 1, -1, -1):
        if path[i] == "/":
            last_slash = i
            break

    if last_slash == -1:
        return path
    else:
        return path.substr(last_slash + 1, len(path) - (last_slash + 1))


fn get_extension(path: String) -> String:
    """Extract file extension from path.

    Args:
        path: Full file path.

    Returns:
        File extension including dot (e.g., ".yaml", ".json").
        Returns empty string if no extension.

    Example:
        ```mojo
        var ext = get_extension("/tmp/config.yaml")
        # ext = ".yaml"
        ```
    """
    # Find last dot
    var last_dot = -1
    for i in range(len(path) - 1, -1, -1):
        if path[i] == ".":
            last_dot = i
            break
        elif path[i] == "/":
            # Hit directory separator before finding dot
            return ""

    if last_dot == -1:
        return ""
    else:
        return path.substr(last_dot, len(path) - last_dot)


# ============================================================================
# Test Data Organization
# ============================================================================


fn get_images_dir() -> String:
    """Get path to test images directory."""
    return get_fixtures_dir() + "images/"


fn get_tensors_dir() -> String:
    """Get path to test tensors directory."""
    return get_fixtures_dir() + "tensors/"


fn get_models_dir() -> String:
    """Get path to test models directory."""
    return get_fixtures_dir() + "models/"


fn get_reference_dir() -> String:
    """Get path to reference outputs directory."""
    return get_fixtures_dir() + "reference/"


fn main() raises:
    print("io_helpers.mojo - This is a HELPER MODULE, not a test file")
    print("It provides utilities for file I/O in tests")
    print("No tests are executed from this file")
