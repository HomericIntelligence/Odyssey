# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_io.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for file I/O utilities module - Part 4: Text File Ops & Path Operations.

This module tests:
- Text file line reading and appending
- Path resolution and manipulation
- Directory listing
- Error handling for missing files
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)


# ============================================================================
# Test Text File Operations (continued)
# ============================================================================


fn test_read_text_lines():
    """Test reading text file line by line."""
    # TODO(#44): Implement when read_lines exists
    # Create file with multiple lines
    # Read lines as list
    # Verify line count and content
    pass


fn test_append_to_text_file():
    """Test appending text to existing file."""
    # TODO(#44): Implement when append_text exists
    # Create file with "line 1\n"
    # Append "line 2\n"
    # Read file
    # Verify: "line 1\nline 2\n"
    pass


# ============================================================================
# Test Path Operations
# ============================================================================


fn test_resolve_path():
    """Test resolving relative paths to absolute paths."""
    # TODO(#44): Implement when resolve_path exists
    # Resolve "./data/file.csv"
    # Verify returns absolute path
    # Resolve "~/data/file.csv"
    # Verify ~ is expanded to home directory
    pass


fn test_join_paths():
    """Test joining path components."""
    # TODO(#44): Implement when join_path exists
    # Join ["data", "train", "images.csv"]
    # Verify: "data/train/images.csv" (Unix) or "data\\train\\images.csv" (Windows)
    pass


fn test_split_path():
    """Test splitting path into directory and filename."""
    # TODO(#44): Implement when split_path exists
    # Split "data/train/images.csv"
    # Verify: directory="data/train", filename="images.csv"
    pass


fn test_get_file_extension():
    """Test extracting file extension."""
    # TODO(#44): Implement when get_extension exists
    # Extension of "model.mojo" -> ".mojo"
    # Extension of "data.tar.gz" -> ".gz" or ".tar.gz"?
    # Extension of "no_extension" -> ""
    pass


fn test_list_directory():
    """Test listing files in directory."""
    # TODO(#44): Implement when list_dir exists
    # Create temp directory with files: a.txt, b.txt, c.csv
    # List directory
    # Verify returns ["a.txt", "b.txt", "c.csv"]
    # Clean up temp directory
    pass


# ============================================================================
# Test Error Handling
# ============================================================================


fn test_load_nonexistent_file():
    """Test loading nonexistent file raises error."""
    # TODO(#44): Implement when load_checkpoint exists
    # Try to load "nonexistent.checkpoint"
    # Verify FileNotFoundError is raised
    pass


fn main() raises:
    """Run all tests."""
    test_read_text_lines()
    test_append_to_text_file()
    test_resolve_path()
    test_join_paths()
    test_split_path()
    test_get_file_extension()
    test_list_directory()
    test_load_nonexistent_file()
