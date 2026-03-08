# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_io.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for file I/O utilities module - Part 3: Directory & Binary File Ops.

This module tests:
- Safe directory creation
- File existence checks
- Binary file read/write operations
- Chunked binary reads
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)


# ============================================================================
# Test Safe File Operations (continued)
# ============================================================================


fn test_create_directory_safe():
    """Test creating directory safely (no error if exists)."""
    # TODO(#44): Implement when mkdir_safe exists
    # Create directory
    # Create same directory again
    # Verify no error on second create
    # Clean up directory
    pass


fn test_file_exists_check():
    """Test checking if file exists."""
    # TODO(#44): Implement when file_exists helper exists
    # Create temp file
    # Verify file_exists returns True
    # Remove file
    # Verify file_exists returns False
    pass


# ============================================================================
# Test Binary File Operations
# ============================================================================


fn test_write_binary_file():
    """Test writing binary data to file."""
    # TODO(#44): Implement when write_binary exists
    # Create byte array with known values
    # Write to temp file
    # Read file and verify bytes match
    # Clean up temp file
    pass


fn test_read_binary_file():
    """Test reading binary data from file."""
    # TODO(#44): Implement when read_binary exists
    # Create temp file with binary data
    # Read file
    # Verify bytes match original data
    # Clean up temp file
    pass


fn test_binary_file_roundtrip():
    """Test writing and reading binary file preserves data."""
    # TODO(#44): Implement when binary I/O exists
    # Create random binary data
    # Write to file
    # Read from file
    # Verify all bytes match exactly
    pass


fn test_read_binary_in_chunks():
    """Test reading large binary file in chunks."""
    # TODO(#44): Implement when chunked read exists
    # Create large binary file (e.g., 100MB)
    # Read in 10MB chunks
    # Verify all chunks read correctly
    # Total data matches file size
    pass


# ============================================================================
# Test Text File Operations
# ============================================================================


fn test_write_text_file():
    """Test writing text to file."""
    # TODO(#44): Implement when write_text exists
    # Create text string
    # Write to temp file
    # Read file and verify text matches
    # Clean up temp file
    pass


fn test_read_text_file():
    """Test reading text from file."""
    # TODO(#44): Implement when read_text exists
    # Create temp file with text
    # Read file
    # Verify text matches original
    # Clean up temp file
    pass


fn main() raises:
    """Run all tests."""
    test_create_directory_safe()
    test_file_exists_check()
    test_write_binary_file()
    test_read_binary_file()
    test_binary_file_roundtrip()
    test_read_binary_in_chunks()
    test_write_text_file()
    test_read_text_file()
