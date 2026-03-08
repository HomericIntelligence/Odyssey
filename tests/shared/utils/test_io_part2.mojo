# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_io.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for file I/O utilities module - Part 2: Tensor Serialization & Safe File Ops.

This module tests:
- Tensor serialization/deserialization
- Atomic file writes
- File backup on write
- Safe file removal
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)


# ============================================================================
# Test Tensor Serialization
# ============================================================================


fn test_serialize_tensor():
    """Test serializing tensor to bytes."""
    # TODO(#44): Implement when Tensor.serialize exists
    # Create tensor with known values
    # Serialize to bytes
    # Verify bytes contain tensor data and metadata (shape, dtype)
    pass


fn test_deserialize_tensor():
    """Test deserializing tensor from bytes."""
    # TODO(#44): Implement when Tensor.deserialize exists
    # Create serialized tensor bytes
    # Deserialize to tensor
    # Verify shape, dtype, and values match original
    pass


fn test_tensor_roundtrip():
    """Test serializing and deserializing tensor preserves values."""
    # TODO(#44): Implement when Tensor serialization exists
    # Create random tensor
    # Serialize to bytes
    # Deserialize back to tensor
    # Verify all values match exactly
    pass


fn test_serialize_large_tensor():
    """Test serializing large tensor (> 1GB)."""
    # TODO(#44): Implement when Tensor serialization exists
    # Create large tensor (e.g., 256M Float32 = 1GB)
    # Serialize to file
    # Verify file size is correct
    # Deserialize and verify values (spot check)
    pass


fn test_serialize_tensor_formats():
    """Test serializing tensors with different dtypes."""
    # TODO(#44): Implement when Tensor serialization exists
    # Test serialization for:
    # - Float32, Float64
    # - Int8, Int16, Int32, Int64
    # - Bool
    # Verify dtype is preserved in serialization
    pass


# ============================================================================
# Test Safe File Operations
# ============================================================================


fn test_atomic_write():
    """Test atomic file write (write to temp, then rename)."""
    # TODO(#44): Implement when atomic_write exists
    # Write data to file atomically
    # Verify temp file is created first
    # Verify temp file is renamed to target
    # No partial writes visible at target path
    pass


fn test_write_with_backup():
    """Test writing file creates backup of existing file."""
    # TODO(#44): Implement when write_with_backup exists
    # Create file with content "old"
    # Write new content "new" with backup
    # Verify: original file has "new"
    # Verify: backup file has "old"
    # Clean up temp files
    pass


fn test_safe_remove() raises:
    """Test remove_safely() actually deletes the file."""
    from shared.utils.file_io import remove_safely, safe_write_file, file_exists

    var test_path = "/tmp/test_remove_safely_3283.txt"

    # Create file
    var written = safe_write_file(test_path, "test content")
    assert_true(written)
    assert_true(file_exists(test_path))

    # Remove it
    var removed = remove_safely(test_path)
    assert_true(removed)
    assert_false(file_exists(test_path))

    # Removing nonexistent file returns False
    var removed_again = remove_safely(test_path)
    assert_false(removed_again)

    print("PASS: test_safe_remove")


fn main() raises:
    """Run all tests."""
    test_serialize_tensor()
    test_deserialize_tensor()
    test_tensor_roundtrip()
    test_serialize_large_tensor()
    test_serialize_tensor_formats()
    test_atomic_write()
    test_write_with_backup()
    test_safe_remove()
