"""Tests for tensor serialization utilities.

Tests the complete serialization pipeline including:
- Single tensor save/load
- Named tensor collections
- Hex encoding/decoding
- DType utilities
"""

from testing import assert_true, assert_equal
from shared.testing.assertions import assert_almost_equal
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.utils.serialization import (
    NamedTensor,
    save_tensor,
    load_tensor,
    load_tensor_with_name,
    save_named_tensors,
    load_named_tensors,
    bytes_to_hex,
    hex_to_bytes,
    get_dtype_size,
    parse_dtype,
    dtype_to_string,
)
from pathlib import Path
from collections import List
import os


fn test_dtype_utilities() raises:
    """Test dtype string conversion functions."""
    # Test parse_dtype
    var f32_dtype = parse_dtype("float32")
    assert_true(f32_dtype == DType.float32, "Failed to parse float32")

    var f64_dtype = parse_dtype("float64")
    assert_true(f64_dtype == DType.float64, "Failed to parse float64")

    var int32_dtype = parse_dtype("int32")
    assert_true(int32_dtype == DType.int32, "Failed to parse int32")

    # Test dtype_to_string
    var s32 = dtype_to_string(DType.float32)
    assert_equal(s32, "float32", "Failed to convert DType to string")

    var s64 = dtype_to_string(DType.float64)
    assert_equal(s64, "float64", "Failed to convert float64 to string")

    # Test get_dtype_size
    var size_f32 = get_dtype_size(DType.float32)
    assert_equal(size_f32, 4, "float32 should be 4 bytes")

    var size_f64 = get_dtype_size(DType.float64)
    assert_equal(size_f64, 8, "float64 should be 8 bytes")

    var size_int32 = get_dtype_size(DType.int32)
    assert_equal(size_int32, 4, "int32 should be 4 bytes")


fn test_hex_encoding() raises:
    """Test hex encoding/decoding."""
    # Test bytes_to_hex and hex_to_bytes roundtrip
    var shape: List[Int] = [4]
    var tensor = ones(shape, DType.float32)

    # Get hex representation
    var dtype_size = get_dtype_size(DType.float32)
    var total_bytes = tensor.numel() * dtype_size
    var hex_str = bytes_to_hex(tensor._data, total_bytes)

    # Verify hex string has expected length
    assert_equal(
        len(hex_str), total_bytes * 2, "Hex string should be 2x byte count"
    )

    # Create new tensor and decode hex
    var tensor2 = zeros(shape, DType.float32)
    hex_to_bytes(hex_str, tensor2)

    # Verify data matches
    for i in range(tensor.numel()):
        var v1 = tensor._get_float64(i)
        var v2 = tensor2._get_float64(i)
        assert_almost_equal(
            v1, v2, tolerance=1e-6, message="Hex decode mismatch"
        )


fn test_single_tensor_serialization() raises:
    """Test saving and loading single tensor."""
    # Create test tensor
    var shape: List[Int] = [2, 3]
    var tensor = full(shape, 3.14, DType.float32)

    # Create temp file
    var tmpfile = "test_tensor_serialization.tmp"

    try:
        # Save tensor
        save_tensor(tensor, tmpfile, "test_param")

        # Verify file exists
        assert_true(_file_exists(tmpfile), "Tensor file not created")

        # Load tensor
        var loaded = load_tensor(tmpfile)

        # Verify shape
        var loaded_shape = loaded.shape()
        assert_equal(len(loaded_shape), 2, "Wrong number of dimensions")
        assert_equal(loaded_shape[0], 2, "Wrong first dimension")
        assert_equal(loaded_shape[1], 3, "Wrong second dimension")

        # Verify values
        assert_equal(loaded.numel(), 6, "Wrong number of elements")
        for i in range(loaded.numel()):
            var v = loaded._get_float64(i)
            assert_almost_equal(
                v, 3.14, tolerance=1e-6, message="Value mismatch after load"
            )

    finally:
        # Clean up
        if _file_exists(tmpfile):
            os.remove(tmpfile)


fn test_tensor_with_name() raises:
    """Test loading tensor with name preservation."""
    # Create test tensor
    var shape: List[Int] = [2, 2]
    var tensor = ones(shape, DType.float32)

    var tmpfile = "test_named_tensor.tmp"

    try:
        # Save with name
        save_tensor(tensor, tmpfile, "my_param")

        # Load with name
        var result = load_tensor_with_name(tmpfile)
        var name = result[0]
        var loaded = result[1]

        # Verify name
        assert_equal(name, "my_param", "Name not preserved")

        # Verify tensor
        assert_equal(loaded.numel(), 4, "Wrong tensor size")

    finally:
        if _file_exists(tmpfile):
            os.remove(tmpfile)


fn test_named_tensor_collection() raises:
    """Test saving and loading NamedTensor collections."""
    # Create directory for test with proper permissions
    var tmpdir = _create_temp_dir("test_named_tensors_dir")

    try:
        # Create named tensors
        var tensors: List[NamedTensor] = []

        var shape1: List[Int] = [2, 3]
        var tensor1 = full(shape1, 1.0, DType.float32)
        tensors.append(NamedTensor("weights", tensor1))

        var shape2: List[Int] = [3]
        var tensor2 = full(shape2, 0.5, DType.float32)
        tensors.append(NamedTensor("bias", tensor2))

        # Save collection
        save_named_tensors(tensors, tmpdir)

        # Verify files were created
        assert_true(
            _file_exists(tmpdir + "/weights.weights"),
            "weights.weights not created",
        )
        assert_true(
            _file_exists(tmpdir + "/bias.weights"), "bias.weights not created"
        )

        # Load collection
        var loaded = load_named_tensors(tmpdir)

        # Verify sizes
        assert_equal(len(loaded), 2, "Wrong number of tensors loaded")

        # Order-agnostic verification using name matching
        var found_weights = False
        var found_bias = False

        for i in range(len(loaded)):
            var name = loaded[i].name
            if name == "weights":
                found_weights = True
                assert_equal(loaded[i].tensor.numel(), 6, "Wrong size for weights")
            elif name == "bias":
                found_bias = True
                assert_equal(loaded[i].tensor.numel(), 3, "Wrong size for bias")

        assert_true(found_weights, "Missing weights tensor")
        assert_true(found_bias, "Missing bias tensor")

    finally:
        # Clean up
        _cleanup_temp_dir(tmpdir)


fn test_different_dtypes() raises:
    """Test serialization with different data types."""
    # Create temp directory with proper permissions
    var tmpdir = _create_temp_dir("test_dtype_serialization")

    try:
        # Test float32
        var f32_shape: List[Int] = [2]
        var f32_tensor = ones(f32_shape, DType.float32)
        var f32_path = tmpdir + "/f32.bin"
        save_tensor(f32_tensor, f32_path)
        var f32_loaded = load_tensor(f32_path)
        assert_equal(
            f32_loaded.dtype(), DType.float32, "float32 dtype mismatch"
        )

        # Test float64
        var f64_shape: List[Int] = [2]
        var f64_tensor = ones(f64_shape, DType.float64)
        var f64_path = tmpdir + "/f64.bin"
        save_tensor(f64_tensor, f64_path)
        var f64_loaded = load_tensor(f64_path)
        assert_true(
            f64_loaded.dtype() == DType.float64, "float64 dtype mismatch"
        )

        # Test int32
        var i32_shape: List[Int] = [2]
        var i32_tensor = ones(i32_shape, DType.int32)
        var i32_path = tmpdir + "/i32.bin"
        save_tensor(i32_tensor, i32_path)
        var i32_loaded = load_tensor(i32_path)
        assert_true(i32_loaded.dtype() == DType.int32, "int32 dtype mismatch")

    finally:
        _cleanup_temp_dir(tmpdir)


# ============================================================================
# Helper Functions
# ============================================================================


fn _file_exists(path: String) -> Bool:
    """Check if file exists."""
    try:
        with open(path, "r") as f:
            _ = f.read()
        return True
    except:
        return False


fn _create_temp_dir(prefix: String) -> String:
    """Create temporary directory with proper permissions for CI compatibility.

    Uses Python's tempfile.mkdtemp() to create a directory with guaranteed
    write permissions, which is necessary in CI environments.

    Args:
        prefix: Prefix for the temporary directory name.

    Returns:
        Full path to the created temporary directory.
    """
    try:
        from python import Python

        var tempfile = Python.import_module("tempfile")
        var builtins = Python.import_module("builtins")
        var py_prefix = builtins.str(prefix + "_")
        var tmp_dir = tempfile.mkdtemp(prefix=py_prefix)
        return String(tmp_dir)
    except:
        return prefix  # Fallback to original path


fn _cleanup_temp_dir(path: String):
    """Clean up temporary directory and all its contents.

    Args:
        path: Full path to the temporary directory to remove.
    """
    try:
        from python import Python

        var shutil = Python.import_module("shutil")
        shutil.rmtree(path)
    except:
        pass


fn main() raises:
    """Run all serialization tests."""
    print("Testing dtype utilities...")
    test_dtype_utilities()

    print("Testing hex encoding...")
    test_hex_encoding()

    print("Testing single tensor serialization...")
    test_single_tensor_serialization()

    print("Testing tensor with name...")
    test_tensor_with_name()

    print("Testing named tensor collection...")
    test_named_tensor_collection()

    print("Testing different dtypes...")
    test_different_dtypes()

    print("All serialization tests passed!")
