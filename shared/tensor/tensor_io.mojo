"""Tensor I/O utilities: save/load AnyTensor to/from hex-encoded text files.

This module provides the core save/load implementation for AnyTensor. It is
defined in shared.core (rather than shared.utils) to avoid a circular type
resolution issue in Mojo v0.26.1:

    Problem: shared.utils.serialization imports AnyTensor from shared.core.any_tensor.
    When any_tensor.mojo has a method that imports from shared.utils.serialization,
    the package compiler compiles any_tensor.mojo twice with distinct type identities,
    breaking all operator overloads with 'AnyTensor cannot convert from AnyTensor' errors.

    Fix: Move save/load core implementation here (shared.core.tensor_io) so
    any_tensor.mojo can use a relative import (from .tensor_io import save_tensor)
    instead of a cross-package import.

    shared.utils.serialization re-exports these functions for backward compatibility.

File format (hex-encoded text):
    Line 1: tensor name (may be empty)
    Line 2: dtype dim0 dim1 ... dimN
    Line 3: hex-encoded raw bytes
"""

from std.memory import UnsafePointer
from std.collections import List
from .any_tensor import AnyTensor
from .tensor_creation import zeros


# ============================================================================
# Core Save/Load
# ============================================================================


def save_tensor(tensor: AnyTensor, filepath: String, name: String = "") raises:
    """Save tensor to file in hex format.

    Args:
        tensor: Tensor to save.
        filepath: Output file path.
        name: Optional tensor name (defaults to empty string).

    Raises:
        Error: If file write fails.
    """
    var local_tensor = tensor

    var shape = local_tensor.shape()
    var dtype = local_tensor.dtype()
    var numel = local_tensor.numel()

    var dtype_str = dtype_to_string(dtype)
    var metadata = dtype_str + " "
    for i in range(len(shape)):
        metadata += String(shape[i])
        if i < len(shape) - 1:
            metadata += " "

    var dtype_size = get_dtype_size(dtype)
    var total_bytes = numel * dtype_size
    var hex_data = bytes_to_hex(local_tensor._data, total_bytes)

    with open(filepath, "w") as f:
        _ = f.write(name + "\n")
        _ = f.write(metadata + "\n")
        _ = f.write(hex_data + "\n")


def load_tensor(filepath: String) raises -> AnyTensor:
    """Load tensor from file.

    Args:
        filepath: Input file path.

    Returns:
        Loaded AnyTensor.

    Raises:
        Error: If file format is invalid or file doesn't exist.
    """
    var content: String
    with open(filepath, "r") as f:
        content = f.read()

    var lines = content.split("\n")
    if len(lines) < 3:
        raise Error("Invalid tensor file format: expected 3+ lines")

    var _ = String(lines[0])
    var metadata = String(lines[1])
    var hex_data = String(lines[2])

    var meta_parts = metadata.split(" ")
    if len(meta_parts) < 1:
        raise Error("Invalid metadata format: expected dtype and shape")

    var dtype_str = meta_parts[0]
    var dtype = parse_dtype(String(dtype_str))

    var shape = List[Int]()
    for i in range(1, len(meta_parts)):
        shape.append(Int(meta_parts[i]))

    var tensor = zeros(shape, dtype)
    hex_to_bytes(hex_data, tensor)

    return tensor^


def load_tensor_with_name(filepath: String) raises -> Tuple[String, AnyTensor]:
    """Load tensor with its associated name.

    Args:
        filepath: Input file path.

    Returns:
        Tuple of (name, tensor).

    Raises:
        Error: If file format is invalid or file doesn't exist.
    """
    var content: String
    with open(filepath, "r") as f:
        content = f.read()

    var lines = content.split("\n")
    if len(lines) < 3:
        raise Error("Invalid tensor file format: expected 3+ lines")

    var name = String(lines[0])
    var metadata = String(lines[1])
    var hex_data = String(lines[2])

    var meta_parts = metadata.split(" ")
    if len(meta_parts) < 1:
        raise Error("Invalid metadata format")

    var dtype_str = meta_parts[0]
    var dtype = parse_dtype(String(dtype_str))

    var shape = List[Int]()
    for i in range(1, len(meta_parts)):
        shape.append(Int(meta_parts[i]))

    var tensor = zeros(shape, dtype)
    hex_to_bytes(hex_data, tensor)

    return Tuple[String, AnyTensor](name, tensor^)


# ============================================================================
# Hex Encoding/Decoding
# ============================================================================


def bytes_to_hex(
    data: UnsafePointer[UInt8, MutAnyOrigin], num_bytes: Int
) -> String:
    """Convert bytes to hexadecimal string.

    Args:
        data: Pointer to byte array.
        num_bytes: Number of bytes to convert.

    Returns:
        Hex string representation.
    """
    if not data:
        return ""

    var hex_chars = "0123456789abcdef"
    var result = String("")

    for i in range(num_bytes):
        var byte = Int(data[i])
        var high = (byte >> 4) & 0xF
        var low = byte & 0xF
        result += chr(Int(hex_chars.as_bytes()[high]))
        result += chr(Int(hex_chars.as_bytes()[low]))

    return result


def hex_to_bytes(hex_str: String, tensor: AnyTensor) raises:
    """Convert hexadecimal string to bytes and store in tensor.

    Args:
        hex_str: Hex string (e.g., "3f800000").
        tensor: Tensor to store decoded bytes in.

    Raises:
        Error: If hex string has odd length or contains invalid characters.
    """
    var length = len(hex_str)
    if length % 2 != 0:
        raise Error("Hex string must have even length")

    var output = tensor._data
    for i in range(0, length, 2):
        var high = _hex_char_to_int(chr(Int(hex_str.as_bytes()[i])))
        var low = _hex_char_to_int(chr(Int(hex_str.as_bytes()[i + 1])))
        var offset = i // 2
        output[offset] = UInt8((high << 4) | low)


def _hex_char_to_int(c: String) raises -> Int:
    """Convert single hex character to integer (0-15).

    Args:
        c: Single character ('0'-'9', 'a'-'f', 'A'-'F').

    Returns:
        Integer value (0-15).

    Raises:
        Error: If character is not a valid hex digit.
    """
    if c >= "0" and c <= "9":
        return ord(c) - ord("0")
    elif c >= "a" and c <= "f":
        return ord(c) - ord("a") + 10
    elif c >= "A" and c <= "F":
        return ord(c) - ord("A") + 10
    else:
        raise Error("Invalid hex character: " + c)


# ============================================================================
# DType Utilities
# ============================================================================


def get_dtype_size(dtype: DType) -> Int:
    """Get size in bytes for a dtype.

    Args:
        dtype: Data type.

    Returns:
        Size in bytes (1, 2, 4, or 8).
    """
    if dtype == DType.float16:
        return 2
    elif dtype == DType.float32:
        return 4
    elif dtype == DType.float64:
        return 8
    elif dtype == DType.int8 or dtype == DType.uint8:
        return 1
    elif dtype == DType.int16 or dtype == DType.uint16:
        return 2
    elif dtype == DType.int32 or dtype == DType.uint32:
        return 4
    elif dtype == DType.int64 or dtype == DType.uint64:
        return 8
    else:
        return 4


def parse_dtype(dtype_str: String) raises -> DType:
    """Parse dtype string to DType enum.

    Args:
        dtype_str: String representation (e.g., "float32", "int64").

    Returns:
        Corresponding DType.

    Raises:
        Error: If dtype string is not recognized.
    """
    if dtype_str == "float16":
        return DType.float16
    elif dtype_str == "float32":
        return DType.float32
    elif dtype_str == "float64":
        return DType.float64
    elif dtype_str == "int8":
        return DType.int8
    elif dtype_str == "int16":
        return DType.int16
    elif dtype_str == "int32":
        return DType.int32
    elif dtype_str == "int64":
        return DType.int64
    elif dtype_str == "uint8":
        return DType.uint8
    elif dtype_str == "uint16":
        return DType.uint16
    elif dtype_str == "uint32":
        return DType.uint32
    elif dtype_str == "uint64":
        return DType.uint64
    else:
        raise Error("Unknown dtype: " + dtype_str)


def dtype_to_string(dtype: DType) -> String:
    """Convert dtype enum to string representation.

    Args:
        dtype: Data type.

    Returns:
        String representation (e.g., "float32").
    """
    if dtype == DType.float16:
        return "float16"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    else:
        return "unknown"
