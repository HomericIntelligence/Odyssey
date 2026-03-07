"""Tensor serialization utilities for ML Odyssey.

Provides comprehensive tensor serialization with support for single tensors,
named tensor collections, and batch operations. Uses hex-encoding format
for text-based storage and efficient binary representation.

File Format (single tensor):
    Line 1: <tensor_name>
    Line 2: <dtype> <shape_dim0> <shape_dim1> ...
    Line 3+: <hex_encoded_bytes>

Example:
    conv1_kernel
    float32 6 1 5 5
    3f800000 3f800000 ... (hex-encoded float values)
    ```

Modules:
    - Tensor saving/loading (single)
    - Named tensor collections
    - DType utilities
    - Hex encoding/decoding

Example:
    from shared.utils.serialization import (
        save_tensor, load_tensor,
        save_named_tensors, load_named_tensors,
        NamedTensor,
    )

    # Save single tensor
    save_tensor(tensor, "weights.bin")

    # Save named collection
    var tensors : List[NamedTensor] = []
    tensors.append(NamedTensor("conv1_w", conv1_weights))
    tensors.append(NamedTensor("conv1_b", conv1_bias))
    save_named_tensors(tensors, "checkpoint/")
    ```
"""

from shared.core.extensor import ExTensor, zeros
from memory import UnsafePointer
from collections import List, Dict
from collections.optional import Optional
from shared.utils.file_io import create_directory
import os


# ============================================================================
# NamedTensor Structure
# ============================================================================


struct NamedTensor(Copyable, Movable):
    """Named tensor for checkpoint collections.

    Associates a human-readable name with tensor data
    Used for organizing model weights and parameters
    """

    var name: String
    var tensor: ExTensor

    fn __init__(out self, name: String, tensor: ExTensor):
        """Create named tensor.

        Args:
            name: Parameter name (e.g., "conv1_kernel", "linear_bias").
            tensor: Tensor data.
        """
        self.name = name
        self.tensor = tensor

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor for ownership transfer."""
        self.name = other.name^
        self.tensor = other.tensor^

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.name = existing.name
        self.tensor = existing.tensor


# ============================================================================
# Single Tensor Serialization
# ============================================================================


fn save_tensor(tensor: ExTensor, filepath: String, name: String = "") raises:
    """Save tensor to file in hex format.

        Saves tensor with metadata (dtype, shape) and hex-encoded byte data
        File format is text-based for portability across platforms.

    Args:
            tensor: Tensor to save.
            filepath: Output file path.
            name: Optional tensor name (defaults to empty string).

    Raises:
            Error: If file write fails.

        Example:
            ```mojo
            var weights = ExTensor(...)
            save_tensor(weights, "checkpoint/conv1.bin", "conv1_weights")
            ```
    """
    # Create local copy to ensure stable data pointer
    # This prevents issues when tensor is accessed through List[NamedTensor]
    var local_tensor = tensor

    var shape = local_tensor.shape()
    var dtype = local_tensor.dtype()
    var numel = local_tensor.numel()

    # Build metadata line: dtype and shape dimensions
    var dtype_str = dtype_to_string(dtype)
    var metadata = dtype_str + " "
    for i in range(len(shape)):
        metadata += String(shape[i])
        if i < len(shape) - 1:
            metadata += " "

    # Convert tensor data to hex
    var dtype_size = get_dtype_size(dtype)
    var total_bytes = numel * dtype_size
    var hex_data = bytes_to_hex(local_tensor._data, total_bytes)

    # Write to file
    with open(filepath, "w") as f:
        _ = f.write(name + "\n")
        _ = f.write(metadata + "\n")
        _ = f.write(hex_data + "\n")


fn load_tensor(filepath: String) raises -> ExTensor:
    """Load tensor from file.

        Reads hex-encoded tensor data and metadata, reconstructs
        ExTensor with original dtype and shape.

    Args:
            filepath: Input file path.

    Returns:
            Loaded ExTensor.

    Raises:
            Error: If file format is invalid or file doesn't exist.

        Example:
            ```mojo
            var tensor = load_tensor("checkpoint/conv1.bin")
            ```
    """
    # Read file
    var content: String
    with open(filepath, "r") as f:
        content = f.read()

    # Parse lines
    var lines = content.split("\n")
    if len(lines) < 3:
        raise Error("Invalid tensor file format: expected 3+ lines")

    var _ = String(lines[0])  # Name not used in loading (stored for info)
    var metadata = String(lines[1])
    var hex_data = String(lines[2])

    # Parse metadata: dtype shape_dims...
    var meta_parts = metadata.split(" ")
    if len(meta_parts) < 1:
        raise Error("Invalid metadata format: expected dtype and shape")

    var dtype_str = meta_parts[0]
    var dtype = parse_dtype(String(dtype_str))

    # Parse shape
    var shape = List[Int]()
    for i in range(1, len(meta_parts)):
        shape.append(Int(meta_parts[i]))

    # Create tensor with zeros, then fill with data
    var tensor = zeros(shape, dtype)

    # Convert hex to bytes and write into tensor
    hex_to_bytes(hex_data, tensor)

    return tensor^


fn load_tensor_with_name(filepath: String) raises -> Tuple[String, ExTensor]:
    """Load tensor with its associated name.

        Similar to load_tensor but also returns the tensor name
        from the file (useful for NamedTensor reconstruction).

    Args:
            filepath: Input file path.

    Returns:
            Tuple of (name, tensor).

    Raises:
            Error: If file format is invalid or file doesn't exist.

        Example:
            ```mojo
            var (name, tensor) = load_tensor_with_name("checkpoint/conv1.bin")
            ```
    """
    # Read file
    var content: String
    with open(filepath, "r") as f:
        content = f.read()

    # Parse lines
    var lines = content.split("\n")
    if len(lines) < 3:
        raise Error("Invalid tensor file format: expected 3+ lines")

    var name = String(lines[0])
    var metadata = String(lines[1])
    var hex_data = String(lines[2])

    # Parse metadata
    var meta_parts = metadata.split(" ")
    if len(meta_parts) < 1:
        raise Error("Invalid metadata format")

    var dtype_str = meta_parts[0]
    var dtype = parse_dtype(String(dtype_str))

    # Parse shape
    var shape = List[Int]()
    for i in range(1, len(meta_parts)):
        shape.append(Int(meta_parts[i]))

    # Create tensor
    var tensor = zeros(shape, dtype)

    # Convert hex to bytes
    hex_to_bytes(hex_data, tensor)

    return Tuple[String, ExTensor](name, tensor^)


# ============================================================================
# Named Tensor Collection Serialization
# ============================================================================


fn save_named_tensors(tensors: List[NamedTensor], dirpath: String) raises:
    """Save collection of named tensors to directory.

        Creates a directory with one .weights file per tensor
        Useful for saving model checkpoints with multiple parameter groups.

    Args:
            tensors: List of NamedTensor objects.
            dirpath: Output directory path (created if doesn't exist).

    Raises:
            Error: If directory creation or file write fails.

        Example:
            ```mojo
            var tensors : List[NamedTensor] = []
            tensors.append(NamedTensor("conv1_w", conv1_weights))
            tensors.append(NamedTensor("conv1_b", conv1_bias))
            save_named_tensors(tensors, "checkpoint/epoch_10/")
            ```
    """
    # Create directory if needed
    if not create_directory(dirpath):
        raise Error("Failed to create directory: " + dirpath)

    # Save each tensor
    for i in range(len(tensors)):
        var filename = tensors[i].name + ".weights"
        var filepath = dirpath + "/" + filename

        save_tensor(tensors[i].tensor, filepath, tensors[i].name)


fn load_named_tensors(dirpath: String) raises -> List[NamedTensor]:
    """Load collection of named tensors from directory.

        Reads all .weights files from directory and reconstructs
        NamedTensor objects. Files are loaded in directory order.

    Args:
            dirpath: Directory containing .weights files.

    Returns:
            List of NamedTensor objects.

    Raises:
            Error: If directory doesn't exist or file format is invalid.

        Example:
            ```mojo
            var tensors = load_named_tensors("checkpoint/epoch_10/")
            for i in range(len(tensors)):
                print(tensors[i].name)
            ```
    """
    var result: List[NamedTensor] = []

    try:
        # List directory contents using Mojo native os.listdir
        var entries = os.listdir(dirpath)

        # Collect .weights files and sort for deterministic ordering
        var weight_files: List[String] = []
        for i in range(len(entries)):
            var entry = entries[i]
            if entry.endswith(".weights"):
                weight_files.append(entry)

        # Sort filenames for deterministic ordering (insertion sort)
        for i in range(1, len(weight_files)):
            var key = weight_files[i]
            var j = i - 1
            while j >= 0 and weight_files[j] > key:
                weight_files[j + 1] = weight_files[j]
                j -= 1
            weight_files[j + 1] = key

        # Load each weights file
        for i in range(len(weight_files)):
            var filepath = dirpath + "/" + weight_files[i]
            var (name, tensor) = load_tensor_with_name(filepath)
            result.append(NamedTensor(name, tensor))

    except:
        raise Error("Failed to load tensors from: " + dirpath)

    return result^


# ============================================================================
# Checkpoint Serialization (with optional metadata)
# ============================================================================


fn save_named_checkpoint(
    tensors: List[NamedTensor],
    path: String,
    metadata: Optional[Dict[String, String]] = None,
) raises:
    """Save model checkpoint with named tensors and optional metadata.

        Creates checkpoint directory with tensor files and metadata file
        Metadata is stored in a separate JSON-like format.

    Args:
            tensors: List of NamedTensor objects to save.
            path: Checkpoint directory path (created if doesn't exist).
            metadata: Optional metadata dictionary (e.g., epoch, loss values).

    Raises:
            Error: If directory creation or file write fails.

        Example:
            ```mojo
            var tensors : List[NamedTensor] = []
            tensors.append(NamedTensor("weights", weights_tensor))
            tensors.append(NamedTensor("bias", bias_tensor))
            var meta = Dict[String, String]()
            meta["epoch"] = "10"
            meta["loss"] = "0.45"
            save_checkpoint(tensors, "checkpoints/model/", meta)
            ```
    """
    # Create checkpoint directory
    if not create_directory(path):
        raise Error("Failed to create checkpoint directory: " + path)

    # Save all named tensors
    save_named_tensors(tensors, path)

    # Save metadata if provided
    if metadata:
        var meta_path = path + "/metadata.txt"
        var meta_content = _serialize_metadata(metadata.value())
        with open(meta_path, "w") as f:
            _ = f.write(meta_content)


fn load_named_checkpoint(
    path: String,
) raises -> Tuple[List[NamedTensor], Dict[String, String]]:
    """Load model checkpoint with named tensors and metadata.

        Reads all tensor files from checkpoint directory and metadata if present
        Returns both the tensors and any associated metadata.

    Args:
            path: Checkpoint directory path.

    Returns:
            Tuple of (tensors, metadata).

    Raises:
            Error: If directory doesn't exist or file format is invalid.

        Example:
            ```mojo
            var (tensors, metadata) = load_checkpoint("checkpoints/model/")
            for i in range(len(tensors)):
                print(tensors[i].name)
            if "epoch" in metadata:
                print("Epoch: " + metadata["epoch"])
            ```
    """
    # Load all named tensors
    var tensors = load_named_tensors(path)

    # Load metadata if it exists
    var metadata = Dict[String, String]()
    var meta_path = path + "/metadata.txt"

    try:
        var meta_content: String
        with open(meta_path, "r") as f:
            meta_content = f.read()
        metadata = _deserialize_metadata(meta_content)
    except:
        # Metadata file not found, return empty metadata
        pass

    return Tuple[List[NamedTensor], Dict[String, String]](tensors^, metadata^)


fn _serialize_metadata(metadata: Dict[String, String]) raises -> String:
    """Serialize metadata dictionary to text format.

        Format: one key=value pair per line

    Args:
            metadata: Dictionary to serialize

    Returns:
            Serialized string
    """
    var lines = List[String]()

    for key_ref in metadata.keys():
        var k = String(key_ref)
        var v = String(metadata[k])
        lines.append(k + "=" + v)

    # Join lines
    var result = String("")
    for i in range(len(lines)):
        if i > 0:
            result += "\n"
        result += lines[i]

    return result


fn _deserialize_metadata(content: String) raises -> Dict[String, String]:
    """Deserialize metadata from text format.

    Args:
            content: Serialized metadata string

    Returns:
            Metadata dictionary

    Raises:
            Error: If format is invalid
    """
    var metadata = Dict[String, String]()
    var lines = content.split("\n")

    for i in range(len(lines)):
        var line = lines[i].strip()
        if len(line) == 0:
            continue

        # Find key=value separator
        var eq_pos = line.find("=")
        if eq_pos == -1:
            continue  # Skip malformed lines.

        var key = String(line[:eq_pos])
        var value = String(line[eq_pos + 1 :])
        metadata[key] = value

    return metadata^


# ============================================================================
# Hex Encoding/Decoding
# ============================================================================


fn bytes_to_hex(data: UnsafePointer[UInt8], num_bytes: Int) -> String:
    """Convert bytes to hexadecimal string.

        Encodes each byte as two hex characters (e.g., 0xFF -> "ff")
        Used for text-based tensor serialization.

    Args:
            data: Pointer to byte array.
            num_bytes: Number of bytes to convert.

    Returns:
            Hex string representation.

        Example:
            ```mojo
            var hex_str = bytes_to_hex(tensor._data, 16)
            # Returns "3f800000..." for float32 values
            ```
    """
    # Defensive null check for pointer safety
    if not data:
        return ""

    var hex_chars = "0123456789abcdef"
    var result = String("")

    for i in range(num_bytes):
        var byte = Int(data[i])
        var high = (byte >> 4) & 0xF
        var low = byte & 0xF
        result += hex_chars[high]
        result += hex_chars[low]

    return result


fn hex_to_bytes(hex_str: String, tensor: ExTensor) raises:
    """Convert hexadecimal string to bytes and store in tensor.

        Decodes hex string back to bytes. Validates that hex string
        has even length (pairs of hex digits).

    Args:
            hex_str: Hex string (e.g., "3f800000").
            tensor: Tensor to store decoded bytes in.

    Raises:
            Error: If hex string has odd length or contains invalid characters.

        Example:
            ```mojo
            var hex_str = "3f800000"
            var tensor = zeros(shape, DType.float32)
            hex_to_bytes(hex_str, tensor)
            ```
    """
    var length = len(hex_str)
    if length % 2 != 0:
        raise Error("Hex string must have even length")

    var output = tensor._data
    for i in range(0, length, 2):
        var high = _hex_char_to_int(String(hex_str[i]))
        var low = _hex_char_to_int(String(hex_str[i + 1]))
        var offset = i // 2
        output[offset] = UInt8((high << 4) | low)


fn _hex_char_to_int(c: String) raises -> Int:
    """Convert single hex character to integer.

        Internal helper for hex decoding

    Args:
            c: Single character ('0'-'9', 'a'-'f', 'A'-'F')

    Returns:
            Integer value (0-15)

    Raises:
            Error: If character is not a valid hex digit
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


fn get_dtype_size(dtype: DType) -> Int:
    """Get size in bytes for a dtype.

        Returns the number of bytes required to store a single
        value of the given dtype. Used for calculating tensor
        byte sizes during serialization.

    Args:
            dtype: Data type.

    Returns:
            Size in bytes (1, 2, 4, or 8).

        Example:
            ```mojo
            var size = get_dtype_size(DType.float32)  # Returns 4
            ```
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
        return 4  # Default to 4 bytes


fn parse_dtype(dtype_str: String) raises -> DType:
    """Parse dtype string to DType enum.

        Converts string representation (e.g., "float32") to corresponding
        DType enum value. Case-sensitive match required.

    Args:
            dtype_str: String representation (e.g., "float32", "int64").

    Returns:
            Corresponding DType.

    Raises:
            Error: If dtype string is not recognized.

        Example:
            ```mojo
            var dtype = parse_dtype("float32")  # Returns DType.float32
            ```
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


fn dtype_to_string(dtype: DType) -> String:
    """Convert dtype enum to string representation.

    Args:
            dtype: Data type.

    Returns:
            String representation (e.g., "float32").

        Example:
            ```mojo
            var s = dtype_to_string(DType.float32)  # Returns "float32"
            ```
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
