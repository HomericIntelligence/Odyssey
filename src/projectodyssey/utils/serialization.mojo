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
    from projectodyssey.utils.serialization import (
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

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.tensor.tensor_io import (
    save_tensor,
    load_tensor,
    load_tensor_with_name,
    bytes_to_hex,
    hex_to_bytes,
    _hex_char_to_int,
    get_dtype_size,
    parse_dtype,
    dtype_to_string,
)
from std.memory import UnsafePointer
from std.collections import List, Dict
from std.collections.optional import Optional
from std import os
from std.os import mkdir


# ============================================================================
# Internal Utilities
# ============================================================================


def _create_directory(dirpath: String) -> Bool:
    """Create directory if it doesn't exist using native Mojo os.mkdir.

    Avoids Python FFI (which trips the Mojo FFI safety check under ASAN).
    Uses std.os.mkdir which handles single-level directories. If the directory
    already exists, the error is silently ignored (idempotent behavior).

    Args:
        dirpath: Directory path to create.

    Returns:
        True if created or already exists, False if an unexpected error occurs.
    """
    try:
        mkdir(dirpath)
    except:
        # Ignore "already exists" and other non-fatal errors.
        # We verify success by attempting to list the directory below.
        pass
    try:
        _ = os.listdir(dirpath)
        return True
    except:
        return False


# ============================================================================
# NamedTensor Structure
# ============================================================================


struct NamedTensor(Copyable, Movable):
    """Named tensor for checkpoint collections.

    Associates a human-readable name with tensor data
    Used for organizing model weights and parameters
    """

    var name: String
    var tensor: AnyTensor

    def __init__(out self, name: String, tensor: AnyTensor):
        """Create named tensor.

        Args:
            name: Parameter name (e.g., "conv1_kernel", "linear_bias").
            tensor: Tensor data.
        """
        self.name = name
        self.tensor = tensor

    def __init__(out self, *, copy: Self):
        """Copy constructor."""
        self.name = copy.name
        self.tensor = copy.tensor


# ============================================================================
# Named Tensor Collection Serialization
# ============================================================================


def save_named_tensors(tensors: List[NamedTensor], dirpath: String) raises:
    """Save collection of named tensors to directory.

        Creates a directory with one .weights file per tensor
        Useful for saving model checkpoints with multiple parameter groups.

    Args:
            tensors: List of NamedTensor objects.
            dirpath: Output directory path (created if doesn't exist).
                     Trailing slash is optional (e.g., `"checkpoint/"` or `"checkpoint"`).

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
    # Normalize dirpath: remove trailing slash if present
    var normalized_dirpath = dirpath
    if dirpath.endswith("/"):
        normalized_dirpath = String(
            dirpath[byte = 0 : dirpath.byte_length() - 1]
        )

    # Create directory if needed
    if not _create_directory(normalized_dirpath):
        raise Error("Failed to create directory: " + normalized_dirpath)

    # Save each tensor
    for i in range(len(tensors)):
        var filename = tensors[i].name + ".weights"
        var filepath = normalized_dirpath + "/" + filename

        save_tensor(tensors[i].tensor, filepath, tensors[i].name)


def load_named_tensors(dirpath: String) raises -> List[NamedTensor]:
    """Load collection of named tensors from directory.

        Reads all .weights files from directory and reconstructs
        NamedTensor objects. Files are loaded in directory order.

    Args:
            dirpath: Directory containing `.weights` files.
                     Trailing slash is optional (e.g., `"checkpoint/"` or `"checkpoint"`).

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
        # Normalize dirpath: remove trailing slash if present
        var normalized_dirpath = dirpath
        if dirpath.endswith("/"):
            normalized_dirpath = String(
                dirpath[byte = 0 : dirpath.byte_length() - 1]
            )

        # List directory contents using Mojo native os.listdir
        var entries = os.listdir(normalized_dirpath)

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
            var filepath = normalized_dirpath + "/" + weight_files[i]
            var (name, tensor) = load_tensor_with_name(filepath)
            result.append(NamedTensor(name, tensor))

    except e:
        raise Error("Failed to load tensors from: " + dirpath)

    return result^


# ============================================================================
# Checkpoint Serialization (with optional metadata)
# ============================================================================


def save_named_checkpoint(
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
    # Normalize path: remove trailing slash if present
    var normalized_path = path
    if path.endswith("/"):
        normalized_path = String(path[byte = 0 : path.byte_length() - 1])

    # Create checkpoint directory
    if not _create_directory(normalized_path):
        raise Error("Failed to create checkpoint directory: " + normalized_path)

    # Save all named tensors
    save_named_tensors(tensors, normalized_path)

    # Save metadata if provided
    if metadata:
        var meta_path = normalized_path + "/metadata.txt"
        var meta_content = _serialize_metadata(metadata.value())
        with open(meta_path, "w") as f:
            _ = f.write(meta_content)


def load_named_checkpoint(
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
    # Normalize path: remove trailing slash if present
    var normalized_path = path
    if path.endswith("/"):
        normalized_path = String(path[byte = 0 : path.byte_length() - 1])

    # Load all named tensors
    var tensors = load_named_tensors(normalized_path)

    # Load metadata if it exists
    var metadata = Dict[String, String]()
    var meta_path = normalized_path + "/metadata.txt"

    try:
        var meta_content: String
        with open(meta_path, "r") as f:
            meta_content = f.read()
        metadata = _deserialize_metadata(meta_content)
    except:
        # Metadata file not found, return empty metadata
        pass

    return Tuple[List[NamedTensor], Dict[String, String]](tensors^, metadata^)


def _serialize_metadata(metadata: Dict[String, String]) raises -> String:
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


def _deserialize_metadata(content: String) raises -> Dict[String, String]:
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
        if line.byte_length() == 0:
            continue

        # Find key=value separator
        var eq_pos = line.find("=")
        if eq_pos == -1:
            continue  # Skip malformed lines.

        var key = String(line[byte=0:eq_pos])
        var value = String(line[byte = eq_pos + 1 : line.byte_length()])
        metadata[key] = value

    return metadata^
